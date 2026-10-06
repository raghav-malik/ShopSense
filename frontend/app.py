"""ShopSense chat UI. Talks only to the FastAPI backend; all tracing is server-side.

Run:  streamlit run frontend/app.py   (backend: uvicorn app.main:app --port 8000)
"""

import contextlib
import os
import sys
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
import streamlit as st

# `streamlit run frontend/app.py` puts only frontend/ on the path; the UI still never imports `app`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from frontend import timefmt

API_BASE = os.getenv("SHOPSENSE_API_URL", "http://localhost:8000")
# An agent turn is several LLM calls plus web searches (~20-45s measured, up to
# 10 steps worst case), so the spec's 60s timeout cut off slow-but-healthy turns.
CHAT_TIMEOUT = 180.0
# One small-model call (2-5s on gemma4:31b), but it can queue behind other
# requests on a rate-limited plan.
SUMMARIZE_TIMEOUT = 60.0
MAX_PRODUCT_CARDS = 3

# The backend's JSON bodies. The UI reads them as plain dicts, like the API returns them.
type JSONObject = dict[str, Any]

st.set_page_config(page_title="ShopSense", page_icon="🛍️", layout="wide")


# ---- Backend calls ----


class ApiError(Exception):
    """A backend error, with the API's error code and the Langfuse trace link when there is one."""

    def __init__(self, message: str, code: str | None = None, trace_url: str | None = None):
        super().__init__(message)
        self.code = code
        self.trace_url = trace_url


def _call(method: str, path: str, *, timeout: float = 15.0, **kwargs: Any) -> JSONObject:
    try:
        response = httpx.request(method, f"{API_BASE}{path}", timeout=timeout, **kwargs)
    except httpx.ConnectError as e:
        raise ApiError(
            "Cannot connect to the backend. Make sure `uvicorn app.main:app --port 8000` is running.", "backend_down"
        ) from e
    except httpx.TimeoutException as e:
        raise ApiError("The assistant took too long to respond. Please try again.", "timeout") from e
    if response.is_error:
        # The backend always answers {"error": {"code", "message"}}, plus
        # trace_url when an agent run failed.
        try:
            error = response.json()["error"]
            message, code = error["message"], error["code"]
        except (ValueError, KeyError, TypeError) as e:
            raise ApiError(f"API error {response.status_code}", f"http_{response.status_code}") from e
        raise ApiError(message, code, error.get("trace_url"))
    if not response.content:  # 204 No Content, e.g. after a delete
        return {}
    body: JSONObject = response.json()
    return body


def create_session() -> str:
    """Create a new session via the API."""
    session_id: str = _call("POST", "/sessions")["session_id"]
    return session_id


def send_message(session_id: str, message: str) -> JSONObject:
    """Send a message to the agent."""
    return _call("POST", f"/sessions/{session_id}/chat", json={"message": message}, timeout=CHAT_TIMEOUT)


def get_cart(session_id: str) -> JSONObject:
    """Fetch current cart."""
    return _call("GET", f"/sessions/{session_id}/cart")


def get_suggestions(session_id: str) -> list[str]:
    """Follow-up suggestions for the latest answer; best-effort."""
    try:
        suggestions: list[str] = _call("POST", f"/sessions/{session_id}/suggestions")["suggestions"]
    except ApiError:
        return []
    return suggestions


def summarize_session(session_id: str) -> None:
    """Save a summary of the session for future chats' memory; best-effort.

    If it fails, the backend summarizes the session later anyway, when the next
    new session starts."""
    with contextlib.suppress(ApiError):
        _call("POST", f"/sessions/{session_id}/summarize", timeout=SUMMARIZE_TIMEOUT)


def get_history(session_id: str) -> JSONObject:
    """The session's saved messages and details."""
    return _call("GET", f"/sessions/{session_id}/history")


def list_chats(query: str = "", limit: int = 30) -> JSONObject:
    """The user's chats, most recently active first ({"items", "total"})."""
    return _call("GET", "/sessions", params={"q": query, "limit": limit})


def rename_chat(chat_id: str, title: str) -> None:
    """Give a chat a new title."""
    _call("PATCH", f"/sessions/{chat_id}", json={"title": title})


def delete_chat(chat_id: str) -> None:
    """Remove a chat from the list."""
    _call("DELETE", f"/sessions/{chat_id}")


def load_messages(chat_id: str) -> list[JSONObject]:
    """A chat's user and assistant messages, to show it again."""
    history = get_history(chat_id)
    return [
        {"role": m["role"], "content": m["content"]} for m in history["messages"] if m["role"] in ("user", "assistant")
    ]


def get_memory_files() -> JSONObject:
    """Every memory file ({"timezone", "files"}): user.md, memory.md, preferences.md, then the day files."""
    return _call("GET", "/memory/files")


def save_memory_file(name: str, content: str) -> None:
    """Save an edited memory file."""
    _call("PUT", f"/memory/files/{quote(name, safe='')}", json={"content": content})


def clear_memory_file(name: str) -> None:
    """Clear a memory file (user.md goes back to its template)."""
    _call("DELETE", f"/memory/files/{quote(name, safe='')}")


@st.cache_data(ttl=300, show_spinner=False)
def get_backend_info() -> JSONObject | None:
    """Model name and Langfuse dashboard link, as configured on the backend."""
    try:
        return _call("GET", "/health", timeout=5.0)
    except ApiError:
        return None


# ---- Session state ----


def start_session(session_id: str, messages: list[JSONObject] | None = None) -> None:
    """Make `session_id` the current session and put it in the URL."""
    st.session_state.session_id = session_id
    st.session_state.messages = messages or []
    st.session_state.suggestions = []
    # In the URL, so a page refresh reopens this conversation instead of
    # silently starting a new one.
    st.query_params["session"] = session_id


def restore_or_create_session() -> None:
    """Reopen the session in the URL (e.g. after a refresh), or create a new one."""
    session_id = st.query_params.get("session")
    if session_id:
        try:
            history = get_history(session_id)
        except ApiError as e:
            if e.code != "session_not_found":
                raise
        else:
            messages = [
                {"role": m["role"], "content": m["content"]}
                for m in history["messages"]
                if m["role"] in ("user", "assistant")
            ]
            start_session(session_id, messages)
            return
    start_session(create_session())


if "session_id" not in st.session_state:
    try:
        restore_or_create_session()
    except ApiError as e:
        st.title("🛍️ ShopSense")
        st.error(str(e))
        st.stop()

session_id = st.session_state.session_id


# ---- Rendering helpers ----


def pick_product_cards(result: JSONObject) -> tuple[list[JSONObject], bool]:
    """Cards for the products the agent actually linked in its answer.

    products_found holds every search result (often roundup articles), so
    labelling the first few "Buy Now" would be misleading. Falls back to the
    top search results, labelled as such, when the answer links none of them.
    """
    products = result.get("products_found", [])
    answer = result.get("response", "")
    cited, seen = [], set()
    for product in products:
        url = product.get("url")
        if url and url in answer and url not in seen:
            cited.append(product)
            seen.add(url)
    if cited:
        return cited[:MAX_PRODUCT_CARDS], True
    return products[:MAX_PRODUCT_CARDS], False


def _short(text: str, limit: int = 90) -> str:
    # ddgs joins the titles when several engines return the same page, which can
    # produce a paragraph-long "title".
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + " …"


def price_badge(check: JSONObject) -> str:
    """How far to trust a card's price: checked on the store page, or only from search results."""
    store, live, stated = check["store"], check.get("live_price"), check.get("stated_price")
    if check["status"] == "verified" and live is not None:
        return f"✓ ₹{live:,.0f} · price checked on {store} just now"
    if check["status"] == "corrected" and live is not None:
        return f"✓ ₹{live:,.0f} · price checked on {store} just now (search results said ₹{stated:,.0f})"
    if check["status"] == "unavailable":
        return f"⚠ Currently unavailable on {store}"
    if check["status"] == "search_page":
        return f"This link is a search page on {store}, not a single product"
    return "Price from search results; it couldn't be checked on the store"


def render_assistant_extras(message: JSONObject, index: int) -> None:
    """Product cards (with how their price was checked) and the Langfuse debug panel under an answer."""
    checks = {check["url"]: check for check in message.get("price_checks", [])}
    for n, product in enumerate(message.get("products", [])):
        icon = "🛒" if message.get("products_cited") else "🔎"
        check = checks.get(product.get("url"))
        with st.expander(f"{icon} {_short(product.get('title') or 'Product')}"):
            if check:
                st.markdown(f"**{price_badge(check)}**")
            if product.get("snippet"):
                # The snippet is the search engine's copy, so its price can be out of date.
                st.write(product["snippet"])
            if product.get("source"):
                st.caption(product["source"] + (" · search result text" if check else ""))
            if product.get("url"):
                if check and check["status"] == "search_page":
                    label = f"Search on {check['store']} →"
                else:
                    label = "Buy Now →" if message.get("products_cited") else "View result →"
                # Keyed: the same product can appear in more than one answer.
                st.link_button(label, product["url"], key=f"buy_{index}_{n}")

    # Every product link in the answer, with its price as checked on the store page just now.
    price_checks = message.get("price_checks", [])
    if price_checks:
        changed = any(c["status"] in ("corrected", "unavailable") for c in price_checks)
        with st.expander("🏷️ Prices checked on the stores" + (" (some changed)" if changed else "")):
            for check in price_checks:
                st.markdown(f"- [{check.get('product') or check['store']}]({check['url']}): {price_badge(check)}")

    meta = message.get("meta")
    if meta and meta.get("trace_url"):
        with st.expander("🔍 Debug: Langfuse Trace"):
            st.write(f"[View trace]({meta['trace_url']})")
            cost = meta.get("estimated_cost_usd")
            cost_text = f", Est. cost: ${cost:.4f}" if cost is not None else ""
            st.write(f"Steps: {meta['step_count']}, Tokens: {meta['total_tokens']}{cost_text}")
            st.write(f"Tools: {', '.join(meta['tool_calls_made']) or 'none'}")


def render_message(message: JSONObject, index: int) -> None:
    """One chat message (errors in red, with a trace link)."""
    with st.chat_message(message["role"]):
        if message.get("error"):
            st.error(message["content"])
            if message.get("trace_url"):
                st.caption(f"[Debug this in Langfuse]({message['trace_url']})")
        else:
            st.markdown(message["content"])
        if message["role"] == "assistant":
            render_assistant_extras(message, index)


# ---- Sidebar: past chats ----

CHAT_PAGE = 30


def _timezone() -> str:
    return (backend_info or {}).get("timezone") or timefmt.DEFAULT_TIMEZONE


def open_chat(chat_id: str) -> None:
    """Button callback: reopen a past chat where it left off."""
    if chat_id == st.session_state.session_id:
        return
    try:
        start_session(chat_id, load_messages(chat_id))
    except ApiError as e:
        st.session_state.sidebar_error = str(e)


def new_chat() -> None:
    """Button callback: summarize the current chat for memory, then start a new one."""
    if not st.session_state.messages:
        return  # already a new chat
    summarize_session(st.session_state.session_id)
    try:
        start_session(create_session())
    except ApiError as e:
        st.session_state.sidebar_error = str(e)


def rename_chat_callback(chat_id: str) -> None:
    """Button callback: save the title typed in the chat's ⋮ menu."""
    title = st.session_state.get(f"rename_{chat_id}", "").strip()
    if not title:
        return
    try:
        rename_chat(chat_id, title)
    except ApiError as e:
        st.session_state.sidebar_error = str(e)


def ask_delete_chat(chat_id: str, title: str) -> None:
    """Button callback: open the delete confirmation for a chat."""
    st.session_state.delete_chat = {"id": chat_id, "title": title}


def _close_delete_dialog() -> None:
    st.session_state.pop("delete_chat", None)


def _confirm_delete(chat_id: str) -> None:
    try:
        delete_chat(chat_id)
    except ApiError as e:
        st.session_state.sidebar_error = str(e)
    st.session_state.pop("delete_chat", None)
    if chat_id == st.session_state.session_id:
        st.session_state.messages = []
        try:
            start_session(create_session())
        except ApiError as e:
            st.session_state.sidebar_error = str(e)


@st.dialog("Delete chat?", on_dismiss=_close_delete_dialog)
def delete_chat_dialog() -> None:
    """Confirm before a chat is deleted."""
    target = st.session_state.get("delete_chat") or {}
    st.write(f"Delete **{target.get('title', 'this chat')}** from your chats?")
    st.caption("It's removed from your chats. What I learned from it stays in Settings, under Memory.")
    cancel, confirm = st.columns(2)
    cancel.button("Cancel", width="stretch", on_click=_close_delete_dialog)
    confirm.button("Delete", type="primary", width="stretch", on_click=_confirm_delete, args=(target.get("id"),))


def render_chat_list() -> None:
    """Search, then every past chat, most recently active first."""
    query = st.text_input("Search chats", placeholder="🔎 Search chats", label_visibility="collapsed")
    limit = st.session_state.get("chat_limit", CHAT_PAGE)
    try:
        page = list_chats(query, limit)
    except ApiError as e:
        st.caption(f"Chats unavailable: {e}")
        return
    st.caption("Recent" if not query else f"{page['total']} matching")
    if not page["items"]:
        st.caption("No chats yet." if not query else "No chats match.")
    tz = _timezone()
    for chat in page["items"]:
        current = chat["id"] == st.session_state.session_id
        label = f"{_short(chat['title'], 34)}  \n:gray[{timefmt.relative_time(chat['updated_at'], tz)}]"
        title_column, menu_column = st.columns([7, 1], vertical_alignment="center")
        title_column.button(
            label,
            key=f"chat_{chat['id']}",
            type="primary" if current else "tertiary",
            width="stretch",
            help=chat["title"],
            on_click=open_chat,
            args=(chat["id"],),
        )
        with menu_column.popover("⋮", help="Rename or delete"):
            st.text_input("Rename", value=chat["title"], key=f"rename_{chat['id']}", max_chars=100)
            st.button("Save name", key=f"save_name_{chat['id']}", on_click=rename_chat_callback, args=(chat["id"],))
            st.button(
                "🗑 Delete chat",
                key=f"delete_{chat['id']}",
                on_click=ask_delete_chat,
                args=(chat["id"], chat["title"]),
            )
    if page["total"] > len(page["items"]) and st.button(f"Show more ({page['total'] - len(page['items'])})"):
        st.session_state.chat_limit = limit + CHAT_PAGE
        st.rerun()


# ---- Settings: You and Memory ----

_FILE_HELP = {
    "user.md": "About you, in your own words. I read it in every chat and never change it myself.",
    "memory.md": (
        'What I\'ve learned about you, one fact per line starting with "- ", under a heading '
        "(Brands you like, Brands you avoid, Budget, Stores, Sizes, Interests, Shopping style, Product feedback, "
        'Other). Delete a line to make me forget it, or add your own. "(inferred)" marks things you didn\'t say '
        "outright."
    ),
    "preferences.md": 'Preferences you asked me to save, one "- key: value" line each.',
    "short_term": (
        "The summaries of that day's chats. Edit a summary to correct it, or delete a chat's section "
        "(from its ## heading) to make me forget it. Keep each <!-- chat … --> line as it is."
    ),
}


def _close_settings() -> None:
    st.session_state.show_settings = False
    st.session_state.pop("confirm_clear", None)


def _save_file_callback(name: str, key: str, original: str) -> None:
    if st.session_state[key] == original:
        st.session_state.settings_notice = ("info", f"No changes to {name}.")
        return
    try:
        save_memory_file(name, st.session_state[key])
        st.session_state.settings_notice = ("success", f"Saved {name}.")
    except ApiError as e:
        st.session_state.settings_notice = ("error", str(e))


def _clear_file_callback(name: str) -> None:
    try:
        clear_memory_file(name)
        st.session_state.settings_notice = ("success", f"Cleared {name}." if name != "user.md" else "Reset user.md.")
    except ApiError as e:
        st.session_state.settings_notice = ("error", str(e))
    st.session_state.pop("confirm_clear", None)


def _set_confirm_clear(name: str | None) -> None:
    st.session_state.confirm_clear = name


def file_editor(file: JSONObject, tz: str, clear_label: str, confirm: tuple[str, str]) -> None:
    """One memory file: its name and "Updated …", a markdown editor, Save, and Clear behind a confirmation."""
    name = file["name"]
    help_text = _FILE_HELP.get(name) or _FILE_HELP["short_term"]
    st.markdown(f"**{name}** &nbsp; :gray[{timefmt.updated_label(file['updated_at'], tz)}]")
    st.caption(help_text)
    # Keyed by the file's version, so the editor shows the saved text after a save or a change elsewhere.
    key = f"edit_{name}_{file['updated_at']}"
    st.text_area(name, value=file["content"], height=320, key=key, label_visibility="collapsed")
    if st.session_state.get("confirm_clear") == name:
        title, body = confirm
        st.warning(f"**{title}** {body}")
        cancel, yes = st.columns(2)
        cancel.button("Cancel", key=f"cancel_clear_{name}", width="stretch", on_click=_set_confirm_clear, args=(None,))
        yes.button(
            clear_label,
            key=f"yes_clear_{name}",
            type="primary",
            width="stretch",
            on_click=_clear_file_callback,
            args=(name,),
        )
        return
    left, right = st.columns(2)
    left.button(clear_label, key=f"clear_{name}", width="stretch", on_click=_set_confirm_clear, args=(name,))
    # Always enabled: Streamlit registers an edit only when the box loses focus, so a
    # Save that's disabled until then would swallow the first click.
    right.button(
        "Save",
        key=f"save_{name}",
        type="primary",
        width="stretch",
        on_click=_save_file_callback,
        args=(name, key, file["content"]),
    )


# Opened through session state, so it stays open across reruns until dismissed;
# its buttons act in on_click callbacks, which run before the rerun.
@st.dialog("Settings", width="large", on_dismiss=_close_settings)
def settings_dialog() -> None:
    """Settings: "You" (user.md) and "Memory" (long term, preferences, short term)."""
    if notice := st.session_state.pop("settings_notice", None):
        {"success": st.success, "info": st.info}.get(notice[0], st.error)(notice[1])
    try:
        memory = get_memory_files()
    except ApiError as e:
        st.error(str(e))
        return
    tz = memory["timezone"]
    files = {f["name"]: f for f in memory["files"]}
    days = [f for f in memory["files"] if f["kind"] == "short_term"]

    you, memory_tab = st.tabs(["You", "Memory"])
    with you:
        file_editor(files["user.md"], tz, "Reset", ("Start user.md over?", "What you wrote here will be removed."))
    with memory_tab:
        long_term, preferences, short_term = st.tabs(["Long term", "Preferences", "Short term"])
        with long_term:
            file_editor(
                files["memory.md"], tz, "Clear memory", ("Forget everything in memory.md?", "This can't be undone.")
            )
        with preferences:
            file_editor(
                files["preferences.md"],
                tz,
                "Clear preferences",
                ("Forget all saved preferences?", "This can't be undone."),
            )
        with short_term:
            if not days:
                st.info("No chat summaries yet. Each chat is summarized when you start a new one.")
            else:
                labels = {f["name"]: timefmt.day_label(f["day"], tz) for f in days}
                picked = st.selectbox("Day", list(labels), format_func=labels.__getitem__, key="short_term_day")
                file_editor(
                    files[picked], tz, "Clear this day", ("Clear this day?", "I'll forget these chat summaries.")
                )
    st.caption(f"Times are in {tz}. Memory files are saved in the ShopSense database on this computer.")


# ---- Sidebar ----

backend_info = get_backend_info()

with st.sidebar:
    st.title("🛍️ ShopSense")
    st.button(":material/add: New chat", type="primary", width="stretch", on_click=new_chat)
    if error := st.session_state.pop("sidebar_error", None):
        st.error(error)

    render_chat_list()

    st.divider()
    # Cart display
    try:
        cart = get_cart(session_id)
        with st.expander(f"🛒 Cart ({len(cart['items'])} items)", expanded=bool(cart["items"])):
            if cart["items"]:
                for item in cart["items"]:
                    price = f"₹{item['price']:,.0f}" if item.get("price") is not None else "price unknown"
                    st.write(f"• [{item['product_name']}]({item['url']}) — {price}")
                st.write(f"**Total: ₹{cart['total']:,.0f}**")
            else:
                st.write("Empty")
            if cart.get("budget"):
                st.caption(f"Budget for this chat: ₹{cart['budget']:,.0f}")
    except ApiError:
        st.write("Cart unavailable")

    if st.button("⚙️ Settings", width="stretch"):
        st.session_state.show_settings = True
    if backend_info:
        st.caption(f"Powered by {backend_info['llm']} · [Langfuse]({backend_info['langfuse_url']})")
    else:
        st.caption("Backend info unavailable")


if st.session_state.get("show_settings"):
    settings_dialog()
elif st.session_state.get("delete_chat"):
    delete_chat_dialog()


# ---- Chat ----

st.title("ShopSense")
st.caption("Your personal shopping concierge. Tell me what you're looking for.")

# Read input first: a clicked suggestion arrives as pending_message on the rerun
# the click triggers, and either way stale suggestion buttons shouldn't render
# while a new message is being answered.
prompt = st.chat_input("What are you looking for?")
if "pending_message" in st.session_state:
    prompt = st.session_state.pop("pending_message")

for i, message in enumerate(st.session_state.messages):
    render_message(message, i)

# Suggestions are fetched after the answer is on screen, so the answer never
# waits for them (the rerun after an answer sets fetch_suggestions).
if st.session_state.pop("fetch_suggestions", False) and not prompt:
    with st.spinner("Thinking of follow-ups..."):
        st.session_state.suggestions = get_suggestions(session_id)

# Follow-up suggestion chips: clicking one sends it as the next message.
if st.session_state.suggestions and not prompt:
    columns = st.columns(len(st.session_state.suggestions))
    for i, suggestion in enumerate(st.session_state.suggestions):
        if columns[i].button(suggestion, key=f"suggestion_{i}", width="stretch"):
            st.session_state.pending_message = suggestion
            st.session_state.suggestions = []
            st.rerun()

if prompt:
    st.session_state.suggestions = []
    user_message = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_message)
    render_message(user_message, len(st.session_state.messages) - 1)

    with st.chat_message("assistant"), st.spinner("Searching and analyzing..."):
        result: JSONObject | None
        try:
            result = send_message(session_id, prompt)
        except ApiError as e:
            result = None
            st.session_state.messages.append(
                {"role": "assistant", "content": str(e), "error": True, "trace_url": e.trace_url}
            )

    if result is not None:
        products, cited = pick_product_cards(result)
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result["response"],
                "products": products,
                "products_cited": cited,
                "price_checks": result.get("price_checks", []),
                "meta": {
                    k: result.get(k)
                    for k in ("trace_url", "step_count", "total_tokens", "estimated_cost_usd", "tool_calls_made")
                },
            }
        )
        st.session_state.fetch_suggestions = True

    # Re-run so the sidebar cart (which the agent may have just changed) and the
    # new messages render from fresh state; suggestions are fetched on that run.
    st.rerun()
