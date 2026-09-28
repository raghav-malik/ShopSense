"""ShopSense chat UI. Talks only to the FastAPI backend; all tracing is server-side.

Run:  streamlit run frontend/app.py   (backend: uvicorn app.main:app --port 8000)
"""

import os

import httpx
import streamlit as st

API_BASE = os.getenv("SHOPSENSE_API_URL", "http://localhost:8000")
# An agent turn is several LLM calls plus web searches (~20-45s measured, up to
# 10 steps worst case), so the spec's 60s timeout cut off slow-but-healthy turns.
CHAT_TIMEOUT = 180.0
MAX_PRODUCT_CARDS = 3

st.set_page_config(page_title="ShopSense", page_icon="🛍️", layout="wide")


# ---- Backend calls ----


class ApiError(Exception):
    def __init__(self, message: str, code: str | None = None, trace_url: str | None = None):
        super().__init__(message)
        self.code = code
        self.trace_url = trace_url


def _call(method: str, path: str, *, timeout: float = 15.0, **kwargs) -> dict:
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
    return response.json()


def create_session() -> str:
    """Create a new session via the API."""
    return _call("POST", "/sessions")["session_id"]


def send_message(session_id: str, message: str) -> dict:
    """Send a message to the agent."""
    return _call("POST", f"/sessions/{session_id}/chat", json={"message": message}, timeout=CHAT_TIMEOUT)


def get_cart(session_id: str) -> dict:
    """Fetch current cart."""
    return _call("GET", f"/sessions/{session_id}/cart")


def get_history(session_id: str) -> dict:
    return _call("GET", f"/sessions/{session_id}/history")


@st.cache_data(ttl=300, show_spinner=False)
def get_backend_info() -> dict | None:
    """Model name and Langfuse dashboard link, as configured on the backend."""
    try:
        return _call("GET", "/health", timeout=5.0)
    except ApiError:
        return None


# ---- Session state ----


def start_session(session_id: str, messages: list[dict] | None = None) -> None:
    st.session_state.session_id = session_id
    st.session_state.messages = messages or []
    st.session_state.suggestions = []
    # In the URL, so a page refresh reopens this conversation instead of
    # silently starting a new one.
    st.query_params["session"] = session_id


def restore_or_create_session() -> None:
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


def pick_product_cards(result: dict) -> tuple[list[dict], bool]:
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


def render_assistant_extras(message: dict, index: int) -> None:
    for n, product in enumerate(message.get("products", [])):
        icon = "🛒" if message.get("products_cited") else "🔎"
        with st.expander(f"{icon} {_short(product.get('title') or 'Product')}"):
            if product.get("snippet"):
                st.write(product["snippet"])
            if product.get("source"):
                st.caption(product["source"])
            if product.get("url"):
                label = "Buy Now →" if message.get("products_cited") else "View result →"
                # Keyed: the same product can appear in more than one answer.
                st.link_button(label, product["url"], key=f"buy_{index}_{n}")

    meta = message.get("meta")
    if meta and meta.get("trace_url"):
        with st.expander("🔍 Debug: Langfuse Trace"):
            st.write(f"[View trace]({meta['trace_url']})")
            st.write(f"Steps: {meta['step_count']}, Tokens: {meta['total_tokens']}")
            st.write(f"Tools: {', '.join(meta['tool_calls_made']) or 'none'}")


def render_message(message: dict, index: int) -> None:
    with st.chat_message(message["role"]):
        if message.get("error"):
            st.error(message["content"])
            if message.get("trace_url"):
                st.caption(f"[Debug this in Langfuse]({message['trace_url']})")
        else:
            st.markdown(message["content"])
        if message["role"] == "assistant":
            render_assistant_extras(message, index)


# ---- Sidebar ----

backend_info = get_backend_info()

with st.sidebar:
    st.title("🛍️ ShopSense")
    st.caption(f"Session: `{session_id[:8]}...`")

    if st.button("New Session"):
        try:
            start_session(create_session())
            st.rerun()
        except ApiError as e:
            st.error(str(e))

    st.divider()

    # Cart display
    try:
        cart = get_cart(session_id)
        st.subheader(f"🛒 Cart ({len(cart['items'])} items)")
        if cart["items"]:
            for item in cart["items"]:
                price = f"₹{item['price']:,.0f}" if item.get("price") is not None else "price unknown"
                st.write(f"• [{item['product_name']}]({item['url']}) — {price}")
            st.write(f"**Total: ₹{cart['total']:,.0f}**")
        else:
            st.write("Empty")
    except ApiError:
        st.write("Cart unavailable")

    st.divider()
    if backend_info:
        st.caption(f"Powered by {backend_info['llm']}")
        st.caption(f"[Langfuse Dashboard]({backend_info['langfuse_url']})")
    else:
        st.caption("Backend info unavailable")


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

# Follow-on suggestion chips (Airtap pattern): clicking one sends it as the next message.
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
                "meta": {k: result.get(k) for k in ("trace_url", "step_count", "total_tokens", "tool_calls_made")},
            }
        )
        st.session_state.suggestions = result.get("suggestions", [])

    # Re-run so the sidebar cart (which the agent may have just changed), the
    # new messages, and the suggestion chips all render from fresh state.
    st.rerun()
