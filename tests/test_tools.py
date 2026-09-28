"""Each tool against its typed output contract, plus the registry that validates
LLM arguments before any tool runs."""

import json

import pytest
from ddgs.exceptions import DDGSException, RatelimitException

from app.tools import registry, search
from app.tools.cart import manage_cart
from app.tools.compare import compare_products
from app.tools.extract import extract_product_info
from app.tools.preferences import handle_preferences
from app.tools.registry import execute_tool, get_tool_schemas
from app.tools.search import MAX_RESULTS, SNIPPET_MAX_CHARS, search_products

KNOWN_PRODUCT_URL = "https://www.boat-lifestyle.com/products/airdopes-141"


async def run(name: str, args: dict, session_id: str = "s1") -> dict:
    """Call a tool the way the agent does: JSON args through the registry."""
    return json.loads(await execute_tool(name, json.dumps({"reasoning": "test", **args}), session_id))


# ---- search_products ----

@pytest.mark.network
async def test_search_products_returns_results():
    result = await search_products("wireless earbuds under 3000 INR", max_results=3)
    assert "error" not in result
    assert isinstance(result["results"], list)
    # DuckDuckGo should return something for a common query
    assert 0 < len(result["results"]) <= 3
    assert result["result_count"] == len(result["results"])
    assert result["query_used"] == "wireless earbuds under 3000 INR"


@pytest.mark.network
async def test_search_products_has_required_fields():
    result = await search_products("laptop bag", max_results=2)
    for r in result["results"]:
        assert set(r) == {"title", "url", "snippet", "source"}
        assert r["url"].startswith("http")
        assert r["source"] and not r["source"].startswith("www.")
        # Snippets are capped so tool results don't bloat every later LLM call.
        assert len(r["snippet"]) <= SNIPPET_MAX_CHARS + 2


async def test_search_products_empty_results(monkeypatch):
    monkeypatch.setattr(search, "_ddgs_text", lambda q, n: [])
    result = await search_products("earbuds", max_results=3)
    assert result["results"] == [] and result["result_count"] == 0
    assert "No products found" in result["message"]


async def test_search_products_no_results_exception_is_not_an_error(monkeypatch):
    def raise_no_results(q, n):
        raise DDGSException("No results found.")
    monkeypatch.setattr(search, "_ddgs_text", raise_no_results)
    result = await search_products("earbuds", max_results=3)
    assert "error" not in result and result["results"] == []


async def test_search_products_rate_limit_has_type_and_hint(monkeypatch):
    def raise_rate_limit(q, n):
        raise RatelimitException("202 Ratelimit")
    monkeypatch.setattr(search, "_ddgs_text", raise_rate_limit)
    result = await search_products("earbuds", max_results=3)
    assert result["error_type"] == "rate_limited" and result["hint"] and result["results"] == []


async def test_search_max_results_is_capped():
    result = await run("search_products", {"query": "earbuds", "max_results": MAX_RESULTS + 5})
    assert result["error"] == "validation_failed"


# ---- extract_product_info ----

@pytest.mark.network
async def test_extract_known_product_page():
    result = await extract_product_info(KNOWN_PRODUCT_URL)
    assert "error" not in result, result
    assert "Airdopes 141" in result["name"]
    assert result["price"].startswith("₹")
    assert result["available"] in (True, False)
    assert result["buy_link"] == KNOWN_PRODUCT_URL
    assert result["source"] == "boat-lifestyle.com"
    assert isinstance(result["features"], list) and len(result["features"]) <= 5


@pytest.mark.network
async def test_extract_missing_page_is_not_found():
    result = await extract_product_info("https://www.boat-lifestyle.com/products/this-product-does-not-exist-xyz")
    assert result["error_type"] == "not_found"
    assert result["available"] is False and result["hint"]


@pytest.mark.network
async def test_extract_unknown_host_is_connection_failed():
    result = await extract_product_info("https://no-such-shop.shopsense-test.invalid/p/1")
    assert result["error_type"] == "connection_failed"


@pytest.mark.parametrize("url", ["file:///etc/passwd", "earbuds under 3000", "https://"])
async def test_extract_rejects_non_http_urls_before_fetching(url):
    result = await run("extract_product_info", {"url": url})
    assert result["error"] == "validation_failed"
    assert "http(s)" in result["validation_errors"][0]["msg"]


# ---- compare_products ----

async def test_compare_products_needs_two():
    result = await compare_products([{"name": "A", "price": "100", "url": "http://a.com"}])
    assert "error" in result


async def test_compare_products_builds_table():
    products = [
        {"name": "Product A", "price": "₹1,299", "features": ["Bluetooth", "IPX4"], "url": "http://a.com"},
        {"name": "Product B", "price": "₹2,499", "features": ["ANC", "USB-C"], "url": "http://b.com"},
    ]
    result = await compare_products(products)
    assert result["product_count"] == 2
    table = result["comparison_table"].splitlines()
    assert table[0] == "| Product | Price | Key Features | Rating | Link |"
    assert "| Product A | ₹1,299 | Bluetooth, IPX4 | N/A | [Buy](http://a.com) |" in table


async def test_compare_escapes_pipes_and_shows_missing_rating_as_na():
    # Through the registry, validated products arrive with rating=None (SR-60).
    result = await run("compare_products", {"products": [
        {"name": "Buds | Pro", "price": "₹999", "url": "http://a.com"},
        {"name": "B", "price": "₹1,999", "url": "http://b.com", "rating": "4.2 / 5"},
    ]})
    assert "Buds \\| Pro" in result["comparison_table"]
    assert result["products"][0]["rating"] == "N/A"
    assert "None" not in result["comparison_table"]


async def test_compare_rejects_one_product_via_registry():
    result = await run("compare_products", {"products": [{"name": "A", "price": "1", "url": "http://a"}]})
    assert result["error"] == "validation_failed"


# ---- manage_cart (real SQLite, isolated per test) ----

async def test_cart_add_view_remove_clear(db):
    from app.db import queries
    session = await queries.create_session()
    sid = session.id

    added = await manage_cart("add", sid, product_name="boAt Airdopes 141", price=799, url="https://x/1")
    assert added == {"message": "Added boAt Airdopes 141 to cart.", "cart_size": 1, "total": 799.0, "currency": "INR"}
    await manage_cart("add", sid, product_name="Noise Buds", price=999, url="https://x/2")

    view = await manage_cart("view", sid)
    assert view["cart_size"] == 2 and view["total"] == 1798.0
    assert [i["name"] for i in view["items"]] == ["boAt Airdopes 141", "Noise Buds"]

    removed = await manage_cart("remove", sid, product_name="Noise Buds")
    assert removed["message"] == "Removed Noise Buds." and removed["total"] == 799.0
    missing = await manage_cart("remove", sid, product_name="Sony")
    assert missing["message"] == "Could not find Sony."

    cleared = await manage_cart("clear", sid)
    assert cleared == {"message": "Cart cleared.", "cart_size": 0, "total": 0}


async def test_cart_is_per_session(db):
    from app.db import queries
    a, b = await queries.create_session(), await queries.create_session()
    await manage_cart("add", a.id, product_name="X", price=1, url="https://x")
    assert (await manage_cart("view", b.id))["items"] == []


async def test_cart_add_without_url_fails_validation_with_json_error(db):
    # A model_validator error: its ctx holds a raw ValueError, which used to
    # crash json.dumps inside the registry's own error handler (SR-59).
    result = await run("manage_cart", {"action": "add", "product_name": "X"})
    assert result["error"] == "validation_failed"
    assert "expected_schema" in result


# ---- get_preferences (real SQLite) ----

async def test_preferences_set_and_get(db):
    set_list = await handle_preferences("set", key="preferred_brands", value='["Samsung", "Sony"]')
    assert set_list["preferences"] == {"preferred_brands": ["Samsung", "Sony"]}
    # A bare string (not JSON) is stored as-is; a second set upserts.
    await handle_preferences("set", key="preferred_brands", value="Samsung")
    assert await handle_preferences("get") == {"preferences": {"preferred_brands": "Samsung"}}


# ---- registry ----

async def test_registry_unknown_tool():
    result = await run("buy_now", {})
    assert result["error"] == "unknown_tool"
    assert set(result["available_tools"]) == {s["function"]["name"] for s in get_tool_schemas()}


@pytest.mark.parametrize("raw", ['{"query": ', '["earbuds"]'])
async def test_registry_rejects_bad_json(raw):
    result = json.loads(await execute_tool("search_products", raw, "s1"))
    assert result["error"] == "invalid_json"


async def test_registry_missing_reasoning_fails_validation():
    result = json.loads(await execute_tool("search_products", json.dumps({"query": "x"}), "s1"))
    assert result["error"] == "validation_failed"
    assert result["validation_errors"][0]["loc"] == ["reasoning"]


async def test_registry_strips_reasoning_and_injects_session(monkeypatch):
    # The LLM-only `reasoning` field must never reach an executor (SR-25), and
    # the cart's session_id comes from the server, not the model.
    received = {}

    async def fake_cart(**kwargs):
        received.update(kwargs)
        return {"ok": True}

    executor, schema, model, needs_session = registry.TOOL_MAP["manage_cart"]
    monkeypatch.setitem(registry.TOOL_MAP, "manage_cart", (fake_cart, schema, model, needs_session))
    await execute_tool("manage_cart", json.dumps({"reasoning": "user asked", "action": "view", "session_id": "hacked"}), "real-session")
    assert "reasoning" not in received
    assert received["session_id"] == "real-session"


def test_tool_schemas_are_openai_function_format():
    schemas = get_tool_schemas()
    assert [s["function"]["name"] for s in schemas] == [
        "search_products", "extract_product_info", "compare_products", "manage_cart", "get_preferences",
    ]
    for s in schemas:
        params = s["function"]["parameters"]
        assert s["type"] == "function" and params["type"] == "object"
        assert "reasoning" in params["required"]
        assert "title" not in params
    cart_props = schemas[3]["function"]["parameters"]["properties"]
    assert "session_id" not in cart_props  # injected server-side, hidden from the LLM
