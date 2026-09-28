"""extract_product_info offline: pages are served by an in-process httpx
transport, so parsing and the fetch-error mapping run without the internet.
(test_tools.py has the `network` tests against real pages.)"""

import json
from collections.abc import Callable
from typing import Any

import httpx
import pytest

from app.llm.types import JSONObject
from app.tools.extract import extract_product_info

URL = "https://www.shop.test/products/airdopes-141"

type Handler = Callable[[httpx.Request], httpx.Response]


@pytest.fixture
def serve(monkeypatch: pytest.MonkeyPatch) -> Callable[[Handler], None]:
    """Route the tool's HTTP client to `handler` instead of the network."""
    real_client = httpx.AsyncClient

    def install(handler: Handler) -> None:
        def client(**kwargs: Any) -> httpx.AsyncClient:
            return real_client(transport=httpx.MockTransport(handler), **kwargs)

        # The tool looks up httpx.AsyncClient on each call.
        monkeypatch.setattr(httpx, "AsyncClient", client)

    return install


def page(head: str) -> Handler:
    html = f"<html><head>{head}</head><body><h1>Shop</h1></body></html>"
    return lambda request: httpx.Response(200, html=html)


def json_ld(data: JSONObject | list[JSONObject]) -> str:
    return f'<script type="application/ld+json">{json.dumps(data)}</script>'


OG_TAGS = """
<meta property="og:title" content="OG title">
<meta property="product:price:amount" content="1299">
<meta property="product:price:currency" content="INR">
<meta property="og:image" content="https://cdn.shop.test/og.jpg">
"""


# ---- parsing ----


async def test_json_ld_product_in_a_graph(serve: Callable[[Handler], None]) -> None:
    product = {
        "@type": "Product",
        "name": "boAt Airdopes 141",
        "description": "42H playback, ENx mics, IPX4",
        "image": [{"@type": "ImageObject", "url": "https://cdn.shop.test/a141.jpg"}],
        "offers": {"@type": "AggregateOffer", "lowPrice": "19.99", "priceCurrency": "usd",
                   "availability": "https://schema.org/InStock"},
        "aggregateRating": {"ratingValue": "4.1", "reviewCount": "2,310"},
    }  # fmt: skip
    serve(page(json_ld({"@context": "https://schema.org", "@graph": [{"@type": "WebPage"}, product]}) + OG_TAGS))

    result = await extract_product_info(URL)

    assert result == {
        "name": "boAt Airdopes 141",  # JSON-LD wins over Open Graph
        "price": "$19.99",  # AggregateOffer's lowPrice, currency symbol from the code
        "rating": "4.1 / 5 (2,310 reviews)",
        "features": ["42H playback", "ENx mics", "IPX4"],
        "buy_link": URL,
        "image": "https://cdn.shop.test/og.jpg",  # Open Graph's image wins over JSON-LD's
        "available": True,
        "source": "shop.test",
    }


async def test_open_graph_when_json_ld_is_broken(serve: Callable[[Handler], None]) -> None:
    broken = '<script type="application/ld+json">{"@type": "Product", "name": </script>'
    serve(page(broken + OG_TAGS))
    result = await extract_product_info(URL)
    assert result["name"] == "OG title" and result["price"] == "₹1299"
    assert result["available"] is None  # the page doesn't say, so it's unknown


async def test_meta_description_fallback(serve: Callable[[Handler], None]) -> None:
    serve(page('<title> Noise Buds VS104 </title><meta name="description" content="Buy now at Rs. 999 only">'))
    result = await extract_product_info(URL)
    assert result["name"] == "Noise Buds VS104" and result["price"] == "Rs. 999"


async def test_out_of_stock_and_unknown_currency(serve: Callable[[Handler], None]) -> None:
    offer = {"price": 45, "priceCurrency": "AED", "availability": "https://schema.org/OutOfStock"}
    serve(page(json_ld([{"@type": ["Product", "Thing"], "name": "Buds", "offers": [offer]}])))
    result = await extract_product_info(URL)
    assert result["price"] == "AED 45" and result["available"] is False


async def test_page_without_product_data(serve: Callable[[Handler], None]) -> None:
    serve(page(""))
    result = await extract_product_info(URL)
    assert result["name"] == "Unknown Product" and result["price"] is None and result["features"] == []


# ---- fetch errors: a type and a hint for the agent, never an exception ----


def status(code: int, content_type: str = "text/html") -> Handler:
    return lambda request: httpx.Response(code, headers={"content-type": content_type}, text="")


def raises(error: type[httpx.RequestError]) -> Handler:
    def handler(request: httpx.Request) -> httpx.Response:
        raise error("simulated", request=request)

    return handler


@pytest.mark.parametrize(
    ("handler", "error_type", "message_part"),
    [
        (status(403), "blocked", "HTTP 403"),
        (status(429), "blocked", "HTTP 429"),
        (status(404), "not_found", "doesn't exist"),
        (status(500), "http_error", "HTTP 500"),
        (status(200, "application/pdf"), "not_html", "application/pdf"),
        (raises(httpx.ConnectTimeout), "timeout", "didn't connect"),
        (raises(httpx.ReadTimeout), "timeout", "didn't respond"),
        (raises(httpx.ConnectError), "connection_failed", "Couldn't connect"),
        (raises(httpx.RemoteProtocolError), "network_error", "RemoteProtocolError"),
    ],
    ids=["403", "429", "404", "500", "pdf", "connect-timeout", "read-timeout", "connect-error", "protocol"],
)
async def test_fetch_errors_come_back_typed(
    serve: Callable[[Handler], None], handler: Handler, error_type: str, message_part: str
) -> None:
    serve(handler)
    result = await extract_product_info(URL)
    assert result["error_type"] == error_type
    assert message_part in result["error"]
    assert result["hint"] and result["buy_link"] == URL and result["available"] is False


async def test_too_many_redirects(serve: Callable[[Handler], None]) -> None:
    serve(lambda request: httpx.Response(302, headers={"location": str(request.url)}))
    result = await extract_product_info(URL)
    assert result["error_type"] == "too_many_redirects"
