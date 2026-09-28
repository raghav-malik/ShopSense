import json
import re
from collections.abc import Iterator
from typing import TypedDict
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup, Tag
from pydantic import BaseModel, Field, field_validator

from app.llm.types import JSONObject
from app.tools.base import pydantic_to_tool_schema

CURRENCY_SYMBOLS = {"INR": "₹", "USD": "$", "EUR": "€", "GBP": "£"}


class ExtractProductInput(BaseModel):
    """Input schema for the extract_product_info tool."""

    reasoning: str = Field(
        ...,
        description="Explain WHY you are extracting info from this URL. What product detail are you looking for?",
    )
    url: str = Field(
        ...,
        description="The product page URL to extract info from. Must be a valid HTTP/HTTPS URL.",
    )

    @field_validator("url")
    @classmethod
    def check_http_url(cls, v: str) -> str:
        parsed = urlparse(v)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError("url must be an absolute http(s) URL, e.g. 'https://www.amazon.in/dp/B0XXXX'")
        return v


EXTRACT_SCHEMA = pydantic_to_tool_schema(
    name="extract_product_info",
    description="Fetch a product URL and extract detailed information: name, price, rating, features, and buy link. Use this after search_products to get details on specific products.",
    input_model=ExtractProductInput,
)


FETCH_TIMEOUT = httpx.Timeout(10.0, connect=5.0)
MAX_REDIRECTS = 5
MAX_PAGE_BYTES = 3_000_000  # product pages are well under this; stops a huge download
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# Hints are for the LLM: what to do next instead of retrying blindly (in traces,
# the agent kept retrying blocked retailers until it hit the rate limit).
_USE_SNIPPET = "Don't retry this site; use the price and link from the search results instead."


class FetchError(Exception):
    def __init__(self, kind: str, message: str, hint: str):
        super().__init__(message)
        self.kind, self.message, self.hint = kind, message, hint


async def _fetch_html(url: str) -> str:
    try:
        async with (
            httpx.AsyncClient(follow_redirects=True, timeout=FETCH_TIMEOUT, max_redirects=MAX_REDIRECTS) as client,
            client.stream("GET", url, headers=HEADERS) as response,
        ):
            status = response.status_code
            if status in (401, 403, 429, 503):
                raise FetchError("blocked", f"HTTP {status}: the site refused automated access", _USE_SNIPPET)
            if status in (404, 410):
                raise FetchError(
                    "not_found",
                    f"HTTP {status}: the page doesn't exist",
                    "The URL may be wrong or outdated; don't guess variations of it.",
                )
            if status >= 400:
                raise FetchError("http_error", f"HTTP {status} fetching the page", _USE_SNIPPET)
            content_type = response.headers.get("content-type", "")
            if content_type and "html" not in content_type:
                raise FetchError(
                    "not_html",
                    f"Not a web page (content-type {content_type.split(';')[0]})",
                    "Use a product page URL, not an image, PDF or file.",
                )

            body = bytearray()
            async for chunk in response.aiter_bytes():
                body.extend(chunk)
                if len(body) >= MAX_PAGE_BYTES:
                    break  # product metadata sits in <head>; the first 3MB is plenty
            return body.decode(response.encoding or "utf-8", errors="replace")
    except httpx.TimeoutException as e:
        # Connect (5s) and read (10s) timeouts differ; say which one hit.
        phase = "connect" if isinstance(e, httpx.ConnectTimeout) else "respond"
        raise FetchError("timeout", f"The site didn't {phase} in time", _USE_SNIPPET) from e
    except httpx.ConnectError as e:
        raise FetchError(
            "connection_failed",
            "Couldn't connect (unknown host, refused, or TLS error)",
            "Check the URL came from the search results; don't invent URLs.",
        ) from e
    except httpx.TooManyRedirects as e:
        raise FetchError("too_many_redirects", f"More than {MAX_REDIRECTS} redirects", _USE_SNIPPET) from e
    except (httpx.InvalidURL, httpx.UnsupportedProtocol) as e:
        raise FetchError(
            "invalid_url",
            "The URL isn't a valid http(s) address",
            "Use a URL exactly as it appeared in the search results.",
        ) from e
    except httpx.RequestError as e:
        raise FetchError("network_error", f"Network error: {type(e).__name__}", _USE_SNIPPET) from e


async def extract_product_info(url: str) -> JSONObject:
    """
    Fetch a product URL and extract structured information.
    Uses Open Graph tags, JSON-LD, and meta tag fallbacks.
    Does NOT execute JavaScript — this is a best-effort extraction.

    Never raises: failures come back as {"error", "error_type", "hint"} so the
    agent can decide what to do next.
    """
    try:
        html = await _fetch_html(url)
    except FetchError as e:
        return {"error": e.message, "error_type": e.kind, "hint": e.hint, "buy_link": url, "available": False}

    try:
        soup = BeautifulSoup(html, "html.parser")

        # 1. Try JSON-LD structured data
        product_data = _extract_json_ld(soup)

        # 2. Try Open Graph tags
        og_data = _extract_og_tags(soup)

        # 3. Fallback to meta tags and title
        meta_data = _extract_meta(soup)

        # Merge: JSON-LD > OG > Meta
        name = product_data.get("name") or og_data.get("name") or meta_data.get("name") or "Unknown Product"
        price = product_data.get("price") or og_data.get("price") or meta_data.get("price")
        rating = product_data.get("rating")
        features = product_data.get("features", [])
        image = og_data.get("image") or product_data.get("image")

        return {
            "name": name,
            "price": price,
            "rating": rating,
            "features": features[:5],  # cap at 5
            "buy_link": url,
            "image": image,
            # True/False only when the page states stock; None means unknown.
            "available": product_data.get("available"),
            "source": _get_domain(url),
        }

    except Exception as e:  # noqa: BLE001 - malformed HTML/JSON-LD we didn't anticipate
        return {
            "error": f"Couldn't read product details: {type(e).__name__}",
            "error_type": "parse_error",
            "hint": _USE_SNIPPET,
            "buy_link": url,
            "available": False,
        }


class ProductFields(TypedDict, total=False):
    """What one extraction source (JSON-LD, Open Graph, meta tags) found."""

    name: str
    price: str | None
    rating: str
    features: list[str]
    image: str | None
    available: bool


def _extract_json_ld(soup: BeautifulSoup) -> ProductFields:
    """Extract product data from the first JSON-LD Product node on the page."""
    for script in soup.find_all("script", type="application/ld+json"):
        if not script.string:  # empty tag, or content split across child nodes
            continue
        try:
            data = json.loads(script.string)
        except json.JSONDecodeError:
            continue
        for node in _iter_json_ld_nodes(data):
            if _is_product(node):
                return _parse_product_node(node)
    return {}


def _iter_json_ld_nodes(data: object) -> Iterator[JSONObject]:
    """Yield every object in a JSON-LD blob: top-level lists and @graph wrappers included."""
    if isinstance(data, list):
        for item in data:
            yield from _iter_json_ld_nodes(item)
    elif isinstance(data, dict):
        yield data
        if "@graph" in data:
            yield from _iter_json_ld_nodes(data["@graph"])


def _is_product(node: JSONObject) -> bool:
    types = node.get("@type")
    types = types if isinstance(types, list) else [types]
    return any(t in ("Product", "IndividualProduct") for t in types)


def _parse_product_node(data: JSONObject) -> ProductFields:
    name = data.get("name")
    result: ProductFields = {"name": name if isinstance(name, str) else ""}
    offers = data.get("offers")
    if isinstance(offers, list):
        offers = offers[0] if offers else None
    if isinstance(offers, dict):
        # AggregateOffer (multiple sellers) carries lowPrice instead of price.
        currency = offers.get("priceCurrency")
        result["price"] = _format_price(
            offers.get("price") or offers.get("lowPrice"), currency if isinstance(currency, str) else None
        )
        availability = str(offers.get("availability", ""))
        if availability:
            result["available"] = availability.rstrip("/").endswith(("InStock", "LimitedAvailability", "OnlineOnly"))
    rating = data.get("aggregateRating")
    if isinstance(rating, dict):
        result["rating"] = f"{rating.get('ratingValue', '?')} / 5 ({rating.get('reviewCount', '?')} reviews)"
    description = data.get("description")
    if isinstance(description, str):
        # Extract feature-like sentences
        result["features"] = [s.strip() for s in description.split(",")[:5] if s.strip()]
    if "image" in data:
        img = data["image"]
        if isinstance(img, list):
            img = img[0] if img else None
        if isinstance(img, dict):  # ImageObject
            img = img.get("url")
        result["image"] = img if isinstance(img, str) else None
    return result


def _format_price(amount: object, currency: str | None) -> str | None:
    """'1299', 'INR' -> '₹1299'. Unknown currencies keep their code; none given means INR."""
    if amount in (None, ""):
        return None
    code = (currency or "INR").upper()
    symbol = CURRENCY_SYMBOLS.get(code, f"{code} ")
    return f"{symbol}{amount}"


def _meta_content(soup: BeautifulSoup, **attrs: str) -> str | None:
    """The `content` of the first matching <meta>, if it's a plain string.
    BeautifulSoup returns lists for multi-valued attributes, so check the type."""
    tag = soup.find("meta", attrs=dict(attrs))
    content = tag.get("content") if isinstance(tag, Tag) else None
    return content if isinstance(content, str) else None


def _extract_og_tags(soup: BeautifulSoup) -> ProductFields:
    """Extract Open Graph meta tags."""
    result: ProductFields = {}
    if (title := _meta_content(soup, property="og:title")) is not None:
        result["name"] = title
    amount = _meta_content(soup, property="product:price:amount") or _meta_content(soup, property="og:price:amount")
    if amount:
        currency = _meta_content(soup, property="product:price:currency") or _meta_content(
            soup, property="og:price:currency"
        )
        result["price"] = _format_price(amount, currency)
    if (image := _meta_content(soup, property="og:image")) is not None:
        result["image"] = image
    return result


def _extract_meta(soup: BeautifulSoup) -> ProductFields:
    """Fallback: extract from title and meta description."""
    result: ProductFields = {}
    title = soup.find("title")
    if isinstance(title, Tag):
        result["name"] = title.get_text(strip=True)
    description = _meta_content(soup, name="description")
    if description:
        # Try to find a price pattern
        price_match = re.search(r"₹[\d,]+(?:\.\d{2})?|Rs\.?\s*[\d,]+", description)
        if price_match:
            result["price"] = price_match.group()
    return result


def _get_domain(url: str) -> str:
    return urlparse(url).netloc.removeprefix("www.")
