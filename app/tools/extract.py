import json
import re
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from pydantic import BaseModel, Field, field_validator

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


async def extract_product_info(url: str) -> dict:
    """
    Fetch a product URL and extract structured information.
    Uses Open Graph tags, JSON-LD, and meta tag fallbacks.
    Does NOT execute JavaScript — this is a best-effort extraction.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    try:
        async with httpx.AsyncClient(follow_redirects=True, timeout=10.0) as client:
            response = await client.get(url, headers=headers)
            response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

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

    except httpx.HTTPStatusError as e:
        return {"error": f"HTTP {e.response.status_code} fetching {url}", "buy_link": url, "available": False}
    except httpx.TimeoutException:
        return {"error": f"Timeout fetching {url}", "buy_link": url, "available": False}
    except Exception as e:
        return {"error": f"Extraction failed: {str(e)}", "buy_link": url, "available": False}


def _extract_json_ld(soup: BeautifulSoup) -> dict:
    """Extract product data from the first JSON-LD Product node on the page."""
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string)
        except (json.JSONDecodeError, TypeError):
            continue
        for node in _iter_json_ld_nodes(data):
            if _is_product(node):
                return _parse_product_node(node)
    return {}


def _iter_json_ld_nodes(data):
    """Yield every object in a JSON-LD blob: top-level lists and @graph wrappers included."""
    if isinstance(data, list):
        for item in data:
            yield from _iter_json_ld_nodes(item)
    elif isinstance(data, dict):
        yield data
        if "@graph" in data:
            yield from _iter_json_ld_nodes(data["@graph"])


def _is_product(node: dict) -> bool:
    types = node.get("@type")
    types = types if isinstance(types, list) else [types]
    return any(t in ("Product", "IndividualProduct") for t in types)


def _parse_product_node(data: dict) -> dict:
    result = {"name": data.get("name", "")}
    offers = data.get("offers")
    if isinstance(offers, list):
        offers = offers[0] if offers else None
    if isinstance(offers, dict):
        # AggregateOffer (multiple sellers) carries lowPrice instead of price.
        result["price"] = _format_price(offers.get("price") or offers.get("lowPrice"), offers.get("priceCurrency"))
        availability = str(offers.get("availability", ""))
        if availability:
            result["available"] = availability.rstrip("/").endswith(("InStock", "LimitedAvailability", "OnlineOnly"))
    if isinstance(data.get("aggregateRating"), dict):
        ar = data["aggregateRating"]
        result["rating"] = f"{ar.get('ratingValue', '?')} / 5 ({ar.get('reviewCount', '?')} reviews)"
    if isinstance(data.get("description"), str):
        # Extract feature-like sentences
        result["features"] = [s.strip() for s in data["description"].split(",")[:5] if s.strip()]
    if "image" in data:
        img = data["image"]
        if isinstance(img, list):
            img = img[0] if img else None
        if isinstance(img, dict):  # ImageObject
            img = img.get("url")
        result["image"] = img if isinstance(img, str) else None
    return result


def _format_price(amount, currency: str | None) -> str | None:
    """'1299', 'INR' -> '₹1299'. Unknown currencies keep their code; none given means INR."""
    if amount in (None, ""):
        return None
    code = (currency or "INR").upper()
    symbol = CURRENCY_SYMBOLS.get(code, f"{code} ")
    return f"{symbol}{amount}"


def _extract_og_tags(soup: BeautifulSoup) -> dict:
    """Extract Open Graph meta tags."""
    result = {}
    og_title = soup.find("meta", property="og:title")
    if og_title:
        result["name"] = og_title.get("content", "")
    og_price = soup.find("meta", property="product:price:amount") or soup.find("meta", property="og:price:amount")
    if og_price:
        og_currency = soup.find("meta", property="product:price:currency") or soup.find("meta", property="og:price:currency")
        result["price"] = _format_price(og_price.get("content"), og_currency.get("content") if og_currency else None)
    og_image = soup.find("meta", property="og:image")
    if og_image:
        result["image"] = og_image.get("content", "")
    return result


def _extract_meta(soup: BeautifulSoup) -> dict:
    """Fallback: extract from title and meta description."""
    result = {}
    title = soup.find("title")
    if title:
        result["name"] = title.get_text(strip=True)
    desc = soup.find("meta", attrs={"name": "description"})
    if desc:
        content = desc.get("content", "")
        # Try to find a price pattern
        price_match = re.search(r'₹[\d,]+(?:\.\d{2})?|Rs\.?\s*[\d,]+', content)
        if price_match:
            result["price"] = price_match.group()
    return result


def _get_domain(url: str) -> str:
    return urlparse(url).netloc.removeprefix("www.")
