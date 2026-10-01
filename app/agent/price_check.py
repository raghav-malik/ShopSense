"""Check the prices in an answer against the live store pages before it's shown.

The agent's prices come from search snippets, and an audit of 125 turns found
96% of them copied faithfully, so it rarely invents prices. But snippets are
cached and often second-hand (price-tracking and review sites), so the price
behind the buy link can differ. This step fetches each linked product page
(through the SSRF-safe fetcher in app.tools.extract) and:

- verified:    the stated price matches the store (to the rupee)
- corrected:   it doesn't; the answer now shows the store's price, with a note
- unavailable: the store says it's out of stock; the answer says so
- unchecked:   the page couldn't be read in time, or has no readable price
- search page: the link is a search or category page, not one product

Store coverage, measured on real links in September 2026: Flipkart and
Amazon.in product pages (Amazon via its buy box), and brand stores that
publish standard product data.
"""

import asyncio
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from urllib.parse import urlparse

from app.agent.schemas import PriceCheck
from app.config import settings
from app.tools.extract import extract_product_info

_LINK = re.compile(r"\[([^\]]*)\]\((https?://[^\s)]+)\)|(?<![(\[])(https?://[^\s)\]>\"'*]+)")
_PRICE = re.compile(r"(?:₹|Rs\.?\s?|INR\s?)\s?(\d[\d,]*(?:\.\d+)?)", re.IGNORECASE)
# Search results, categories and listings: many products, so there's no single price to check.
_LISTING = re.compile(r"/s\?k=|/search|[?&](?:q|k|query)=|/b\?|/b/|/collections/|/category|/l/|/browse|/sr\?", re.I)
# Words in link text that name the store, not the product ("Buy on Amazon.in").
_GENERIC_LINK_TEXT = re.compile(r"^(buy|view|check|see|shop|open|link|here|listing|price)\b", re.I)
MAX_CONCURRENT_FETCHES = 4


@dataclass
class _Link:
    url: str
    text: str
    line: int


def _store(url: str) -> str:
    return urlparse(url).netloc.removeprefix("www.")


def _amount(text: str) -> float:
    return float(text.replace(",", ""))


def _links(answer: str) -> list[_Link]:
    """Each distinct link in the answer, with its link text and line number."""
    found: dict[str, _Link] = {}
    for number, line in enumerate(answer.split("\n")):
        for match in _LINK.finditer(line):
            url = (match.group(2) or match.group(3)).rstrip(".,;:")
            found.setdefault(url, _Link(url, match.group(1) or "", number))
    return list(found.values())


def _stated_price(lines: list[str], link: _Link) -> tuple[int, str, float] | None:
    """The price the answer gives for this link: on the link's line, else on
    the closest of the two lines above it, within the same block."""
    for number in range(link.line, max(link.line - 3, -1), -1):
        if number < link.line and not lines[number].strip():
            break  # a blank line ends the item
        line = lines[number]
        if number == link.line:
            # On the link's own line, the price closest before the link belongs to it.
            prices = list(_PRICE.finditer(line[: line.find(link.url)]))
            match = prices[-1] if prices else _PRICE.search(line)
        else:
            match = _PRICE.search(line)
        if match:
            return number, match.group(0), _amount(match.group(1))
    return None


# Prices the check trusts: the store's own product data. A price read from a
# page's description text isn't one product's price (a category page described
# as "shoes under ₹3,000" once "matched" ₹3,000).
_TRUSTED_PRICE_SOURCES = {"structured_data", "amazon_buy_box", "open_graph"}


async def _live_details(url: str) -> tuple[float | None, bool | None, str | None]:
    """The store page's price, stock, and product name. Tests replace this."""
    result = await extract_product_info(url)
    if "error_type" in result:
        return None, None, None
    trusted = result.get("price_source") in _TRUSTED_PRICE_SOURCES
    price = _PRICE.search(str(result.get("price") or "")) if trusted else None
    return (_amount(price.group(1)) if price else None), result.get("available"), result.get("name")


async def _fetch_all(urls: list[str]) -> dict[str, tuple[float | None, bool | None, str | None]]:
    """Live details for every URL that answers within the time limit."""
    limit = asyncio.Semaphore(MAX_CONCURRENT_FETCHES)

    async def one(url: str) -> tuple[str, tuple[float | None, bool | None, str | None]]:
        async with limit:
            return url, await _live_details(url)

    tasks = [asyncio.create_task(one(url)) for url in urls]
    done, pending = await asyncio.wait(tasks, timeout=settings.price_check_timeout)
    for task in pending:
        task.cancel()
    return dict(task.result() for task in done if not task.exception())


def _product_name(link: _Link, page_name: str | None) -> str:
    if link.text and not _GENERIC_LINK_TEXT.match(link.text):
        return link.text
    name = page_name or _store(link.url)
    return name if len(name) <= 60 else name[:60].rsplit(" ", 1)[0] + " …"  # cut at a word boundary


async def check_prices(answer: str) -> tuple[str, list[PriceCheck]]:
    """The answer with its prices checked against the stores, and one check per link."""
    links = _links(answer)
    if not settings.price_check_enabled or not links:
        return answer, []
    product_links = [link for link in links if not _LISTING.search(link.url)]
    live = await _fetch_all([link.url for link in product_links]) if product_links else {}
    lines = answer.split("\n")
    checked_at = datetime.now(UTC).isoformat(timespec="seconds")
    checks: list[PriceCheck] = []
    notes: list[str] = []

    for link in links:
        stated = _stated_price(lines, link)
        stated_value = stated[2] if stated else None
        check = PriceCheck(
            url=link.url,
            store=_store(link.url),
            product=_product_name(link, None),
            stated_price=stated_value,
            checked_at=checked_at,
        )
        if _LISTING.search(link.url):
            check.status = "search_page"
        elif link.url in live:
            live_price, available, page_name = live[link.url]
            check.live_price = live_price
            name = check.product = _product_name(link, page_name)
            if available is False:
                check.status = "unavailable"
                notes.append(f"**{name}** is currently unavailable on {check.store}.")
            elif live_price is None:
                check.status = "unchecked"
            # To the rupee: a shopper who clicks through should see the same number.
            elif stated_value is None or abs(stated_value - live_price) < 1.0:
                check.status = "verified"
            elif stated:
                check.status = "corrected"
                number, text, _ = stated
                lines[number] = lines[number].replace(text, f"₹{live_price:,.0f}", 1)
                notes.append(
                    f"**{name}** is ₹{live_price:,.0f} on {check.store} (search results said ₹{stated_value:,.0f})."
                )
        checks.append(check)

    corrected = "\n".join(lines)
    if notes:
        corrected += "\n\n_Prices checked on the store pages just now: " + " ".join(notes) + "_"
    return corrected, checks
