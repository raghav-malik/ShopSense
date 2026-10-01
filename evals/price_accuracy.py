"""Eval: do the prices shown next to product links match the store pages?

Runs real shopping requests through the agent, then fetches every product link
in each answer and compares the price shown beside it with the store's live
price. Run it with the live price check off and on to see what it changes:

    uv run python -m evals.price_accuracy            # check on (default)
    PRICE_CHECK_ENABLED=false uv run python -m evals.price_accuracy

Costs about a cent on gpt-6-luna, plus page fetches. Runs locally with your .env.
"""

import asyncio
import io
import os
import sys
import tempfile
import time
from collections import Counter

os.environ.setdefault("LANGFUSE_TRACING_ENVIRONMENT", "price-eval")

from app.agent.core import run_agent
from app.agent.price_check import _LISTING, _links, _live_details, _stated_price
from app.config import settings
from app.db import database, queries
from app.llm.errors import LLMError
from app.tracing.langfuse_setup import flush_langfuse

REQUESTS = [
    "find me wireless earbuds under 3000",
    "best budget smartphone under 15k",
    "a laptop for students under 50000",
    "smartwatch with calling under 4000",
    "mixer grinder under 3500",
    "20000mAh power bank with fast charging",
    "running shoes for men under 3000",
    "mechanical keyboard under 5000",
    "noise cancelling headphones under 10000",
    "laptop backpack under 1500",
]


DETAILS: list[str] = []


async def one(request: str) -> tuple[list[str], float]:
    """Run one request; classify each link's shown price against the live store page."""
    session = await queries.create_session()
    started = time.perf_counter()
    try:
        result = await run_agent(session.id, request)
    except LLMError as e:  # a provider failure isn't a price result either way
        print(f"  {request!r}: agent error ({e.code})")
        return [], 0.0
    seconds = time.perf_counter() - started
    lines = result.response.split("\n")
    outcomes = []
    for link in _links(result.response):
        if _LISTING.search(link.url):
            outcomes.append("search page")
            continue
        stated = _stated_price(lines, link)
        live_price, available, _ = await _live_details(link.url)
        if stated is None:
            outcomes.append("no price shown")
        elif available is False:
            outcomes.append("unavailable at store")
        elif live_price is None:
            outcomes.append("store price unreadable")
        elif abs(stated[2] - live_price) < 1.0:
            outcomes.append("matches store")
        else:
            outcomes.append("DIFFERS from store")
        if stated is not None and live_price is not None:
            DETAILS.append(f"  shown ₹{stated[2]:,.0f} vs store ₹{live_price:,.0f}  {stated[1]!r} near {link.url[:70]}")
    return outcomes, seconds


async def run_all() -> list[tuple[list[str], float]]:
    """Every request, three at a time, in a throwaway database."""
    with tempfile.TemporaryDirectory() as tmp:
        settings.db_path = f"{tmp}/price.db"
        await database.init_db()
        try:
            results = []
            for i in range(0, len(REQUESTS), 3):
                results += await asyncio.gather(*(one(r) for r in REQUESTS[i : i + 3]))
            return results
        finally:
            await database.close_db()
            flush_langfuse()


def main() -> None:
    """Run the eval and print how the shown prices compare with the stores."""
    results = asyncio.run(run_all())
    counts = Counter(outcome for outcomes, _ in results for outcome in outcomes)
    checkable = counts["matches store"] + counts["DIFFERS from store"]
    print(f"price check: {'on' if settings.price_check_enabled else 'off'}")
    print("links:", dict(counts))
    if checkable:
        print(f"shown prices matching the store: {counts['matches store']}/{checkable}")
    seconds = sorted(s for _, s in results if s) or [0.0]
    for line in DETAILS:
        print(line)
    print(f"seconds per answer: median {seconds[len(seconds) // 2]:.1f}, max {seconds[-1]:.1f}")


if __name__ == "__main__":
    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    main()
