import asyncio
from urllib.parse import urlparse

from ddgs import DDGS
from pydantic import BaseModel, Field

from app.config import settings
from app.tools.base import pydantic_to_tool_schema

# ShopSense prices in INR, so bias results toward Indian stores (ddgs defaults to us-en).
SEARCH_REGION = "in-en"


class SearchProductsInput(BaseModel):
    """Input schema for the search_products tool."""
    reasoning: str = Field(
        ...,
        description="Explain WHY you are searching with this query. What is the user looking for, and how does this query address their needs? This field is logged for tracing.",
    )
    query: str = Field(
        ...,
        description="Search query optimized for finding products. Include category, budget if known, and key requirements. Example: 'wireless earbuds under 3000 INR waterproof'",
    )
    max_results: int = Field(
        default=settings.max_search_results,
        ge=1,
        le=10,
        description="Number of results to return (1-10).",
    )


SEARCH_SCHEMA = pydantic_to_tool_schema(
    name="search_products",
    description="Search the web for products matching the user's requirements. ALWAYS call this before recommending any product. Never fabricate product names or prices.",
    input_model=SearchProductsInput,
)


async def search_products(query: str, max_results: int = settings.max_search_results) -> dict:
    """
    Search DuckDuckGo for products. Returns structured results.

    ddgs is synchronous; it runs in a worker thread so a slow search doesn't
    block the event loop (and every other request) while it waits.
    """
    try:
        results = await asyncio.to_thread(_ddgs_text, query, max_results)

        if not results:
            return {
                "results": [],
                "message": f"No products found for '{query}'. Try broadening the search or using different keywords.",
            }

        formatted = []
        for r in results:
            url = r.get("href", "")
            formatted.append({
                "title": r.get("title", ""),
                "url": url,
                "snippet": r.get("body", ""),
                "source": urlparse(url).netloc.removeprefix("www.") if url else "",
            })

        return {
            "results": formatted,
            "result_count": len(formatted),
            "query_used": query,
        }

    except Exception as e:
        return {"error": f"Search failed: {str(e)}", "results": []}


def _ddgs_text(query: str, max_results: int) -> list[dict]:
    with DDGS() as ddgs:
        return ddgs.text(query, region=SEARCH_REGION, max_results=max_results)


if __name__ == "__main__":
    # Smoke test. From the project root:  python -m app.tools.search
    import json
    import sys

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(json.dumps(SEARCH_SCHEMA, indent=2))
    print(json.dumps(asyncio.run(search_products("wireless earbuds under 3000 INR")), indent=2, ensure_ascii=False))
