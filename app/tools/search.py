import asyncio
from urllib.parse import urlparse

from ddgs import DDGS
from ddgs.exceptions import DDGSException, RatelimitException, TimeoutException
from pydantic import BaseModel, Field

from app.config import settings
from app.llm.types import JSONObject
from app.tools.base import pydantic_to_tool_schema
from app.tools.untrusted import WEB_CONTENT_NOTICE, clean_text

# ShopSense prices in INR, so bias results toward Indian stores (ddgs defaults to us-en).
SEARCH_REGION = "in-en"

# Every result is resent to the LLM on each later step of the turn, so result
# size multiplies into cost and latency on any provider (and on Groq's free tier,
# 8K tokens/min, it decides whether a turn fits at all). Measured on real traces:
# cutting snippets at 400 chars kept 36/38 prices while dropping 36% of the text;
# a few 1-2.5K-char snippets were most of the bulk.
MAX_RESULTS = 5
SNIPPET_MAX_CHARS = 400
# ddgs joins titles when several engines return the same page (SR-70).
TITLE_MAX_CHARS = 200


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
        default=min(settings.max_search_results, MAX_RESULTS),
        ge=1,
        le=MAX_RESULTS,
        description=f"Number of results to return (1-{MAX_RESULTS}).",
    )


SEARCH_SCHEMA = pydantic_to_tool_schema(
    name="search_products",
    description="Search the web for products matching the user's requirements. ALWAYS call this before recommending any product. Never fabricate product names or prices.",
    input_model=SearchProductsInput,
)


async def search_products(query: str, max_results: int = settings.max_search_results) -> JSONObject:
    """
    Search DuckDuckGo for products. Returns structured results.

    ddgs is synchronous; it runs in a worker thread so a slow search doesn't
    block the event loop (and every other request) while it waits.

    Never raises: failures come back as {"error", "error_type", "hint", "results": []}.
    """
    no_results = {
        "results": [],
        "result_count": 0,
        "message": f"No products found for '{query}'. Try broadening the search or using different keywords.",
    }
    try:
        results = await asyncio.to_thread(_ddgs_text, query, max_results)

        if not results:
            return no_results

        formatted = []
        for r in results:
            url = r.get("href", "")
            formatted.append(
                {
                    "title": clean_text(r.get("title", ""), TITLE_MAX_CHARS),
                    "url": url,
                    "snippet": clean_text(r.get("body", ""), SNIPPET_MAX_CHARS),
                    "source": urlparse(url).netloc.removeprefix("www.") if url else "",
                }
            )

        return {
            "web_content_notice": WEB_CONTENT_NOTICE,
            "results": formatted,
            "result_count": len(formatted),
            "query_used": query,
        }

    except RatelimitException:
        return {
            "error": "The search engines are rate limiting us",
            "error_type": "rate_limited",
            "results": [],
            "hint": "Wait before searching again; answer from results you already have if you can.",
        }
    except TimeoutException:
        return {
            "error": "The search timed out",
            "error_type": "timeout",
            "results": [],
            "hint": "Retry once with a shorter, simpler query.",
        }
    except DDGSException as e:
        # ddgs raises (rather than returning []) when no engine finds anything.
        if "no results" in str(e).lower():
            return no_results
        return {
            "error": f"Search failed: {e}",
            "error_type": "search_error",
            "results": [],
            "hint": "Retry once with a different query.",
        }
    except Exception as e:  # noqa: BLE001 - any search failure becomes a result the agent can act on
        return {
            "error": f"Search failed: {type(e).__name__}",
            "error_type": "search_error",
            "results": [],
            "hint": "Retry once with a different query.",
        }


def _ddgs_text(query: str, max_results: int) -> list[JSONObject]:
    with DDGS() as ddgs:
        return ddgs.text(query, region=SEARCH_REGION, max_results=max_results)


if __name__ == "__main__":
    # Smoke test. From the project root:  python -m app.tools.search
    import io
    import json
    import sys

    if isinstance(sys.stdout, io.TextIOWrapper):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    print(json.dumps(SEARCH_SCHEMA, indent=2))
    print(json.dumps(asyncio.run(search_products("wireless earbuds under 3000 INR")), indent=2, ensure_ascii=False))
