from pydantic import BaseModel, Field

from app.llm.types import JSONObject
from app.tools.base import pydantic_to_tool_schema


class ProductForComparison(BaseModel):
    """A single product in a comparison request."""

    name: str
    price: str
    features: list[str] = Field(default_factory=list)
    rating: str | None = None
    url: str


class CompareProductsInput(BaseModel):
    """Input schema for the compare_products tool."""

    reasoning: str = Field(
        ...,
        description="Explain WHY you are comparing these products. What criteria matter most to the user?",
    )
    products: list[ProductForComparison] = Field(
        ...,
        min_length=2,
        max_length=5,
        description="List of products to compare (2-5). Each must have name, price, and url.",
    )


COMPARE_SCHEMA = pydantic_to_tool_schema(
    name="compare_products",
    description="Compare multiple products side by side. Call this after extracting info from at least 2 products. Returns a formatted comparison.",
    input_model=CompareProductsInput,
)


async def compare_products(products: list[JSONObject]) -> JSONObject:
    """
    Build a structured comparison from a list of products.
    Pure function — no external calls.
    """
    if len(products) < 2:
        return {"error": "Need at least 2 products to compare"}

    # Build comparison rows. `or` rather than a .get() default: a validated
    # product carries rating=None, which should still show as N/A.
    rows = [
        {
            "name": p.get("name") or "Unknown",
            "price": p.get("price") or "N/A",
            "features": ", ".join((p.get("features") or [])[:3]),
            "rating": p.get("rating") or "N/A",
            "buy_link": p.get("url", ""),
        }
        for p in products
    ]

    # Build markdown table
    header = "| Product | Price | Key Features | Rating | Link |"
    separator = "| --- | --- | --- | --- | --- |"
    table_rows = [
        f"| {_cell(r['name'])} | {_cell(r['price'])} | {_cell(r['features'])} | {_cell(r['rating'])} | [Buy]({r['buy_link']}) |"
        for r in rows
    ]

    comparison_table = "\n".join([header, separator, *table_rows])

    return {
        "comparison_table": comparison_table,
        "product_count": len(products),
        "products": rows,
    }


def _cell(value: object) -> str:
    """Escape pipes so a product name like 'Buds | 2024' doesn't split the table row."""
    return str(value).replace("|", "\\|")
