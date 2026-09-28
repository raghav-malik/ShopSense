"""The manage_cart tool: add, remove, view or clear the session's cart."""

from typing import Literal, Self

from pydantic import BaseModel, Field, model_validator
from pydantic.json_schema import SkipJsonSchema

from app.db import queries
from app.llm.types import JSONObject
from app.tools.base import pydantic_to_tool_schema


class ManageCartInput(BaseModel):
    """Input schema for the manage_cart tool."""

    reasoning: str = Field(
        ...,
        description="Explain WHY you are performing this cart action. What did the user ask for?",
    )
    action: Literal["add", "remove", "view", "clear"] = Field(
        ...,
        description="Cart operation to perform",
    )
    # Injected by the registry. SkipJsonSchema keeps it out of the schema the
    # LLM sees, so the model never tries to invent a session id.
    session_id: SkipJsonSchema[str] = Field(default="", description="Injected by the registry, not sent by the LLM")
    product_name: str | None = Field(default=None, description="Product name (required for add/remove)")
    price: float | None = Field(default=None, description="Product price (required for add)")
    url: str | None = Field(default=None, description="Product URL (required for add)")
    source: str | None = Field(default=None, description="Source website, e.g. 'amazon.in'")

    @model_validator(mode="after")
    def check_required_fields(self) -> Self:
        """Validate that add/remove have the fields they need."""
        if self.action == "add" and (not self.product_name or not self.url):
            raise ValueError("product_name and url are required when action is 'add'")
        if self.action == "remove" and not self.product_name:
            raise ValueError("product_name is required when action is 'remove'")
        return self


CART_SCHEMA = pydantic_to_tool_schema(
    name="manage_cart",
    description="Add, remove, view, or clear items in the user's shopping cart. Use 'add' when the user wants to save a product, 'view' to show cart contents, 'remove' to remove a specific product, 'clear' to empty the cart.",
    input_model=ManageCartInput,
)


async def manage_cart(
    action: str,
    session_id: str,
    product_name: str | None = None,
    price: float | None = None,
    url: str | None = None,
    source: str | None = None,
) -> JSONObject:
    """Execute a cart operation."""

    if action == "add":
        if not product_name or not url:
            return {"error": "product_name and url are required for add"}
        await queries.add_to_cart(session_id, product_name, price, url, source)
        cart = await queries.get_cart(session_id)
        total = sum(i.get("price", 0) or 0 for i in cart)
        return {
            "message": f"Added {product_name} to cart.",
            "cart_size": len(cart),
            "total": total,
            "currency": "INR",
        }

    elif action == "remove":
        if not product_name:
            return {"error": "product_name is required for remove"}
        removed = await queries.remove_from_cart(session_id, product_name)
        cart = await queries.get_cart(session_id)
        total = sum(i.get("price", 0) or 0 for i in cart)
        return {
            "message": f"{'Removed' if removed else 'Could not find'} {product_name}.",
            "cart_size": len(cart),
            "total": total,
        }

    elif action == "view":
        cart = await queries.get_cart(session_id)
        total = sum(i.get("price", 0) or 0 for i in cart)
        items = [{"name": i["product_name"], "price": i["price"], "url": i["url"]} for i in cart]
        return {
            "items": items,
            "cart_size": len(items),
            "total": total,
            "currency": "INR",
        }

    elif action == "clear":
        await queries.clear_cart(session_id)
        return {"message": "Cart cleared.", "cart_size": 0, "total": 0}

    return {"error": f"Unknown cart action: {action}"}
