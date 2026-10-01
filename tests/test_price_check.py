"""Answer prices are checked against the live store pages before they're shown."""

import asyncio

import pytest

import app.agent.core as core
import app.agent.price_check as price_check
from app.agent.price_check import check_prices
from app.config import settings
from app.db import queries
from app.db.models import Session
from app.llm.types import LLMResponse
from tests.test_agent import FakeLLM, answer

AMAZON = "https://www.amazon.in/boAt-Airdopes-141/dp/B09N3ZNHTY"
FLIPKART = "https://www.flipkart.com/noise-buds-vs104/p/itm123"
SEARCH = "https://www.amazon.in/s?k=wireless+earbuds"
# Captured at import, before the autouse fixture stubs it for every test.
REAL_LIVE_DETAILS = price_check._live_details


class Stores:
    """Fake store pages: url -> (live price, available, product name), and which URLs were fetched."""

    def __init__(self) -> None:
        self.pages: dict[str, tuple[float | None, bool | None, str | None]] = {}
        self.fetched: list[str] = []

    def __setitem__(self, url: str, details: tuple[float | None, bool | None, str | None]) -> None:
        self.pages[url] = details

    async def live(self, url: str) -> tuple[float | None, bool | None, str | None]:
        self.fetched.append(url)
        return self.pages.get(url, (None, None, None))


@pytest.fixture
def stores(monkeypatch: pytest.MonkeyPatch) -> Stores:
    fake = Stores()
    monkeypatch.setattr(price_check, "_live_details", fake.live)
    return fake


async def test_matching_price_is_verified_and_left_alone(stores: Stores) -> None:
    stores[AMAZON] = (1099.0, True, "boAt Airdopes 141")
    text = f"**boAt Airdopes 141** — ₹1,099 on Amazon.in. [Buy on Amazon.in]({AMAZON})"
    checked, checks = await check_prices(text)
    assert checked == text
    assert [(c.status, c.stated_price, c.live_price) for c in checks] == [("verified", 1099.0, 1099.0)]


async def test_a_wrong_price_is_corrected_with_a_note(stores: Stores) -> None:
    stores[AMAZON] = (749.0, True, "boAt Airdopes 141")
    text = f"**boAt Airdopes 141** — ₹499 per a PriceHistory listing. [Buy on Amazon.in]({AMAZON})"
    checked, checks = await check_prices(text)
    assert "₹749 per a PriceHistory listing" in checked and "₹499 per" not in checked
    assert "_Prices checked on the store pages just now: **boAt Airdopes 141** is ₹749 on amazon.in" in checked
    assert "(search results said ₹499)" in checked
    assert checks[0].status == "corrected"


async def test_price_on_the_line_above_the_link(stores: Stores) -> None:
    stores[AMAZON] = (1599.0, True, "Razer DeathAdder Essential")
    text = f"**Best fit: Razer DeathAdder Essential — ₹1,499**\n[Buy on Amazon.in]({AMAZON})"
    checked, checks = await check_prices(text)
    assert checks[0].status == "corrected" and "Razer DeathAdder Essential — ₹1,599" in checked


async def test_a_blank_line_ends_the_item(stores: Stores) -> None:
    stores[AMAZON] = (1599.0, True, "Razer")
    text = f"Budget: under ₹1,500.\n\n[Buy on Amazon.in]({AMAZON})"  # the ₹1,500 isn't this product's price
    checked, checks = await check_prices(text)
    assert checked == text and checks[0].status == "verified" and checks[0].stated_price is None


async def test_table_rows(stores: Stores) -> None:
    stores[AMAZON] = (799.0, True, "boAt")
    stores[FLIPKART] = (1099.0, True, "Noise")
    text = (
        "| Earbuds | Price |\n|---|---|\n"
        f"| [boAt Airdopes 141]({AMAZON}) | ₹799 on Amazon.in |\n"
        f"| [Noise Buds VS104]({FLIPKART}) | ₹999 on Flipkart |"
    )
    checked, checks = await check_prices(text)
    assert [c.status for c in checks] == ["verified", "corrected"]
    assert "| ₹1,099 on Flipkart |" in checked and "**Noise Buds VS104** is ₹1,099 on flipkart.com" in checked


async def test_unavailable_products_are_flagged(stores: Stores) -> None:
    stores[AMAZON] = (None, False, "realme Buds Air 5")
    checked, checks = await check_prices(f"**realme Buds Air 5** — ₹2,399. [Buy]({AMAZON})")
    assert checks[0].status == "unavailable"
    assert "**realme Buds Air 5** is currently unavailable on amazon.in." in checked


async def test_unreadable_pages_are_unchecked_and_unchanged(stores: Stores) -> None:
    text = f"**boAt** — ₹1,099. [Buy]({AMAZON})"
    checked, checks = await check_prices(text)  # no page registered: no price found
    assert checked == text and checks[0].status == "unchecked"


async def test_search_pages_are_not_fetched(stores: Stores) -> None:
    _, checks = await check_prices(f"Earbuds from ₹999: [see options on Amazon.in]({SEARCH})")
    assert checks[0].status == "search_page"
    assert stores.fetched == []


async def test_slow_pages_are_unchecked(monkeypatch: pytest.MonkeyPatch) -> None:
    async def slow(url: str) -> tuple[float | None, bool | None, str | None]:
        await asyncio.sleep(3600)
        return 1.0, True, "x"

    monkeypatch.setattr(price_check, "_live_details", slow)
    monkeypatch.setattr(settings, "price_check_timeout", 0.05)
    checked, checks = await check_prices(f"**boAt** — ₹1,099. [Buy]({AMAZON})")
    assert checks[0].status == "unchecked" and "₹1,099" in checked


async def test_the_check_can_be_turned_off(stores: Stores, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "price_check_enabled", False)
    stores[AMAZON] = (749.0, True, "boAt")
    text = f"**boAt** — ₹499. [Buy]({AMAZON})"
    assert await check_prices(text) == (text, [])


async def test_generic_link_text_uses_the_page_name(stores: Stores) -> None:
    stores[AMAZON] = (749.0, True, "boAt Airdopes 141 Bluetooth TWS Earbuds")
    checked, _ = await check_prices(f"Great value at ₹499. [Buy on Amazon.in]({AMAZON})")
    assert "**boAt Airdopes 141 Bluetooth TWS Earbuds** is ₹749" in checked


# ---- in the agent ----


@pytest.fixture
async def session(db: None) -> Session:
    return await queries.create_session()


async def test_the_agent_shows_and_saves_the_checked_price(session: Session, stores: Stores) -> None:
    stores[AMAZON] = (749.0, True, "boAt Airdopes 141")
    reply: LLMResponse = answer(f"**boAt Airdopes 141** — ₹499. [Buy on Amazon.in]({AMAZON})")
    llm = FakeLLM(reply)
    result = await core.run_agent(session.id, "cheap earbuds", llm=llm, small_llm=llm)
    assert "₹749" in result.response and "₹499." not in result.response
    assert [(c.url, c.status, c.live_price) for c in result.price_checks] == [(AMAZON, "corrected", 749.0)]
    saved = await queries.get_messages(session.id)
    assert "₹749" in saved[-1]["content"]  # the next turn (and the cart) sees the corrected price


@pytest.mark.parametrize(
    ("source", "trusted"),
    [("structured_data", True), ("amazon_buy_box", True), ("open_graph", True), ("page_description", False)],
)
async def test_only_structured_store_prices_are_trusted(
    monkeypatch: pytest.MonkeyPatch, source: str, trusted: bool
) -> None:
    async def page(url: str) -> dict[str, object]:
        return {"name": "Running shoes under ₹3,000", "price": "₹3,000", "price_source": source, "available": None}

    monkeypatch.setattr(price_check, "extract_product_info", page)
    live_price, _, _ = await REAL_LIVE_DETAILS("https://in.puma.com/in/en/shop/shoes-under-3000")
    assert (live_price == 3000.0) is trusted


async def test_even_small_differences_are_corrected(stores: Stores) -> None:
    stores[AMAZON] = (8965.0, True, "Sony WH-CH720N")
    checked, checks = await check_prices(f"**Sony WH-CH720N** — ₹8,929. [Buy]({AMAZON})")
    assert checks[0].status == "corrected" and "₹8,965" in checked


async def test_paise_are_not_a_difference(stores: Stores) -> None:
    stores[AMAZON] = (1099.0, True, "boAt")
    _, checks = await check_prices(f"**boAt** — ₹1,099.00. [Buy]({AMAZON})")
    assert checks[0].status == "verified"


async def test_long_page_names_are_cut_at_a_word(stores: Stores) -> None:
    stores[AMAZON] = (5799.0, True, "JBL Tune 770NC Wireless Over Ear Headphones with Adaptive Noise Cancellation")
    checked, checks = await check_prices(f"Great pick at ₹4,799. [Buy on Amazon.in]({AMAZON})")
    assert checks[0].product == "JBL Tune 770NC Wireless Over Ear Headphones with Adaptive …"
    assert "(search results said ₹4,799)" in checked
