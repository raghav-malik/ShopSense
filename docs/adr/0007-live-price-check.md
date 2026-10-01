# 7. Check answer prices against the live store pages

**Status:** Accepted (2026-10-01)

## Context

Shoppers saw prices that didn't match the store: the answer said ₹499, and the linked page said ₹749. An audit of 125 turns (196 price mentions) found the model **almost never invents prices**: 96% appeared in the search results or pages it read, and most of the rest were false alarms such as budgets echoed back. The prices were wrong because the *sources* were:

- **Stale or second-hand snippets.** Search engines return cached text, often from price-tracking and review sites ("₹13,999 on Flipkart, per a 91mobiles listing"), so the live price has moved.
- **Search pages instead of products.** 17% of links were search or category pages.
- **Different variants and sellers,** and products that are out of stock.

A prompt can't fix this: the model is faithfully repeating its sources. Before this change, `evals/price_accuracy.py` found **0 of 6** shown prices matched the store.

## Decision

After the answer is written and before it's shown, `app/agent/price_check.py` checks it in code:

- **Fetch each product page.** Every product link in the answer is fetched in parallel, through the SSRF-safe fetcher, with an 8-second limit.
- **Read the live price and stock.** The price comes from the store's structured product data, or from Amazon's buy box (a new parser, since Amazon publishes no structured price). Prices found in page description text aren't trusted.
- **Compare with the price stated next to the link,** to the rupee.
- **Act on the result:**
  - A wrong price is replaced, and a note lists the old and new prices.
  - An out-of-stock product is flagged.
  - An unreadable page is left as it was, marked "not checked".
  - Search pages are labelled as such.
- **Save the checked answer,** so the next turn and the cart see the corrected prices.
- **Show the outcome on each product card:** "✓ ₹1,949 · price checked on amazon.in just now", or "Price from search results; it couldn't be checked on the store".

## Consequences

- **The shown prices match the stores.** In `evals/price_accuracy.py`, 7 of 7 and 5 of 5 checkable prices matched, against 0 of 6 before. Out-of-stock recommendations are now flagged.
- **The cost:** about 4 seconds (median; at most 8) on answers with product links. Answers without links aren't affected.
- **Store coverage** is Flipkart, Amazon.in and stores that publish product data. Other sites go unchecked, and the cards say so.
- **It's a snapshot.** Prices can change after the check, which is why each card says *when* it was checked.
- **It can be switched off** with `PRICE_CHECK_ENABLED=false`, for example for a measurement run.
