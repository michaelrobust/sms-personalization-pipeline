"""Cohort-learning lookup. Most-specific (segment, last_category) wins."""
from __future__ import annotations

from dataclasses import dataclass

# Per-cohort copy guidance. Add a row to extend; no code change needed.
CONTEXT_SNIPPETS: dict[tuple[str, str | None], str] = {
    ("winback_dormant", None): (
        "Subscriber inactive 60+ days. Acknowledge absence without guilting; "
        "lead with a tangible incentive (free shipping or % off); single-tap CTA."
    ),
    ("winback_dormant", "apparel"): (
        "Apparel winback: reference seasonal turnover ('new arrivals'); items "
        "from past carts are likely out of stock or out of season."
    ),
    ("winback_dormant", "beauty"): (
        "Beauty winback: replenishment framing ('time to restock?'). Skip "
        "promoting brand-new lines they have no relationship with."
    ),
    ("vip_loyalist", None): (
        "VIPs respond to access, not discounts. Lead with early access, "
        "members-only drops, or concierge framing."
    ),
    ("price_sensitive", None): (
        "Lead with the % off or dollar value in the first 40 chars. Soft "
        "framing depresses CTR for this cohort."
    ),
    ("new_subscriber", None): (
        "First 1-2 sends should establish brand voice and confirm welcome "
        "promise. Aggressive promo before that erodes trust."
    ),
    ("high_intent_browser", None): (
        "Subscriber visited within 7 days. Reference what they were looking "
        "at if data permits; otherwise use scarcity (stock left, time left)."
    ),
    ("post_purchase", None): (
        "Subscriber in satisfaction window. Avoid promoting competing "
        "categories; cross-sell complementary items or reinforce confidence."
    ),
}


@dataclass
class ContextSnippet:
    text: str
    source_key: tuple[str, str | None]


def retrieve_context(segment: str, last_category: str | None = None) -> ContextSnippet | None:
    if last_category is not None:
        key = (segment, last_category)
        if key in CONTEXT_SNIPPETS:
            return ContextSnippet(text=CONTEXT_SNIPPETS[key], source_key=key)
    key = (segment, None)
    if key in CONTEXT_SNIPPETS:
        return ContextSnippet(text=CONTEXT_SNIPPETS[key], source_key=key)
    return None
