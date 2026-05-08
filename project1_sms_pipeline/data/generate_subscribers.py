"""Generate the synthetic subscriber CSV used by the pipeline + eval."""
from __future__ import annotations

import csv
import random
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

OUT_PATH = Path(__file__).parent / "subscribers.csv"

# (label, weight, recency_days_mu, freq_mu, monetary_mu)
SEGMENTS = [
    ("high_intent_browser", 0.28, 5, 1, 0),
    ("price_sensitive",     0.22, 18, 4, 65),
    ("vip_loyalist",        0.10, 12, 22, 850),
    ("new_subscriber",      0.18, 3, 0, 0),
    ("post_purchase",       0.14, 8, 5, 140),
    ("winback_dormant",     0.08, 95, 9, 220),
]

CATEGORIES = ["apparel", "beauty", "home", "electronics", "wellness", "kids"]
CHANNELS = ["popup", "checkout_optin", "loyalty_signup", "referral", "paid_social"]


@dataclass
class Subscriber:
    subscriber_id: str
    segment_truth: str
    days_since_last_visit: int
    purchase_count_90d: int
    avg_order_value: float
    last_category: str
    subscribe_channel: str
    is_loyalty_member: bool
    state: str
    last_engagement_date: str


def _draw(rng: random.Random, mu: float, sigma_frac: float = 0.4, lo: float = 0.0) -> float:
    val = rng.gauss(mu, max(mu * sigma_frac, 1.0))
    return max(lo, val)


def generate(n: int = 5000, seed: int = 7) -> list[Subscriber]:
    rng = random.Random(seed)
    out: list[Subscriber] = []
    today = date(2026, 5, 1)

    weights = [w for (_, w, *_rest) in SEGMENTS]
    labels = [s for (s, *_rest) in SEGMENTS]

    for i in range(n):
        seg = rng.choices(labels, weights=weights, k=1)[0]
        params = next(p for p in SEGMENTS if p[0] == seg)
        _, _, recency_mu, freq_mu, monetary_mu = params

        recency = int(_draw(rng, recency_mu))
        freq = int(_draw(rng, freq_mu))
        aov = round(_draw(rng, monetary_mu if monetary_mu > 0 else 35, sigma_frac=0.5), 2)
        last_eng = today - timedelta(days=recency)

        out.append(
            Subscriber(
                subscriber_id=f"sub_{i:05d}",
                segment_truth=seg,
                days_since_last_visit=recency,
                purchase_count_90d=freq,
                avg_order_value=aov,
                last_category=rng.choice(CATEGORIES),
                subscribe_channel=rng.choice(CHANNELS),
                is_loyalty_member=(seg == "vip_loyalist") or (rng.random() < 0.15),
                state=rng.choice(
                    ["CA", "NY", "TX", "FL", "IL", "WA", "MA", "PA", "OH", "GA"]
                ),
                last_engagement_date=last_eng.isoformat(),
            )
        )
    return out


def write_csv(rows: list[Subscriber], path: Path = OUT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))
    return path


def segment_distribution(rows: list[Subscriber]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        out[r.segment_truth] = out.get(r.segment_truth, 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


if __name__ == "__main__":
    rows = generate()
    write_csv(rows)
    print(f"Wrote {len(rows)} subscribers -> {OUT_PATH}")
    for seg, n in segment_distribution(rows).items():
        print(f"  {seg:24s} {n:5d}  ({n/len(rows)*100:.1f}%)")
