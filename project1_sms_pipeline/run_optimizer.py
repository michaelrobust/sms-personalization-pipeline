"""Variant-selection demo: Thompson sampling against simulated CTR.

Generates 4 candidate variants per cohort (mix of tones / CTA styles), then
runs T rounds of Thompson sampling. Reports cumulative regret and which arm
the bandit converged on, vs the hidden ground-truth best arm.

This closes the "generate copy -> send -> learn -> resend" loop that real
SMS marketing platforms run on.

    python -m project1_sms_pipeline.run_optimizer --cohort winback_dormant --rounds 500
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .optimization import (
    BanditStore,
    PersonaCTRModel,
    run_bandit_loop,
)
from .optimization.simulator import format_run_summary, COHORT_TRUE_PREFERENCES


ROOT = Path(__file__).parent
LOG_DIR = ROOT / "logs"


# Pre-defined candidate variant slates per cohort. In production these would
# come from the variant-generation agent. Hand-written here so the demo runs
# without an API key for the bandit phase.
DEMO_VARIANT_SLATES: dict[str, list[dict[str, str]]] = {
    "winback_dormant": [
        {"variant_id": "wb_v1", "tone": "empathetic",
         "cta_action": "Tap to come back — 20% off"},
        {"variant_id": "wb_v2", "tone": "playful",
         "cta_action": "Tap to claim"},
        {"variant_id": "wb_v3", "tone": "urgent",
         "cta_action": "Last chance to save"},
        {"variant_id": "wb_v4", "tone": "premium",
         "cta_action": "VIP early access"},
    ],
    "high_intent_browser": [
        {"variant_id": "hi_v1", "tone": "urgent",
         "cta_action": "Only 3 left in stock — tap now"},
        {"variant_id": "hi_v2", "tone": "warm",
         "cta_action": "Hope you find what you need"},
        {"variant_id": "hi_v3", "tone": "direct",
         "cta_action": "Free shipping today"},
        {"variant_id": "hi_v4", "tone": "premium",
         "cta_action": "Curated for you"},
    ],
    "vip_loyalist": [
        {"variant_id": "vip_v1", "tone": "premium",
         "cta_action": "Members early access — tap to unlock"},
        {"variant_id": "vip_v2", "tone": "direct",
         "cta_action": "Save 30% today"},
        {"variant_id": "vip_v3", "tone": "playful",
         "cta_action": "Treat yourself"},
        {"variant_id": "vip_v4", "tone": "urgent",
         "cta_action": "Limited stock"},
    ],
    "price_sensitive": [
        {"variant_id": "ps_v1", "tone": "direct",
         "cta_action": "30% off — tap to save"},
        {"variant_id": "ps_v2", "tone": "warm",
         "cta_action": "Hope you find something nice"},
        {"variant_id": "ps_v3", "tone": "premium",
         "cta_action": "Exclusive access"},
        {"variant_id": "ps_v4", "tone": "playful",
         "cta_action": "Fancy a deal?"},
    ],
    "new_subscriber": [
        {"variant_id": "ns_v1", "tone": "warm",
         "cta_action": "Welcome — tap for your first reward"},
        {"variant_id": "ns_v2", "tone": "urgent",
         "cta_action": "Last call"},
        {"variant_id": "ns_v3", "tone": "direct",
         "cta_action": "Shop now"},
        {"variant_id": "ns_v4", "tone": "premium",
         "cta_action": "Curated picks"},
    ],
    "post_purchase": [
        {"variant_id": "pp_v1", "tone": "warm",
         "cta_action": "Goes great with your last order"},
        {"variant_id": "pp_v2", "tone": "urgent",
         "cta_action": "Selling out fast"},
        {"variant_id": "pp_v3", "tone": "premium",
         "cta_action": "Members early access"},
        {"variant_id": "pp_v4", "tone": "playful",
         "cta_action": "More to love"},
    ],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cohort",
        default="winback_dormant",
        choices=list(DEMO_VARIANT_SLATES),
    )
    parser.add_argument("--rounds", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--persist",
        action="store_true",
        help="Persist bandit state to logs/bandit_state.json across runs.",
    )
    args = parser.parse_args()

    LOG_DIR.mkdir(exist_ok=True)
    variants = DEMO_VARIANT_SLATES[args.cohort]
    store = (
        BanditStore(LOG_DIR / "bandit_state.json") if args.persist else None
    )

    result = run_bandit_loop(
        cohort_id=args.cohort,
        variants=variants,
        n_rounds=args.rounds,
        store=store,
        seed=args.seed,
    )

    # Compute ground-truth CTRs for the report.
    model = PersonaCTRModel()
    true_ctrs = {
        v["variant_id"]: round(
            model.true_ctr(args.cohort, v["tone"], v["cta_action"]), 4
        )
        for v in variants
    }
    best_true_id = max(true_ctrs, key=true_ctrs.get)
    converged = result.best_arm_id == best_true_id

    text = format_run_summary(result, true_ctrs=true_ctrs)
    print(text)
    print()
    if converged:
        print(f"  -> bandit identified the true-best arm ({best_true_id}).")
    else:
        print(
            f"  -> bandit picked {result.best_arm_id}, true best is {best_true_id} "
            f"(might converge with more rounds)."
        )

    summary = {
        "cohort": args.cohort,
        "rounds": args.rounds,
        "cumulative_reward": result.cumulative_reward,
        "cumulative_regret": result.cumulative_regret,
        "best_arm_chosen": result.best_arm_id,
        "best_arm_true": best_true_id,
        "converged_to_optimum": converged,
        "true_ctrs": true_ctrs,
        "final_arms": result.arms,
        "trace": [asdict(s) for s in result.steps],
    }
    out_path = LOG_DIR / f"bandit_run_{args.cohort}.json"
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"  -> trace written to {out_path}")


if __name__ == "__main__":
    main()
