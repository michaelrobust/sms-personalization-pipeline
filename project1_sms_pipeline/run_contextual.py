"""Contextual bandit demo + off-policy evaluation.

Pipeline:
  1. Sample N subscribers from the synthetic CRM.
  2. Run epsilon-greedy LinUCB online: at each step pick a variant given the
     subscriber's feature vector, observe a Bernoulli reward from the CTR
     simulator, log (context, arm, propensity, reward), update LinUCB.
  3. After the online loop, take the logged dataset and evaluate two target
     policies via SNIPS with bootstrap 95% CIs:
        - the trained LinUCB itself (as its epsilon-greedy distribution)
        - a uniform-random baseline
     Compare to the empirical online mean reward as a sanity check.

This is the standard production-RL gating pattern: log decisions with
propensities, then evaluate counterfactual policies offline before
committing to a rollout.

    python -m project1_sms_pipeline.run_contextual --n 1000 --epsilon 0.1
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .data.generate_subscribers import generate as generate_subscribers
from .optimization import (
    FeatureExtractor,
    LinUCB,
    LinUCBStore,
    LogEntry,
    PersonaCTRModel,
    bootstrap_ope,
    effective_sample_size,
    linucb_propensity_policy,
    uniform_random_policy,
)


ROOT = Path(__file__).parent
LOG_DIR = ROOT / "logs"


# Variant slate (4 tones x a representative CTA each). LinUCB learns which
# tone fits which user-feature pattern.
VARIANTS = [
    {"variant_id": "v_emp",   "tone": "empathetic", "cta_action": "Tap to come back — 20% off"},
    {"variant_id": "v_urg",   "tone": "urgent",     "cta_action": "Last chance — only a few left"},
    {"variant_id": "v_dir",   "tone": "direct",     "cta_action": "30% off — tap to save"},
    {"variant_id": "v_prem",  "tone": "premium",    "cta_action": "Members early access"},
]
VARIANTS_BY_ID = {v["variant_id"]: v for v in VARIANTS}


def _subscriber_dicts(n: int, seed: int = 7) -> list[dict]:
    rows = generate_subscribers(n=n, seed=seed)
    return [
        {
            "subscriber_id": r.subscriber_id,
            "segment_truth": r.segment_truth,
            "days_since_last_visit": r.days_since_last_visit,
            "purchase_count_90d": r.purchase_count_90d,
            "avg_order_value": r.avg_order_value,
            "last_category": r.last_category,
        }
        for r in rows
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000, help="number of online rounds")
    parser.add_argument("--alpha", type=float, default=1.0, help="LinUCB exploration constant")
    parser.add_argument("--epsilon", type=float, default=0.1, help="epsilon-greedy exploration")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap-iter", type=int, default=500)
    parser.add_argument(
        "--persist",
        action="store_true",
        help="Persist trained LinUCB state to logs/linucb_state.json",
    )
    args = parser.parse_args()

    LOG_DIR.mkdir(exist_ok=True)
    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    subscribers = _subscriber_dicts(args.n, seed=args.seed + 1)
    extractor = FeatureExtractor()
    bandit = LinUCB(
        arm_ids=[v["variant_id"] for v in VARIANTS],
        dim=extractor.DIM,
        alpha=args.alpha,
        epsilon=args.epsilon,
        rng=rng,
    )
    ctr_model = PersonaCTRModel()

    logs: list[LogEntry] = []
    cumulative_reward = 0
    arm_pull_counts: dict[str, int] = {v["variant_id"]: 0 for v in VARIANTS}

    for sub in subscribers:
        x = extractor.featurize(sub)
        chosen_arm, propensity, _props = bandit.select(x)
        v = VARIANTS_BY_ID[chosen_arm]
        # Reward: Bernoulli(true CTR for the cohort given this variant).
        true_ctr = ctr_model.true_ctr(sub["segment_truth"], v["tone"], v["cta_action"])
        reward = 1 if rng.random() < true_ctr else 0

        bandit.update(chosen_arm, x, reward)

        logs.append(
            LogEntry(context=x, arm_id=chosen_arm, propensity=propensity, reward=float(reward))
        )
        cumulative_reward += reward
        arm_pull_counts[chosen_arm] += 1

    online_mean = cumulative_reward / args.n

    # ---- Off-policy evaluation -------------------------------------------

    target_linucb = linucb_propensity_policy(bandit)
    target_uniform = uniform_random_policy(bandit.arm_ids)

    def best_arm_per_context(x: np.ndarray) -> str:
        idx = int(np.argmax(bandit._ucb_scores(x)))
        return bandit.arm_ids[idx]

    snips_linucb = bootstrap_ope(
        logs, target_linucb, estimator="snips", n_iter=args.bootstrap_iter, seed=args.seed
    )
    ips_linucb = bootstrap_ope(
        logs, target_linucb, estimator="ips", n_iter=args.bootstrap_iter, seed=args.seed
    )
    snips_uniform = bootstrap_ope(
        logs, target_uniform, estimator="snips", n_iter=args.bootstrap_iter, seed=args.seed
    )
    ess_linucb = effective_sample_size(logs, target_linucb)
    ess_uniform = effective_sample_size(logs, target_uniform)

    # Persist if requested.
    if args.persist:
        store = LinUCBStore(LOG_DIR / "linucb_state.json")
        store.save(bandit)

    # ---- Print + write summary -------------------------------------------

    lines = []
    lines.append("Contextual bandit run + off-policy evaluation")
    lines.append("=" * 60)
    lines.append(f"rounds                : {args.n}")
    lines.append(f"alpha (UCB scale)     : {args.alpha}")
    lines.append(f"epsilon (explore)     : {args.epsilon}")
    lines.append(f"feature dim           : {extractor.DIM}")
    lines.append("")
    lines.append("Online performance (logging policy):")
    lines.append(f"  cumulative reward   : {cumulative_reward}")
    lines.append(f"  empirical mean ctr  : {online_mean:.4f}")
    lines.append("")
    lines.append("Arm pulls (out of {}):".format(args.n))
    for v in VARIANTS:
        n_pulls = arm_pull_counts[v["variant_id"]]
        share = n_pulls / args.n
        lines.append(f"  {v['variant_id']:8s}  ({v['tone']:10s})  pulls={n_pulls:4d}  share={share:.1%}")
    lines.append("")
    lines.append("Off-policy estimates (95% bootstrap CI):")
    lines.append(
        f"  IPS    on logged LinUCB policy : {ips_linucb.estimate:.4f} "
        f"[{ips_linucb.ci_low:.4f}, {ips_linucb.ci_high:.4f}]"
    )
    lines.append(
        f"  SNIPS  on logged LinUCB policy : {snips_linucb.estimate:.4f} "
        f"[{snips_linucb.ci_low:.4f}, {snips_linucb.ci_high:.4f}]    "
        f"ESS={ess_linucb:.0f}/{args.n}"
    )
    lines.append(
        f"  SNIPS  on uniform-random target: {snips_uniform.estimate:.4f} "
        f"[{snips_uniform.ci_low:.4f}, {snips_uniform.ci_high:.4f}]    "
        f"ESS={ess_uniform:.0f}/{args.n}"
    )
    lines.append("")
    delta = snips_linucb.estimate - snips_uniform.estimate
    lift_pct = (delta / snips_uniform.estimate * 100) if snips_uniform.estimate > 0 else 0.0
    lines.append(
        f"  estimated lift LinUCB vs uniform: +{delta:.4f}  ({lift_pct:+.1f}%)"
    )
    lines.append("")
    lines.append("Per-arm theta (learned weights):")
    for arm_id in bandit.arm_ids:
        theta = bandit.theta(arm_id)
        lines.append(
            f"  {arm_id:8s}  pulls={bandit.states[arm_id].pulls:4d}  "
            f"theta=[{', '.join(f'{w:+.2f}' for w in theta)}]"
        )

    summary_text = "\n".join(lines)
    print(summary_text)

    out_path = LOG_DIR / "contextual_run.json"
    out_path.write_text(json.dumps({
        "rounds": args.n,
        "alpha": args.alpha,
        "epsilon": args.epsilon,
        "feature_dim": extractor.DIM,
        "online_mean_ctr": online_mean,
        "arm_pull_counts": arm_pull_counts,
        "ope": {
            "ips_linucb": asdict(ips_linucb),
            "snips_linucb": asdict(snips_linucb),
            "snips_uniform": asdict(snips_uniform),
            "ess_linucb": ess_linucb,
            "ess_uniform": ess_uniform,
        },
        "estimated_lift_vs_uniform": delta,
    }, indent=2))
    (LOG_DIR / "contextual_summary.txt").write_text(summary_text)
    print(f"\nSummary -> {out_path}")


if __name__ == "__main__":
    main()
