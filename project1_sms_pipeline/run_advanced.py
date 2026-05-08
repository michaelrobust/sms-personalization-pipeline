"""Advanced OPE + sequential testing + counterfactual policy learning demo.

Three sections, each printed on stdout and written to logs/advanced_*.json:

  1. Variance comparison: IPS vs SNIPS vs cross-fitted DR on the same logs.
     DR's CI should be tighter than IPS's because the reward-model term acts
     as a control variate.

  2. Always-valid sequential A/B test: simulate a fixed-horizon t-test with
     peeking and compare to a Hoeffding confidence sequence. Then stress-test
     type-I error under H0 (p_A == p_B) — the CS should hold alpha; the
     fixed-horizon-with-peeking baseline does not.

  3. Counterfactual policy learning (POEM): collect logs under uniform random
     logging policy, train a softmax-linear policy by minimizing the
     variance-regularized IPS objective, evaluate via DR on a held-out fold.
     Compare the trained policy to the logging policy.

    python -m project1_sms_pipeline.run_advanced --n 2000
"""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .data.generate_subscribers import generate as generate_subscribers
from .optimization import (
    FeatureExtractor,
    LinUCB,
    LogEntry,
    PersonaCTRModel,
    bootstrap_dr,
    bootstrap_ope,
    doubly_robust_estimate,
    effective_sample_size,
    hoeffding_cs,
    linucb_propensity_policy,
    run_sequential_ab,
    simulate_type_i_error,
    snips_estimate,
    softmax_policy_propensity,
    train_poem,
    uniform_random_policy,
)


ROOT = Path(__file__).parent
LOG_DIR = ROOT / "logs"


VARIANTS = [
    {"variant_id": "v_emp",  "tone": "empathetic", "cta_action": "Tap to come back — 20% off"},
    {"variant_id": "v_urg",  "tone": "urgent",     "cta_action": "Last chance — only a few left"},
    {"variant_id": "v_dir",  "tone": "direct",     "cta_action": "30% off — tap to save"},
    {"variant_id": "v_prem", "tone": "premium",    "cta_action": "Members early access"},
]
ARM_IDS = [v["variant_id"] for v in VARIANTS]
VARIANTS_BY_ID = {v["variant_id"]: v for v in VARIANTS}


def _subscriber_dicts(n: int, seed: int) -> list[dict]:
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


def _collect_logs_uniform(
    subscribers: list[dict],
    extractor: FeatureExtractor,
    rng: random.Random,
) -> tuple[list[LogEntry], float]:
    """Logging policy: uniform random over arms."""
    K = len(ARM_IDS)
    ctr_model = PersonaCTRModel()
    logs: list[LogEntry] = []
    total_reward = 0.0
    for sub in subscribers:
        x = extractor.featurize(sub)
        arm = ARM_IDS[rng.randrange(K)]
        v = VARIANTS_BY_ID[arm]
        ctr = ctr_model.true_ctr(sub["segment_truth"], v["tone"], v["cta_action"])
        r = 1.0 if rng.random() < ctr else 0.0
        logs.append(LogEntry(context=x, arm_id=arm, propensity=1.0 / K, reward=r))
        total_reward += r
    return logs, total_reward / max(1, len(subscribers))


def _collect_logs_linucb(
    subscribers: list[dict],
    extractor: FeatureExtractor,
    rng: random.Random,
    epsilon: float = 0.1,
    alpha: float = 1.0,
) -> tuple[list[LogEntry], float, LinUCB]:
    bandit = LinUCB(arm_ids=ARM_IDS, dim=extractor.DIM, alpha=alpha, epsilon=epsilon, rng=rng)
    ctr_model = PersonaCTRModel()
    logs: list[LogEntry] = []
    total_reward = 0.0
    for sub in subscribers:
        x = extractor.featurize(sub)
        chosen, prop, _ = bandit.select(x)
        v = VARIANTS_BY_ID[chosen]
        ctr = ctr_model.true_ctr(sub["segment_truth"], v["tone"], v["cta_action"])
        r = 1.0 if rng.random() < ctr else 0.0
        bandit.update(chosen, x, r)
        logs.append(LogEntry(context=x, arm_id=chosen, propensity=prop, reward=r))
        total_reward += r
    return logs, total_reward / max(1, len(subscribers)), bandit


# ---- Section 1: variance comparison ---------------------------------------


def section_1_dr_variance(logs: list[LogEntry], target_policy, bootstrap_iter: int):
    print()
    print("Section 1 — IPS vs SNIPS vs cross-fitted DR (target = logging policy LinUCB)")
    print("-" * 72)
    snips = bootstrap_ope(logs, target_policy, estimator="snips", n_iter=bootstrap_iter, seed=0)
    ips = bootstrap_ope(logs, target_policy, estimator="ips", n_iter=bootstrap_iter, seed=0)
    dr = bootstrap_dr(logs, target_policy, n_iter=bootstrap_iter, n_folds=5, lam=1.0, seed=0)

    def width(r):
        return r.ci_high - r.ci_low

    print(f"  IPS    : {ips.estimate:.4f}  CI [{ips.ci_low:.4f}, {ips.ci_high:.4f}]  width={width(ips):.4f}")
    print(f"  SNIPS  : {snips.estimate:.4f}  CI [{snips.ci_low:.4f}, {snips.ci_high:.4f}]  width={width(snips):.4f}")
    print(f"  DR(xfit) : {dr.estimate:.4f}  CI [{dr.ci_low:.4f}, {dr.ci_high:.4f}]  width={width(dr):.4f}")
    print(f"  DR width / SNIPS width = {width(dr) / max(1e-9, width(snips)):.2f}")
    return {
        "ips": asdict(ips),
        "snips": asdict(snips),
        "dr_xfit": asdict(dr),
        "snips_width": width(snips),
        "dr_width": width(dr),
        "dr_to_snips_width_ratio": width(dr) / max(1e-9, width(snips)),
    }


# ---- Section 2: sequential testing ----------------------------------------


def section_2_sequential(rng: random.Random, n_trials: int = 200, horizon: int = 600,
                         p_shared: float = 0.10, p_alt: float = 0.13):
    print()
    print("Section 2 — Always-valid CS vs fixed-horizon test (peeking exploit)")
    print("-" * 72)

    # Type-I error stress test under H0 (p_A == p_B == p_shared).
    type_i_cs = simulate_type_i_error(
        n_trials=n_trials, horizon=horizon, p_shared=p_shared, alpha=0.05, seed=42
    )
    # Compare to "fixed-horizon t-test with peeking" baseline: at every step,
    # check whether |Δhat| / SE > 1.96. This INTENTIONALLY mirrors the bug
    # marketers commit when they peek at fixed-horizon tests.
    n_false_peek = 0
    for trial in range(n_trials):
        n_a = sum_a = n_b = sum_b = 0
        rejected = False
        for _ in range(horizon):
            arm = "A" if rng.random() < 0.5 else "B"
            r = 1 if rng.random() < p_shared else 0
            if arm == "A":
                n_a += 1; sum_a += r
            else:
                n_b += 1; sum_b += r
            if n_a >= 30 and n_b >= 30:
                pa = sum_a / n_a; pb = sum_b / n_b
                se = np.sqrt(pa * (1 - pa) / n_a + pb * (1 - pb) / n_b)
                if se > 0 and abs(pa - pb) / se > 1.96:
                    rejected = True
                    break
        if rejected:
            n_false_peek += 1
    fixed_peek_alpha = n_false_peek / n_trials
    print(f"  CS (always-valid) empirical type-I error: {type_i_cs.empirical_alpha:.3f}  (target {type_i_cs.target_alpha})")
    print(f"  Fixed-horizon test + peeking type-I error: {fixed_peek_alpha:.3f}  (should be inflated)")

    # Power demo: under H1 (p_A=p_alt > p_B=p_shared), at what horizon does
    # the always-valid CS reject?
    arms, rewards = [], []
    for _ in range(horizon):
        arm = "A" if rng.random() < 0.5 else "B"
        p_true = p_alt if arm == "A" else p_shared
        arms.append(arm)
        rewards.append(1.0 if rng.random() < p_true else 0.0)
    decision = run_sequential_ab(arms, rewards, alpha=0.05, min_per_arm=30)
    print(f"  Under H1 ({p_alt} vs {p_shared}): CS stops at t={decision.stopped_at}, "
          f"decision={decision.decision}, Δ̂={decision.final_diff:.3f}, CI={tuple(round(x, 3) for x in decision.final_ci)}")

    return {
        "cs_empirical_type_i": type_i_cs.empirical_alpha,
        "cs_target_alpha": type_i_cs.target_alpha,
        "fixed_peeking_empirical_type_i": fixed_peek_alpha,
        "h1_demo": {
            "stopped_at": decision.stopped_at,
            "decision": decision.decision,
            "final_diff": decision.final_diff,
            "final_ci": list(decision.final_ci),
        },
    }


# ---- Section 3: counterfactual policy learning ----------------------------


def section_3_poem(
    train_logs: list[LogEntry],
    eval_logs: list[LogEntry],
    bootstrap_iter: int,
):
    print()
    print("Section 3 — POEM counterfactual policy learning")
    print("-" * 72)

    policy = train_poem(train_logs, arm_ids=ARM_IDS, lam_var=0.05, l2=1e-3, seed=0)
    target = softmax_policy_propensity(policy)
    target_uniform = uniform_random_policy(ARM_IDS)

    snips_target = bootstrap_ope(
        eval_logs, target, estimator="snips", n_iter=bootstrap_iter, seed=0
    )
    snips_uniform = bootstrap_ope(
        eval_logs, target_uniform, estimator="snips", n_iter=bootstrap_iter, seed=0
    )
    dr_target = bootstrap_dr(
        eval_logs, target, n_iter=bootstrap_iter, n_folds=5, seed=0
    )

    print(f"  V(uniform random)                : {snips_uniform.estimate:.4f}  "
          f"CI [{snips_uniform.ci_low:.4f}, {snips_uniform.ci_high:.4f}]")
    print(f"  V(POEM-learned policy)  via SNIPS: {snips_target.estimate:.4f}  "
          f"CI [{snips_target.ci_low:.4f}, {snips_target.ci_high:.4f}]")
    print(f"  V(POEM-learned policy)  via DR   : {dr_target.estimate:.4f}  "
          f"CI [{dr_target.ci_low:.4f}, {dr_target.ci_high:.4f}]")
    lift = snips_target.estimate - snips_uniform.estimate
    print(f"  Counterfactual lift (POEM vs uniform): +{lift:.4f}  "
          f"({lift / max(1e-9, snips_uniform.estimate) * 100:+.1f}%)")

    return {
        "v_uniform_snips": asdict(snips_uniform),
        "v_poem_snips": asdict(snips_target),
        "v_poem_dr": asdict(dr_target),
        "counterfactual_lift_vs_uniform": lift,
        "policy_theta_shape": list(policy.theta.shape),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap-iter", type=int, default=200)
    args = parser.parse_args()

    LOG_DIR.mkdir(exist_ok=True)
    rng = random.Random(args.seed)
    extractor = FeatureExtractor()

    # Same subscribers, two log streams: half for training, half for eval.
    subs = _subscriber_dicts(args.n, seed=args.seed + 1)
    half = len(subs) // 2
    train_subs, eval_subs = subs[:half], subs[half:]

    print(f"Subscribers: {len(subs)} (train={half}, eval={len(eval_subs)})")
    train_logs, train_mean = _collect_logs_uniform(train_subs, extractor, rng)
    eval_logs, eval_mean = _collect_logs_uniform(eval_subs, extractor, rng)
    print(f"Logged under uniform random: train mean reward={train_mean:.4f}, eval mean reward={eval_mean:.4f}")

    # For Section 1 we use a LinUCB log (so propensities vary, making variance
    # comparisons interesting).
    print()
    print(f"Collecting LinUCB log for Section 1 ({args.n} rounds, eps=0.1) ...")
    linucb_logs, linucb_mean, trained = _collect_logs_linucb(subs, extractor, rng)
    print(f"  empirical mean reward (logging policy): {linucb_mean:.4f}")
    target_linucb = linucb_propensity_policy(trained)

    sec1 = section_1_dr_variance(linucb_logs, target_linucb, args.bootstrap_iter)
    sec2 = section_2_sequential(rng)
    sec3 = section_3_poem(train_logs, eval_logs, args.bootstrap_iter)

    out = {
        "n": args.n,
        "seed": args.seed,
        "linucb_mean_reward": linucb_mean,
        "section_1_dr_variance": sec1,
        "section_2_sequential": sec2,
        "section_3_poem": sec3,
    }
    (LOG_DIR / "advanced_run.json").write_text(json.dumps(out, indent=2))
    print(f"\nFull report -> {LOG_DIR / 'advanced_run.json'}")


if __name__ == "__main__":
    main()
