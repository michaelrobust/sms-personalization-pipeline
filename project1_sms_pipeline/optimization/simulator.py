"""Simulated CTR for bandit demos.

Real production would observe actual clicks. For demo runs we synthesize
binary rewards with a `PersonaCTRModel` so the bandit has something to learn.
The model gives each cohort a "preferred tone" and a "preferred CTA pattern";
variants matching either get a CTR boost on top of the cohort's base rate.
"""
from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .bandit import Arm, BanditState, BanditStore, BetaBernoulliBandit


# Hidden ground-truth preferences per cohort. The bandit must discover these.
COHORT_TRUE_PREFERENCES: dict[str, dict[str, Any]] = {
    "winback_dormant":     {"base_ctr": 0.04, "preferred_tone": "empathetic",
                            "preferred_cta_keywords": ["come back", "miss you", "% off"]},
    "high_intent_browser": {"base_ctr": 0.10, "preferred_tone": "urgent",
                            "preferred_cta_keywords": ["last", "left", "now", "stock"]},
    "vip_loyalist":        {"base_ctr": 0.12, "preferred_tone": "premium",
                            "preferred_cta_keywords": ["early access", "members", "exclusive"]},
    "price_sensitive":     {"base_ctr": 0.08, "preferred_tone": "direct",
                            "preferred_cta_keywords": ["%", "save", "off", "$"]},
    "new_subscriber":      {"base_ctr": 0.06, "preferred_tone": "warm",
                            "preferred_cta_keywords": ["welcome", "first", "tap"]},
    "post_purchase":       {"base_ctr": 0.07, "preferred_tone": "warm",
                            "preferred_cta_keywords": ["matches", "goes with", "complete"]},
}


@dataclass
class PersonaCTRModel:
    """CTR = base + tone_bonus + cta_bonus (clipped to [0, 1])."""
    tone_bonus: float = 0.06
    cta_bonus: float = 0.05

    def true_ctr(self, cohort_id: str, variant_tone: str, variant_cta: str) -> float:
        prefs = COHORT_TRUE_PREFERENCES.get(cohort_id)
        if prefs is None:
            return 0.05  # default cohort
        ctr = prefs["base_ctr"]
        if variant_tone == prefs["preferred_tone"]:
            ctr += self.tone_bonus
        cta_lower = variant_cta.lower()
        if any(kw in cta_lower for kw in prefs["preferred_cta_keywords"]):
            ctr += self.cta_bonus
        return max(0.0, min(1.0, ctr))


def simulate_send(
    rng: random.Random,
    model: PersonaCTRModel,
    cohort_id: str,
    variant_tone: str,
    variant_cta: str,
) -> int:
    """Sample 1 (clicked) or 0 (not) from Bernoulli(true_ctr)."""
    ctr = model.true_ctr(cohort_id, variant_tone, variant_cta)
    return 1 if rng.random() < ctr else 0


@dataclass
class StepRecord:
    t: int
    chosen_arm_id: str
    reward: int
    cumulative_reward: int
    posterior_means: dict[str, float]


@dataclass
class BanditRunResult:
    cohort_id: str
    n_rounds: int
    arms: list[dict[str, Any]]                        # final arm states
    best_arm_id: str
    cumulative_reward: int
    cumulative_regret: float
    steps: list[StepRecord] = field(default_factory=list)


def run_bandit_loop(
    cohort_id: str,
    variants: list[dict[str, Any]],
    n_rounds: int = 500,
    store: BanditStore | None = None,
    seed: int = 0,
    record_every: int = 10,
) -> BanditRunResult:
    """Run T rounds of Thompson sampling against the simulated CTR model.

    `variants` is a list of dicts with keys: variant_id, tone, cta_action.

    Returns the final arm states + a sparse per-step trace for plotting.
    """
    rng = random.Random(seed)
    model = PersonaCTRModel()
    bandit = BetaBernoulliBandit(rng=rng)

    arms = [
        Arm(arm_id=v["variant_id"], label=f"{v['tone']} | {v['cta_action']}")
        for v in variants
    ]
    state = BanditState(cohort_id=cohort_id, arms=arms)

    # Best arm by true CTR (used to compute regret).
    true_ctrs = {
        v["variant_id"]: model.true_ctr(cohort_id, v["tone"], v["cta_action"])
        for v in variants
    }
    best_true_ctr = max(true_ctrs.values())
    variants_by_id = {v["variant_id"]: v for v in variants}

    cumulative_reward = 0
    cumulative_regret = 0.0
    steps: list[StepRecord] = []

    for t in range(1, n_rounds + 1):
        chosen = bandit.pick_arm(state)
        v = variants_by_id[chosen.arm_id]
        reward = simulate_send(rng, model, cohort_id, v["tone"], v["cta_action"])
        bandit.update(state, chosen.arm_id, reward)

        cumulative_reward += reward
        cumulative_regret += best_true_ctr - true_ctrs[chosen.arm_id]

        if t % record_every == 0 or t == n_rounds:
            steps.append(
                StepRecord(
                    t=t,
                    chosen_arm_id=chosen.arm_id,
                    reward=reward,
                    cumulative_reward=cumulative_reward,
                    posterior_means={a.arm_id: round(a.posterior_mean, 4) for a in state.arms},
                )
            )

    if store is not None:
        store.upsert(state)

    return BanditRunResult(
        cohort_id=cohort_id,
        n_rounds=n_rounds,
        arms=[asdict(a) for a in state.arms],
        best_arm_id=bandit.best_arm(state).arm_id,
        cumulative_reward=cumulative_reward,
        cumulative_regret=round(cumulative_regret, 4),
        steps=steps,
    )


def format_run_summary(result: BanditRunResult, true_ctrs: dict[str, float] | None = None) -> str:
    lines = [
        f"Bandit run: cohort={result.cohort_id}  rounds={result.n_rounds}",
        f"  cumulative reward     : {result.cumulative_reward}  ({result.cumulative_reward/result.n_rounds:.1%})",
        f"  cumulative regret     : {result.cumulative_regret:.2f}",
        f"  best arm (posterior)  : {result.best_arm_id}",
        "",
        f"  {'arm':24s}  {'pulls':>6s}  {'emp_ctr':>8s}  {'post_mean':>9s}  {'true_ctr':>8s}",
    ]
    for arm in result.arms:
        true_str = (
            f"{true_ctrs[arm['arm_id']]:.3f}"
            if true_ctrs and arm["arm_id"] in true_ctrs
            else "  -  "
        )
        emp = arm["rewards"] / arm["pulls"] if arm["pulls"] else 0.0
        post = arm["alpha"] / (arm["alpha"] + arm["beta"])
        lines.append(
            f"  {arm['label'][:24]:24s}  {arm['pulls']:>6d}  {emp:>8.3f}  {post:>9.3f}  {true_str:>8s}"
        )
    return "\n".join(lines)
