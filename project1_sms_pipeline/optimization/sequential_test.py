"""Always-valid (peeking-safe) confidence sequences for Bernoulli A/B tests.

Standard fixed-horizon t-tests fail when a marketer peeks at p-values during a
running experiment — type-I error inflates with every peek. A confidence
sequence (CS) is a sequence of intervals (CI_t)_{t>=1} such that

    P( for all t >= 1, mu ∈ CI_t ) >= 1 - alpha.

Equivalently, you can stop the experiment at ANY data-dependent stopping rule
and the coverage guarantee still holds. Peek as much as you want.

Implementation. We use a Hoeffding-type uniform CS for [0, 1]-bounded outcomes
(Howard, Ramdas, McAuliffe, Sekhon 2021):

    half_width(n) = sqrt( (log(log(2 max(n, 2))) + log(2 / alpha)) / (2 n) )

This is a valid confidence sequence and is the simplest such bound that does
not require sample-variance estimates. Tighter empirical-Bernstein versions
exist; this one is preferred here for clarity and zero-tuning.

For two-arm A/B tests, we run two independent CSs (one per arm) at level
alpha/2 and intersect: the resulting CS for the difference Δ = p_A - p_B
satisfies the union-bound guarantee at level alpha.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class ConfidenceSequence:
    n: int
    mean: float
    half_width: float

    @property
    def lo(self) -> float:
        return max(0.0, self.mean - self.half_width)

    @property
    def hi(self) -> float:
        return min(1.0, self.mean + self.half_width)


def hoeffding_cs(n_samples: int, sample_mean: float, alpha: float = 0.05) -> ConfidenceSequence:
    """Always-valid CS for a [0,1]-bounded mean."""
    if n_samples <= 0:
        return ConfidenceSequence(n=0, mean=0.0, half_width=1.0)
    n = max(n_samples, 1)
    inner = math.log(math.log(max(2.0 * n, 2.0))) + math.log(2.0 / alpha)
    half = math.sqrt(inner / (2.0 * n))
    return ConfidenceSequence(n=n, mean=sample_mean, half_width=half)


@dataclass
class ABTestState:
    n_a: int = 0
    sum_a: int = 0
    n_b: int = 0
    sum_b: int = 0

    def observe(self, arm: str, reward: float) -> None:
        if arm == "A":
            self.n_a += 1
            self.sum_a += int(reward)
        elif arm == "B":
            self.n_b += 1
            self.sum_b += int(reward)
        else:
            raise ValueError(f"arm must be 'A' or 'B', got {arm!r}")

    def diff_cs(self, alpha: float = 0.05) -> tuple[float, float, float]:
        """Always-valid CS for Δ = p_A - p_B at level alpha (union bound)."""
        ca = hoeffding_cs(self.n_a, self.sum_a / max(1, self.n_a), alpha=alpha / 2)
        cb = hoeffding_cs(self.n_b, self.sum_b / max(1, self.n_b), alpha=alpha / 2)
        center = ca.mean - cb.mean
        half = ca.half_width + cb.half_width
        return center, center - half, center + half

    def is_significant(self, alpha: float = 0.05) -> str | None:
        """If the CS for Δ excludes 0, return the winner. Otherwise None."""
        center, lo, hi = self.diff_cs(alpha=alpha)
        if lo > 0:
            return "A"
        if hi < 0:
            return "B"
        return None


@dataclass
class StopDecision:
    stopped_at: int
    decision: str          # "A_better" | "B_better" | "no_decision"
    final_diff: float
    final_ci: tuple[float, float]


def run_sequential_ab(
    arm_seq: list[str],            # length T, "A" or "B" per step
    reward_seq: list[float],       # length T
    alpha: float = 0.05,
    min_per_arm: int = 30,
) -> StopDecision:
    """Run a peeking-safe A/B test, stop on first significant CS.

    `min_per_arm` guards against very early stops on a tiny sample.
    """
    state = ABTestState()
    for t, (arm, r) in enumerate(zip(arm_seq, reward_seq)):
        state.observe(arm, r)
        if state.n_a >= min_per_arm and state.n_b >= min_per_arm:
            sig = state.is_significant(alpha=alpha)
            if sig is not None:
                center, lo, hi = state.diff_cs(alpha=alpha)
                return StopDecision(
                    stopped_at=t + 1,
                    decision=("A_better" if sig == "A" else "B_better"),
                    final_diff=center,
                    final_ci=(lo, hi),
                )
    center, lo, hi = state.diff_cs(alpha=alpha)
    return StopDecision(
        stopped_at=len(arm_seq),
        decision="no_decision",
        final_diff=center,
        final_ci=(lo, hi),
    )


@dataclass
class TypeIErrorResult:
    n_trials: int
    n_false_rejections: int
    empirical_alpha: float
    target_alpha: float


def simulate_type_i_error(
    n_trials: int,
    horizon: int,
    p_shared: float,
    alpha: float = 0.05,
    min_per_arm: int = 30,
    seed: int = 0,
) -> TypeIErrorResult:
    """Stress test: under H0 (p_A == p_B == p_shared), how often does the CS
    falsely reject? Should be <= alpha regardless of how many times we peek.

    This is the property that fixed-horizon t-tests with peeking violate.
    """
    import random
    rng = random.Random(seed)
    n_false = 0
    for _ in range(n_trials):
        arms = [rng.choice(["A", "B"]) for _ in range(horizon)]
        rewards = [1.0 if rng.random() < p_shared else 0.0 for _ in range(horizon)]
        decision = run_sequential_ab(arms, rewards, alpha=alpha, min_per_arm=min_per_arm)
        if decision.decision != "no_decision":
            n_false += 1
    return TypeIErrorResult(
        n_trials=n_trials,
        n_false_rejections=n_false,
        empirical_alpha=n_false / n_trials,
        target_alpha=alpha,
    )
