"""Off-policy evaluation: IPS, self-normalized IPS (SNIPS), bootstrap CIs.

Setup. We have a logged dataset of (context, action, propensity, reward) tuples
collected by some logging policy mu. We want to estimate the expected reward of
a (potentially different) target policy pi WITHOUT running pi on real users.

Estimators (Horvitz-Thompson family):

  IPS:    V_pi = (1/N) * sum_i  pi(a_i | x_i) / mu(a_i | x_i)  *  r_i
  SNIPS:  V_pi = sum_i  w_i  *  r_i  /  sum_i  w_i
                where w_i = pi(a_i | x_i) / mu(a_i | x_i)

IPS is unbiased but high-variance. SNIPS trades a small bias for substantially
lower variance and is the more practical estimator most of the time.

Both require mu(a_i | x_i) > 0 for every logged action — i.e. the logging
policy must explore every arm with non-zero probability for every context.
The epsilon-greedy LinUCB in `contextual_bandit.py` satisfies this.

References:
  Horvitz & Thompson (1952), JASA.
  Dudík, Langford, Li (2011), "Doubly Robust Policy Evaluation".
  Swaminathan & Joachims (2015), "The Self-Normalized Estimator for
  Counterfactual Learning".
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np


@dataclass
class LogEntry:
    """A single logged decision."""
    context: np.ndarray            # feature vector x
    arm_id: str                    # action a taken by the logging policy
    propensity: float              # mu(a | x), must be > 0
    reward: float                  # observed reward r


# ---- Target policies ------------------------------------------------------


def uniform_random_policy(arm_ids: Sequence[str]) -> Callable[[np.ndarray], dict[str, float]]:
    K = len(arm_ids)
    p = 1.0 / K

    def policy(x: np.ndarray) -> dict[str, float]:
        return {a: p for a in arm_ids}

    return policy


def deterministic_policy(arm_ids: Sequence[str], pick_fn: Callable[[np.ndarray], str]) -> Callable[[np.ndarray], dict[str, float]]:
    """Wrap a deterministic policy as propensity dict."""
    def policy(x: np.ndarray) -> dict[str, float]:
        chosen = pick_fn(x)
        return {a: (1.0 if a == chosen else 0.0) for a in arm_ids}

    return policy


def linucb_propensity_policy(bandit) -> Callable[[np.ndarray], dict[str, float]]:
    """Use a trained LinUCB's epsilon-greedy propensities as the target policy."""
    def policy(x: np.ndarray) -> dict[str, float]:
        p = bandit.propensity_vector(x)
        return {arm: float(p[i]) for i, arm in enumerate(bandit.arm_ids)}

    return policy


# ---- Estimators -----------------------------------------------------------


def _importance_weight(target_prop: float, logged_prop: float, clip: float | None) -> float:
    if logged_prop <= 0:
        # Logged policy never picks this arm under this context: undefined.
        return 0.0
    w = target_prop / logged_prop
    if clip is not None and w > clip:
        return clip
    return w


def ips_estimate(
    logs: Sequence[LogEntry],
    target_policy: Callable[[np.ndarray], dict[str, float]],
    clip: float | None = None,
) -> float:
    """Standard IPS (Horvitz-Thompson) estimate of V_pi."""
    if not logs:
        return 0.0
    total = 0.0
    for e in logs:
        target_prop = target_policy(e.context).get(e.arm_id, 0.0)
        w = _importance_weight(target_prop, e.propensity, clip)
        total += w * e.reward
    return total / len(logs)


def snips_estimate(
    logs: Sequence[LogEntry],
    target_policy: Callable[[np.ndarray], dict[str, float]],
    clip: float | None = None,
) -> float:
    """Self-normalized IPS estimate. Lower variance, slightly biased."""
    if not logs:
        return 0.0
    weighted = 0.0
    sum_w = 0.0
    for e in logs:
        target_prop = target_policy(e.context).get(e.arm_id, 0.0)
        w = _importance_weight(target_prop, e.propensity, clip)
        weighted += w * e.reward
        sum_w += w
    if sum_w == 0:
        return 0.0
    return weighted / sum_w


# ---- Bootstrap CIs --------------------------------------------------------


@dataclass
class OPEResult:
    estimator: str
    estimate: float
    ci_low: float
    ci_high: float
    n_logs: int


def bootstrap_ope(
    logs: Sequence[LogEntry],
    target_policy: Callable[[np.ndarray], dict[str, float]],
    estimator: str = "snips",
    n_iter: int = 500,
    alpha: float = 0.05,
    clip: float | None = None,
    seed: int = 0,
) -> OPEResult:
    """Percentile bootstrap CI on the chosen estimator."""
    if estimator == "ips":
        fn = ips_estimate
    elif estimator == "snips":
        fn = snips_estimate
    else:
        raise ValueError(f"unknown estimator {estimator!r}")

    rng = random.Random(seed)
    n = len(logs)
    samples = []
    for _ in range(n_iter):
        boot = [logs[rng.randrange(n)] for _ in range(n)]
        samples.append(fn(boot, target_policy, clip=clip))
    samples.sort()
    lo = samples[int(n_iter * (alpha / 2))]
    hi = samples[int(n_iter * (1 - alpha / 2))]
    point = fn(logs, target_policy, clip=clip)
    return OPEResult(
        estimator=estimator,
        estimate=point,
        ci_low=lo,
        ci_high=hi,
        n_logs=n,
    )


def effective_sample_size(
    logs: Sequence[LogEntry],
    target_policy: Callable[[np.ndarray], dict[str, float]],
    clip: float | None = None,
) -> float:
    """ESS = (sum w)^2 / sum w^2. Sanity check: if ESS is very small relative
    to len(logs), the IPS estimate is dominated by a few high-weight samples
    and you should treat the CI with suspicion."""
    if not logs:
        return 0.0
    weights = []
    for e in logs:
        target_prop = target_policy(e.context).get(e.arm_id, 0.0)
        weights.append(_importance_weight(target_prop, e.propensity, clip))
    s1 = sum(weights)
    s2 = sum(w * w for w in weights)
    return (s1 * s1) / s2 if s2 > 0 else 0.0
