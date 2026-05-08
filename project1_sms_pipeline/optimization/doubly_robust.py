"""Doubly-robust off-policy estimator with k-fold cross-fitting.

The DR estimator (Dudík, Langford, Li 2011) combines a direct-method reward
model with an IPS correction:

    V_DR(pi) = (1/N) Σ_i [
        Σ_a pi(a | x_i) Q_hat(x_i, a)               # direct method (DM) term
        + (pi(a_i | x_i) / mu(a_i | x_i)) * (r_i - Q_hat(x_i, a_i))   # IPS correction
    ]

Doubly robust: unbiased if EITHER the propensities OR the reward model is
correct. Variance is typically much lower than IPS or SNIPS because the
correction term has zero mean when the reward model is well-specified.

Cross-fitting (Chernozhukov et al. 2018) avoids the bias from training Q_hat
on the same data we use for evaluation. Standard practice for DR in production:
split into K folds, train Q_hat on K-1 folds, evaluate the DR sum on the held-
out fold, average across folds.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from .off_policy_eval import LogEntry, OPEResult
from .reward_model import RidgeRewardModel, fit_reward_model


def _split_folds(n: int, k: int, seed: int) -> list[np.ndarray]:
    """Return a list of K index arrays, each disjoint, covering 0..n-1."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    return np.array_split(perm, k)


def _dr_term_for_log(
    e: LogEntry,
    target_policy: Callable[[np.ndarray], dict[str, float]],
    q_hat: RidgeRewardModel,
    clip: float | None,
) -> float:
    target_props = target_policy(e.context)
    # Direct method term: Σ_a pi(a|x) Q_hat(x, a)
    dm_term = sum(
        target_props.get(a, 0.0) * q_hat.predict(e.context, a)
        for a in q_hat.arm_ids
    )
    # IPS correction: (pi(a_i|x_i) / mu(a_i|x_i)) * (r - Q_hat(x, a_i))
    target_p = target_props.get(e.arm_id, 0.0)
    if e.propensity > 0 and target_p > 0:
        weight = target_p / e.propensity
        if clip is not None:
            weight = min(weight, clip)
        ips_correction = weight * (e.reward - q_hat.predict(e.context, e.arm_id))
    else:
        ips_correction = 0.0
    return dm_term + ips_correction


def doubly_robust_estimate(
    logs: Sequence[LogEntry],
    target_policy: Callable[[np.ndarray], dict[str, float]],
    n_folds: int = 5,
    lam: float = 1.0,
    clip: float | None = None,
    seed: int = 0,
) -> float:
    """Cross-fitted doubly-robust point estimate of V_pi."""
    if not logs:
        return 0.0
    n = len(logs)
    folds = _split_folds(n, n_folds, seed=seed)
    arm_ids = sorted({e.arm_id for e in logs})

    contexts = np.array([e.context for e in logs])
    actions = [e.arm_id for e in logs]
    rewards = np.array([e.reward for e in logs])

    fold_terms = []
    for k in range(n_folds):
        eval_idx = folds[k]
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != k])
        # Fit Q_hat on training fold only.
        q_hat = fit_reward_model(
            contexts=contexts[train_idx],
            arm_ids=[actions[i] for i in train_idx],
            rewards=rewards[train_idx],
            all_arm_ids=arm_ids,
            lam=lam,
        )
        # Compute DR term on each held-out log row.
        for i in eval_idx:
            fold_terms.append(_dr_term_for_log(logs[i], target_policy, q_hat, clip))

    return float(np.mean(fold_terms))


def bootstrap_dr(
    logs: Sequence[LogEntry],
    target_policy: Callable[[np.ndarray], dict[str, float]],
    n_iter: int = 500,
    alpha: float = 0.05,
    n_folds: int = 5,
    lam: float = 1.0,
    clip: float | None = None,
    seed: int = 0,
) -> OPEResult:
    """Percentile-bootstrap CI on the cross-fitted DR estimator.

    Bootstrap is over the *log entries*, with cross-fitting re-done per resample.
    This is the honest version; cheaper variants resample DR terms instead of
    refitting, but those underestimate variance.
    """
    rng = random.Random(seed)
    n = len(logs)
    samples = []
    for j in range(n_iter):
        idx = [rng.randrange(n) for _ in range(n)]
        boot = [logs[i] for i in idx]
        samples.append(
            doubly_robust_estimate(
                boot, target_policy, n_folds=n_folds, lam=lam, clip=clip, seed=seed + 1 + j
            )
        )
    samples.sort()
    lo = samples[int(n_iter * (alpha / 2))]
    hi = samples[int(n_iter * (1 - alpha / 2))]
    point = doubly_robust_estimate(
        logs, target_policy, n_folds=n_folds, lam=lam, clip=clip, seed=seed
    )
    return OPEResult(estimator="dr_xfit", estimate=point, ci_low=lo, ci_high=hi, n_logs=n)
