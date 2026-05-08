"""Per-arm ridge regression reward model used as the direct method in DR estimation.

Q_hat(x, a) is fit by ridge regression on the rows of the log where the logged
action equals a. Closed-form: theta_a = (X_a^T X_a + lambda I)^-1 X_a^T r_a.

The model is small, deterministic, and cheap to k-fold cross-fit. It does not
need to be a great regressor — DR is unbiased even if Q_hat is wrong, as long
as propensities are correct (and vice versa). A reasonable Q_hat just
reduces variance.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class RidgeRewardModel:
    """One ridge regressor per arm. Predicts E[r | x, a]."""
    arm_ids: list[str]
    theta_by_arm: dict[str, np.ndarray]  # arm_id -> weight vector (d,)
    lam: float = 1.0

    def predict(self, x: np.ndarray, arm_id: str) -> float:
        return float(self.theta_by_arm[arm_id] @ x)

    def predict_all(self, x: np.ndarray) -> dict[str, float]:
        return {a: self.predict(x, a) for a in self.arm_ids}


def fit_reward_model(
    contexts: np.ndarray,            # (n, d)
    arm_ids: Sequence[str],          # length n
    rewards: np.ndarray,             # (n,)
    all_arm_ids: Sequence[str],
    lam: float = 1.0,
) -> RidgeRewardModel:
    """Closed-form per-arm ridge regression."""
    d = contexts.shape[1]
    theta_by_arm: dict[str, np.ndarray] = {}
    for arm in all_arm_ids:
        mask = np.array([a == arm for a in arm_ids], dtype=bool)
        if not mask.any():
            theta_by_arm[arm] = np.zeros(d)
            continue
        X_a = contexts[mask]
        r_a = rewards[mask]
        A = X_a.T @ X_a + lam * np.eye(d)
        b = X_a.T @ r_a
        theta_by_arm[arm] = np.linalg.solve(A, b)
    return RidgeRewardModel(
        arm_ids=list(all_arm_ids),
        theta_by_arm=theta_by_arm,
        lam=lam,
    )
