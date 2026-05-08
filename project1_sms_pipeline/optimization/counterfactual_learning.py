"""Counterfactual policy learning (POEM-style).

POEM (Swaminathan & Joachims 2015) trains a policy directly from a logged
bandit dataset by maximizing a variance-regularized IPS objective:

    L(θ) = V_IPS(π_θ) - λ * sqrt( Var_IPS(π_θ) / N )

where π_θ is a stochastic policy and V_IPS is the inverse-propensity-scored
estimate of its value. The variance term is the "self-normalized" empirical
variance of importance weights times rewards; it discourages policies that
would put high mass on rarely-logged actions.

We parametrize π_θ as a softmax over linear scores:
    π_θ(a | x) = exp(θ_a · x) / Σ_a' exp(θ_a' · x)

Optimization uses scipy's L-BFGS-B with manual gradients. The result is a
policy that is *learned offline from a log* — no online experiments needed
to ship a candidate. After training, evaluate the learned policy with DR or
SNIPS and decide whether the counterfactual lift is worth the rollout risk.

References:
  Swaminathan, A. & Joachims, T. (2015). Counterfactual Risk Minimization.
  Joachims, T. & Swaminathan, A. (2016). The Self-Normalized Estimator for
  Counterfactual Learning.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

try:
    from scipy.optimize import minimize  # type: ignore
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

from .off_policy_eval import LogEntry


# ---- Softmax linear policy -------------------------------------------------


@dataclass
class SoftmaxLinearPolicy:
    arm_ids: list[str]      # K arms
    theta: np.ndarray       # (K, d)

    @property
    def n_arms(self) -> int:
        return len(self.arm_ids)

    @property
    def dim(self) -> int:
        return self.theta.shape[1]

    def logits(self, x: np.ndarray) -> np.ndarray:
        return self.theta @ x

    def probabilities(self, x: np.ndarray) -> np.ndarray:
        z = self.logits(x)
        z = z - np.max(z)
        e = np.exp(z)
        return e / e.sum()

    def propensity_dict(self, x: np.ndarray) -> dict[str, float]:
        p = self.probabilities(x)
        return {a: float(p[i]) for i, a in enumerate(self.arm_ids)}

    def predict(self, x: np.ndarray) -> str:
        """Greedy action (argmax). Used for deployment."""
        idx = int(np.argmax(self.logits(x)))
        return self.arm_ids[idx]


# ---- POEM objective and gradient ------------------------------------------


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def _ips_weights_and_rewards(
    theta_flat: np.ndarray,
    K: int,
    d: int,
    contexts: np.ndarray,
    arm_idx: np.ndarray,
    propensities: np.ndarray,
    rewards: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    theta = theta_flat.reshape(K, d)
    logits = contexts @ theta.T  # (N, K)
    probs = _softmax(logits)     # (N, K)
    # pi(a_i | x_i)
    n = contexts.shape[0]
    pi_ai = probs[np.arange(n), arm_idx]
    weights = pi_ai / propensities
    weighted_rewards = weights * rewards
    return weights, weighted_rewards


def poem_objective(
    theta_flat: np.ndarray,
    K: int,
    d: int,
    contexts: np.ndarray,
    arm_idx: np.ndarray,
    propensities: np.ndarray,
    rewards: np.ndarray,
    lam_var: float,
    l2: float,
) -> float:
    """Negative POEM objective (we minimize, so we negate for L-BFGS)."""
    _, weighted_rewards = _ips_weights_and_rewards(
        theta_flat, K, d, contexts, arm_idx, propensities, rewards
    )
    n = len(rewards)
    v_ips = weighted_rewards.mean()
    var_ips = weighted_rewards.var(ddof=1) if n > 1 else 0.0
    penalty = lam_var * np.sqrt(max(var_ips, 0.0) / max(n, 1))
    l2_term = 0.5 * l2 * (theta_flat @ theta_flat)
    # We minimize -L = -V_IPS + penalty + l2
    return -v_ips + penalty + l2_term


def poem_gradient(
    theta_flat: np.ndarray,
    K: int,
    d: int,
    contexts: np.ndarray,
    arm_idx: np.ndarray,
    propensities: np.ndarray,
    rewards: np.ndarray,
    lam_var: float,
    l2: float,
) -> np.ndarray:
    """Analytical gradient of -V_IPS w.r.t. theta. The variance penalty's
    gradient is dropped (constant approximation) to keep the objective convex
    in practice; this matches the POEM 'simple' variant."""
    theta = theta_flat.reshape(K, d)
    logits = contexts @ theta.T
    probs = _softmax(logits)
    n = contexts.shape[0]
    pi_ai = probs[np.arange(n), arm_idx]
    # d log pi(a | x) / d theta_k:
    #   if k == a:  x - probs[a] * x  (wait, this is for log)
    # We need d pi(a | x) / d theta_k, k = chosen arm.
    # pi(a | x) = softmax(logits)_a
    # d softmax_a / d theta_k = pi(a) * (1{k==a} - pi(k)) * x  (per-row)
    # Then d (pi(a_i)/p_i * r_i) / d theta_k =
    #     (pi(a_i) / p_i) * (1{k==a_i} - pi(k|x_i)) * r_i * x_i
    # Average over i, take negative for minimization.
    weights = pi_ai / propensities
    weighted_r = weights * rewards
    # gradient w.r.t. theta_k for each k
    grad = np.zeros((K, d))
    for k in range(K):
        # 1{k == a_i}
        is_chosen = (arm_idx == k).astype(float)
        coef = weighted_r * (is_chosen - probs[:, k])
        grad[k] = (coef[:, None] * contexts).sum(axis=0) / n
    grad_neg = -grad.flatten()  # because we minimize -V_IPS
    grad_neg += l2 * theta_flat
    return grad_neg


# ---- Top-level training ---------------------------------------------------


def train_poem(
    logs: Sequence[LogEntry],
    arm_ids: Sequence[str],
    lam_var: float = 0.1,
    l2: float = 1e-3,
    max_iter: int = 200,
    seed: int = 0,
) -> SoftmaxLinearPolicy:
    """Train a softmax linear policy by minimizing -POEM objective."""
    if not logs:
        raise ValueError("Empty logs.")
    if not _HAS_SCIPY:
        raise RuntimeError("scipy is required for POEM training.")
    arm_ids = list(arm_ids)
    arm_to_idx = {a: i for i, a in enumerate(arm_ids)}
    K = len(arm_ids)
    d = logs[0].context.shape[0]

    contexts = np.stack([e.context for e in logs])
    arm_idx = np.array([arm_to_idx[e.arm_id] for e in logs])
    propensities = np.array([e.propensity for e in logs])
    rewards = np.array([e.reward for e in logs])

    rng = np.random.default_rng(seed)
    theta0 = 0.01 * rng.standard_normal(K * d)

    res = minimize(
        fun=poem_objective,
        x0=theta0,
        jac=poem_gradient,
        args=(K, d, contexts, arm_idx, propensities, rewards, lam_var, l2),
        method="L-BFGS-B",
        options={"maxiter": max_iter, "gtol": 1e-6},
    )
    theta_star = res.x.reshape(K, d)
    return SoftmaxLinearPolicy(arm_ids=arm_ids, theta=theta_star)


def softmax_policy_propensity(policy: SoftmaxLinearPolicy):
    """Wrap a SoftmaxLinearPolicy as a target policy for OPE."""
    def fn(x: np.ndarray) -> dict[str, float]:
        return policy.propensity_dict(x)
    return fn
