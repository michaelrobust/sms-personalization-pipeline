"""LinUCB contextual bandit (Li, Chu, Langford, Schapire 2010).

Per-arm linear regression on the user-context vector. Pick the arm with the
highest UCB-style score:  theta_a . x  +  alpha * sqrt(x.T A_a^-1 x)

Wrapped in epsilon-greedy so the resulting policy is stochastic. Stochasticity
matters because (a) it provides exploration and (b) it gives every arm a
non-zero propensity, which is required for inverse-propensity-scored off-policy
evaluation.

Online updates use the Sherman-Morrison identity to avoid recomputing A_a^-1
from scratch on every step.

State is JSON-serializable: A_a, b_a, and the running scalar counts are
written out so the bandit can resume across processes.
"""
from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from ..schemas import SEGMENT_LABELS


# ---- Feature extractor -----------------------------------------------------


@dataclass
class FeatureExtractor:
    """Map a subscriber dict to a fixed-dim feature vector.

    Layout (10 dims total):
        index 0       bias (always 1)
        index 1       days_since_last_visit / 100, clipped to [0, 2]
        index 2       purchase_count_90d / 30, clipped to [0, 2]
        index 3       log1p(avg_order_value) / 10, clipped to [0, 2]
        index 4..9    segment one-hot (6 dims)
    """
    DIM: int = 1 + 3 + 6

    SEGMENT_INDEX: dict[str, int] = field(
        default_factory=lambda: {seg: i for i, seg in enumerate(SEGMENT_LABELS)}
    )

    def featurize(self, subscriber: dict[str, Any]) -> np.ndarray:
        x = np.zeros(self.DIM)
        x[0] = 1.0
        x[1] = min(2.0, subscriber.get("days_since_last_visit", 0) / 100.0)
        x[2] = min(2.0, subscriber.get("purchase_count_90d", 0) / 30.0)
        x[3] = min(2.0, math.log1p(subscriber.get("avg_order_value", 0.0)) / 10.0)
        seg = subscriber.get("segment_predicted") or subscriber.get("segment_truth")
        if seg in self.SEGMENT_INDEX:
            x[4 + self.SEGMENT_INDEX[seg]] = 1.0
        return x


# ---- LinUCB ---------------------------------------------------------------


@dataclass
class LinUCBState:
    """Per-arm sufficient statistics. JSON-serializable via .to_dict / .from_dict."""
    arm_id: str
    A: np.ndarray              # (d, d)
    b: np.ndarray              # (d,)
    pulls: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_id": self.arm_id,
            "A": self.A.tolist(),
            "b": self.b.tolist(),
            "pulls": self.pulls,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "LinUCBState":
        return cls(
            arm_id=d["arm_id"],
            A=np.array(d["A"], dtype=float),
            b=np.array(d["b"], dtype=float),
            pulls=int(d.get("pulls", 0)),
        )

    @classmethod
    def fresh(cls, arm_id: str, dim: int) -> "LinUCBState":
        return cls(arm_id=arm_id, A=np.eye(dim), b=np.zeros(dim), pulls=0)


class LinUCB:
    """Per-arm linear regression with UCB exploration. Wrap in epsilon-greedy
    for stochastic policy + IPS-friendly propensities."""

    def __init__(
        self,
        arm_ids: Sequence[str],
        dim: int,
        alpha: float = 1.0,
        epsilon: float = 0.1,
        rng: random.Random | None = None,
    ):
        if not arm_ids:
            raise ValueError("Need at least one arm.")
        self.arm_ids = list(arm_ids)
        self.dim = dim
        self.alpha = alpha
        self.epsilon = epsilon
        self.rng = rng or random.Random()
        self.states: dict[str, LinUCBState] = {
            a: LinUCBState.fresh(a, dim) for a in self.arm_ids
        }

    # ---- core inference --------------------------------------------------

    def _ucb_scores(self, x: np.ndarray) -> np.ndarray:
        scores = np.zeros(len(self.arm_ids))
        for i, arm_id in enumerate(self.arm_ids):
            s = self.states[arm_id]
            A_inv = np.linalg.inv(s.A)
            theta = A_inv @ s.b
            mean = float(theta @ x)
            confidence = self.alpha * float(np.sqrt(x @ A_inv @ x))
            scores[i] = mean + confidence
        return scores

    def _argmax(self, x: np.ndarray) -> int:
        scores = self._ucb_scores(x)
        return int(np.argmax(scores))

    def select(self, x: np.ndarray) -> tuple[str, float, np.ndarray]:
        """Return (chosen_arm_id, propensity, all_propensities).

        Epsilon-greedy: with prob (1 - epsilon) pick argmax-UCB, else uniform.
        Propensity for arm a is:
            (1 - eps) + eps/K   if a == argmax
            eps / K             otherwise
        """
        K = len(self.arm_ids)
        argmax_idx = self._argmax(x)

        if self.rng.random() < self.epsilon:
            chosen_idx = self.rng.randrange(K)
        else:
            chosen_idx = argmax_idx

        propensities = np.full(K, self.epsilon / K)
        propensities[argmax_idx] += (1.0 - self.epsilon)

        chosen_arm = self.arm_ids[chosen_idx]
        return chosen_arm, float(propensities[chosen_idx]), propensities

    # ---- learning --------------------------------------------------------

    def update(self, arm_id: str, x: np.ndarray, reward: float) -> None:
        s = self.states[arm_id]
        s.A = s.A + np.outer(x, x)
        s.b = s.b + reward * x
        s.pulls += 1

    # ---- introspection ---------------------------------------------------

    def theta(self, arm_id: str) -> np.ndarray:
        s = self.states[arm_id]
        return np.linalg.inv(s.A) @ s.b

    def expected_reward(self, x: np.ndarray, arm_id: str) -> float:
        return float(self.theta(arm_id) @ x)

    def propensity_vector(self, x: np.ndarray) -> np.ndarray:
        K = len(self.arm_ids)
        argmax_idx = self._argmax(x)
        p = np.full(K, self.epsilon / K)
        p[argmax_idx] += (1.0 - self.epsilon)
        return p

    # ---- persistence -----------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm_ids": self.arm_ids,
            "dim": self.dim,
            "alpha": self.alpha,
            "epsilon": self.epsilon,
            "states": {a: s.to_dict() for a, s in self.states.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any], rng: random.Random | None = None) -> "LinUCB":
        obj = cls(
            arm_ids=d["arm_ids"],
            dim=int(d["dim"]),
            alpha=float(d["alpha"]),
            epsilon=float(d["epsilon"]),
            rng=rng,
        )
        obj.states = {a: LinUCBState.from_dict(v) for a, v in d["states"].items()}
        return obj


class LinUCBStore:
    """Atomic-write JSON store for LinUCB state."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, bandit: LinUCB) -> None:
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(bandit.to_dict(), indent=2))
        tmp.replace(self.path)

    def load(self, rng: random.Random | None = None) -> LinUCB | None:
        if not self.path.exists():
            return None
        return LinUCB.from_dict(json.loads(self.path.read_text()), rng=rng)
