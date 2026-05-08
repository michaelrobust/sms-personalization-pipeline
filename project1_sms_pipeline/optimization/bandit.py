"""Beta-Bernoulli multi-armed bandit with Thompson sampling.

One bandit per cohort. Arms are SMS variants. Reward is binary (clicked / didn't).
The posterior on arm i is Beta(alpha_i, beta_i); after observing reward r in {0, 1},
update is alpha_i += r, beta_i += 1 - r.

State is JSON-serializable so it persists across pipeline runs and across
processes. The store is keyed by `cohort_id`.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class Arm:
    arm_id: str                      # variant_id, deterministic
    label: str = ""                  # human-readable, e.g. tone string
    alpha: float = 1.0               # Beta prior alpha
    beta: float = 1.0                # Beta prior beta
    pulls: int = 0
    rewards: int = 0

    @property
    def empirical_rate(self) -> float:
        return self.rewards / self.pulls if self.pulls else 0.0

    @property
    def posterior_mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)


@dataclass
class BanditState:
    cohort_id: str
    arms: list[Arm] = field(default_factory=list)
    total_pulls: int = 0


class BetaBernoulliBandit:
    """Single-cohort bandit. Stateless container; pass a BanditState in/out."""

    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random()

    def pick_arm(self, state: BanditState) -> Arm:
        """Thompson sampling: draw theta_i ~ Beta(alpha_i, beta_i); pick max."""
        if not state.arms:
            raise ValueError("Bandit has no arms.")
        best_arm = state.arms[0]
        best_theta = -1.0
        for arm in state.arms:
            theta = self.rng.betavariate(arm.alpha, arm.beta)
            if theta > best_theta:
                best_theta = theta
                best_arm = arm
        return best_arm

    def update(self, state: BanditState, arm_id: str, reward: int) -> BanditState:
        if reward not in (0, 1):
            raise ValueError(f"reward must be 0 or 1, got {reward!r}")
        for arm in state.arms:
            if arm.arm_id == arm_id:
                arm.pulls += 1
                arm.rewards += reward
                arm.alpha += reward
                arm.beta += (1 - reward)
                state.total_pulls += 1
                return state
        raise KeyError(f"Unknown arm_id {arm_id!r}")

    def best_arm(self, state: BanditState) -> Arm:
        """The arm with the highest posterior mean (used for reporting)."""
        return max(state.arms, key=lambda a: a.posterior_mean)


# ---- Persistence ----------------------------------------------------------


class BanditStore:
    """JSON-backed store keyed by cohort_id. Safe across processes (atomic write)."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load(self) -> dict[str, BanditState]:
        if not self.path.exists():
            return {}
        raw = json.loads(self.path.read_text())
        out: dict[str, BanditState] = {}
        for cohort_id, payload in raw.items():
            arms = [Arm(**a) for a in payload["arms"]]
            out[cohort_id] = BanditState(
                cohort_id=cohort_id,
                arms=arms,
                total_pulls=payload.get("total_pulls", 0),
            )
        return out

    def save(self, states: dict[str, BanditState]) -> None:
        payload = {
            cid: {
                "arms": [asdict(a) for a in s.arms],
                "total_pulls": s.total_pulls,
            }
            for cid, s in states.items()
        }
        # Atomic write: write to temp then rename.
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(self.path)

    def get_or_init(
        self,
        cohort_id: str,
        arms: list[tuple[str, str]],
    ) -> BanditState:
        """Return state for cohort_id, creating fresh Beta(1,1) arms if missing.

        `arms` is a list of (arm_id, label) tuples; only used if the cohort is new.
        """
        states = self.load()
        if cohort_id in states:
            return states[cohort_id]
        new_state = BanditState(
            cohort_id=cohort_id,
            arms=[Arm(arm_id=aid, label=lbl) for aid, lbl in arms],
        )
        states[cohort_id] = new_state
        self.save(states)
        return new_state

    def upsert(self, state: BanditState) -> None:
        states = self.load()
        states[state.cohort_id] = state
        self.save(states)
