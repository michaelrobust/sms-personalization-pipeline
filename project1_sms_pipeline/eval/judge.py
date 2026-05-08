"""LLM judge + cross-rater agreement (Cohen's kappa, bootstrap CIs)."""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from typing import Any

from shared import LLMClient

from ..schemas import JUDGE_TOOL


JUDGE_SYSTEM = """You are a strict QA reviewer for SMS marketing copy. Score \
each variant against the rubric. Be conservative: a fuzzy CTA scores cta_clarity \
<= 1. would_send_unedited is true ONLY if you would ship that copy without \
changing a single character.

Per-axis scale:
  0 fails outright   1 significantly weak   2 acceptable   3 strong, no notes

When would_send_unedited is false, populate failure_categories with EVERY \
category that applies. Categories:
  tone_off          : voice doesn't match the persona's voice_pillars
  cta_unclear       : CTA missing, fuzzy, or asks for >1 action
  off_brand         : claims that need legal review, or breaks brand voice
  segment_mismatch  : could be for any cohort; not specific to this segment
  char_limit        : exceeds 160 chars
  duplicate_angle   : not meaningfully different from another variant
  other             : reasons not covered above (use sparingly)

When would_send_unedited is true, failure_categories MUST be an empty array.
Always call emit_acceptability_score exactly once."""


@dataclass
class VariantScore:
    variant_index: int
    tone_match: int
    cta_clarity: int
    segment_relevance: int
    char_limit_ok: bool
    would_send_unedited: bool
    failure_categories: list[str]
    notes: str


@dataclass
class JudgeResult:
    judge_model: str
    per_variant: list[VariantScore]
    overall_pass: bool


class Judge:
    def __init__(self, client: LLMClient):
        self.client = client

    def score(
        self,
        segment: str,
        persona: dict[str, Any],
        variants: list[dict[str, Any]],
    ) -> JudgeResult:
        user_msg = (
            f"Segment: {segment}\n"
            f"Persona frame:\n{json.dumps(persona, indent=2)}\n\n"
            f"Variants to score:\n{json.dumps(variants, indent=2)}\n\n"
            "Score every variant and emit overall_pass."
        )
        resp = self.client.call(
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": user_msg}],
            tools=[JUDGE_TOOL],
            tool_choice={"type": "tool", "name": JUDGE_TOOL["name"]},
            cache_system=True,
            temperature=0.1,
        )
        data = resp.first_tool_input()
        if not data:
            raise RuntimeError("Judge returned no tool call.")
        per_variant = [_to_variant_score(v) for v in data["per_variant"]]
        return JudgeResult(
            judge_model=self.client.model,
            per_variant=per_variant,
            overall_pass=data["overall_pass"],
        )


def _to_variant_score(v: dict[str, Any]) -> VariantScore:
    """Tolerant of legacy rows that don't carry failure_categories."""
    return VariantScore(
        variant_index=v["variant_index"],
        tone_match=v["tone_match"],
        cta_clarity=v["cta_clarity"],
        segment_relevance=v["segment_relevance"],
        char_limit_ok=v["char_limit_ok"],
        would_send_unedited=v["would_send_unedited"],
        failure_categories=list(v.get("failure_categories", [])),
        notes=v.get("notes", ""),
    )


# ---- Inter-rater agreement -------------------------------------------------


def cohens_kappa(a: list[bool], b: list[bool]) -> float:
    assert len(a) == len(b) and len(a) > 0
    n = len(a)
    po = sum(1 for x, y in zip(a, b) if x == y) / n
    pa_true = sum(a) / n
    pb_true = sum(b) / n
    pe = pa_true * pb_true + (1 - pa_true) * (1 - pb_true)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def bootstrap_ci(
    values: list[float],
    statistic_fn,
    n_iter: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for a single-list statistic."""
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    samples: list[float] = []
    for _ in range(n_iter):
        boot = [values[rng.randrange(n)] for _ in range(n)]
        samples.append(statistic_fn(boot))
    samples.sort()
    lo = samples[int(n_iter * (alpha / 2))]
    hi = samples[int(n_iter * (1 - alpha / 2))]
    return (lo, hi)


def bootstrap_kappa_ci(
    a: list[bool],
    b: list[bool],
    n_iter: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(a)
    samples: list[float] = []
    for _ in range(n_iter):
        idx = [rng.randrange(n) for _ in range(n)]
        boot_a = [a[i] for i in idx]
        boot_b = [b[i] for i in idx]
        samples.append(cohens_kappa(boot_a, boot_b))
    samples.sort()
    lo = samples[int(n_iter * (alpha / 2))]
    hi = samples[int(n_iter * (1 - alpha / 2))]
    return (lo, hi)


@dataclass
class CrossJudgeReport:
    judge_a_model: str
    judge_b_model: str
    judge_a_pass_rate: float
    judge_b_pass_rate: float
    judge_a_pass_rate_ci: tuple[float, float]
    judge_b_pass_rate_ci: tuple[float, float]
    agreement_pct: float
    cohens_kappa: float
    cohens_kappa_ci: tuple[float, float]
    per_subscriber_results: list[dict[str, Any]]


def run_cross_judge(
    pipeline_outputs: list[dict[str, Any]],
    judge_a: Judge,
    judge_b: Judge,
) -> CrossJudgeReport:
    a_flags: list[bool] = []
    b_flags: list[bool] = []
    rows: list[dict[str, Any]] = []

    for out in pipeline_outputs:
        sa = judge_a.score(
            segment=out["segment"],
            persona=out["persona"],
            variants=out["variants"],
        )
        sb = judge_b.score(
            segment=out["segment"],
            persona=out["persona"],
            variants=out["variants"],
        )
        a_flags.append(sa.overall_pass)
        b_flags.append(sb.overall_pass)
        rows.append(
            {
                "subscriber_id": out["subscriber_id"],
                "segment": out["segment"],
                "judge_a": asdict(sa),
                "judge_b": asdict(sb),
                # Keep legacy keys for downstream compatibility.
                "sonnet": asdict(sa),
                "haiku": asdict(sb),
            }
        )

    n = max(1, len(a_flags))
    agreement = sum(1 for x, y in zip(a_flags, b_flags) if x == y) / n
    kappa = cohens_kappa(a_flags, b_flags) if a_flags else 0.0

    a_rate_vals = [1.0 if f else 0.0 for f in a_flags]
    b_rate_vals = [1.0 if f else 0.0 for f in b_flags]
    a_ci = bootstrap_ci(a_rate_vals, lambda xs: sum(xs) / len(xs))
    b_ci = bootstrap_ci(b_rate_vals, lambda xs: sum(xs) / len(xs))
    k_ci = bootstrap_kappa_ci(a_flags, b_flags) if a_flags else (0.0, 0.0)

    return CrossJudgeReport(
        judge_a_model=judge_a.client.model,
        judge_b_model=judge_b.client.model,
        judge_a_pass_rate=(sum(a_flags) / n),
        judge_b_pass_rate=(sum(b_flags) / n),
        judge_a_pass_rate_ci=a_ci,
        judge_b_pass_rate_ci=b_ci,
        agreement_pct=agreement,
        cohens_kappa=kappa,
        cohens_kappa_ci=k_ci,
        per_subscriber_results=rows,
    )
