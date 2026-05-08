"""Cohort-level pass-rate + failure-category slicing for the SMS pipeline."""
from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class CohortStats:
    segment: str
    n: int
    pass_rate: float
    avg_tone_match: float
    avg_cta_clarity: float
    avg_segment_relevance: float
    char_limit_violations: int
    failure_breakdown: dict[str, int] = field(default_factory=dict)
    failure_total: int = 0

    def failure_pct(self, category: str) -> float:
        return self.failure_breakdown.get(category, 0) / self.failure_total if self.failure_total else 0.0

    def top_failure(self) -> tuple[str, float] | None:
        if not self.failure_total:
            return None
        cat, n = max(self.failure_breakdown.items(), key=lambda kv: kv[1])
        return cat, n / self.failure_total


def slice_by_segment(
    judge_rows: list[dict[str, Any]],
    judge_key: str = "judge_a",
) -> list[CohortStats]:
    by_seg: dict[str, list[dict[str, Any]]] = {}
    for row in judge_rows:
        # Tolerate legacy keys.
        if judge_key not in row and judge_key == "judge_a" and "sonnet" in row:
            judge_key = "sonnet"
        by_seg.setdefault(row["segment"], []).append(row)

    out: list[CohortStats] = []
    for seg, rows in by_seg.items():
        passed = sum(1 for r in rows if r[judge_key]["overall_pass"])
        tone, cta, rel, char_viol = [], [], [], 0
        fail_counter: Counter[str] = Counter()
        fail_total = 0
        for r in rows:
            for v in r[judge_key]["per_variant"]:
                tone.append(v["tone_match"])
                cta.append(v["cta_clarity"])
                rel.append(v["segment_relevance"])
                if not v["char_limit_ok"]:
                    char_viol += 1
                if not v["would_send_unedited"]:
                    fail_total += 1
                    for cat in v.get("failure_categories", []) or []:
                        fail_counter[cat] += 1
        out.append(
            CohortStats(
                segment=seg,
                n=len(rows),
                pass_rate=passed / max(1, len(rows)),
                avg_tone_match=sum(tone) / max(1, len(tone)),
                avg_cta_clarity=sum(cta) / max(1, len(cta)),
                avg_segment_relevance=sum(rel) / max(1, len(rel)),
                char_limit_violations=char_viol,
                failure_breakdown=dict(fail_counter),
                failure_total=fail_total,
            )
        )
    return sorted(out, key=lambda c: c.pass_rate)


def write_report(stats: list[CohortStats], path: str | Path) -> Path:
    """Wide-format CSV: one row per segment, one column per failure category."""
    all_categories = sorted({c for s in stats for c in s.failure_breakdown})
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        header = [
            "segment", "n", "pass_rate",
            "avg_tone_match", "avg_cta_clarity", "avg_segment_relevance",
            "char_limit_violations", "failure_total",
        ] + [f"fail_{c}" for c in all_categories]
        writer.writerow(header)
        for s in stats:
            row = [
                s.segment, s.n, f"{s.pass_rate:.3f}",
                f"{s.avg_tone_match:.2f}", f"{s.avg_cta_clarity:.2f}",
                f"{s.avg_segment_relevance:.2f}",
                s.char_limit_violations, s.failure_total,
            ] + [s.failure_breakdown.get(c, 0) for c in all_categories]
            writer.writerow(row)
    return p


def format_failure_breakdown(stats: list[CohortStats]) -> str:
    lines = ["Failure breakdown by cohort (% of variant-level failures):"]
    for s in stats:
        if s.failure_total == 0:
            lines.append(f"  {s.segment:24s} n_fail=0 (no failures)")
            continue
        items = sorted(s.failure_breakdown.items(), key=lambda kv: -kv[1])
        breakdown = "  ".join(
            f"{cat}={n / s.failure_total:.0%}" for cat, n in items
        )
        lines.append(f"  {s.segment:24s} n_fail={s.failure_total:3d}   {breakdown}")
    return "\n".join(lines)


def find_worst_cohort(stats: list[CohortStats]) -> CohortStats | None:
    if not stats:
        return None
    return min(stats, key=lambda c: c.pass_rate)


if __name__ == "__main__":
    import sys

    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("logs/judge_rows.json")
    rows = json.loads(src.read_text())
    stats = slice_by_segment(rows)
    out = write_report(stats, "logs/cohort_report.csv")
    worst = find_worst_cohort(stats)
    print(f"Wrote cohort report -> {out}")
    if worst:
        print(
            f"Worst cohort: {worst.segment} "
            f"(pass_rate={worst.pass_rate:.1%}, n={worst.n})"
        )
        top = worst.top_failure()
        if top:
            print(f"  top failure category: {top[0]} ({top[1]:.0%})")
