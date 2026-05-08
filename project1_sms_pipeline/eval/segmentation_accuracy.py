"""Segmentation accuracy: predicted segment vs ground-truth `segment_truth`."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any


@dataclass
class SegmentationReport:
    n: int
    overall_accuracy: float
    macro_f1: float
    per_segment_recall: dict[str, float]
    per_segment_precision: dict[str, float]
    per_segment_f1: dict[str, float]
    per_segment_support: dict[str, int]
    confusion_matrix: dict[str, dict[str, int]]  # truth -> predicted -> count


def evaluate_segmentation(pipeline_outputs: list[dict[str, Any]]) -> SegmentationReport:
    """Compute classification metrics for the segmentation agent."""
    rows = [
        r
        for r in pipeline_outputs
        if r.get("segment_truth") is not None
    ]
    if not rows:
        return SegmentationReport(
            n=0,
            overall_accuracy=0.0,
            macro_f1=0.0,
            per_segment_recall={},
            per_segment_precision={},
            per_segment_f1={},
            per_segment_support={},
            confusion_matrix={},
        )

    cm: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        cm[r["segment_truth"]][r["segment_predicted"]] += 1

    labels = sorted(set(cm.keys()) | {p for d in cm.values() for p in d.keys()})

    # Per-class precision / recall / f1
    recalls: dict[str, float] = {}
    precisions: dict[str, float] = {}
    f1s: dict[str, float] = {}
    supports: dict[str, int] = {}
    correct_total = 0

    for label in labels:
        tp = cm.get(label, {}).get(label, 0)
        fn = sum(v for k, v in cm.get(label, {}).items() if k != label)
        fp = sum(
            cm.get(other, {}).get(label, 0)
            for other in labels
            if other != label
        )
        support = tp + fn
        recall = tp / support if support else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        recalls[label] = recall
        precisions[label] = precision
        f1s[label] = f1
        supports[label] = support
        correct_total += tp

    overall = correct_total / len(rows)
    macro_f1 = sum(f1s.values()) / max(1, len(f1s))

    # Materialize confusion matrix as plain dicts (default-dict serialization is fragile).
    cm_plain: dict[str, dict[str, int]] = {}
    for truth in labels:
        cm_plain[truth] = {pred: cm.get(truth, {}).get(pred, 0) for pred in labels}

    return SegmentationReport(
        n=len(rows),
        overall_accuracy=overall,
        macro_f1=macro_f1,
        per_segment_recall=recalls,
        per_segment_precision=precisions,
        per_segment_f1=f1s,
        per_segment_support=supports,
        confusion_matrix=cm_plain,
    )


def format_report(rep: SegmentationReport) -> str:
    lines = [
        f"Segmentation accuracy on n={rep.n}",
        f"  overall accuracy : {rep.overall_accuracy:.3f}",
        f"  macro F1         : {rep.macro_f1:.3f}",
        "",
        f"  {'segment':24s} {'support':>7s}  {'precision':>9s}  {'recall':>6s}  {'f1':>5s}",
    ]
    for seg in sorted(rep.per_segment_support):
        lines.append(
            f"  {seg:24s} {rep.per_segment_support[seg]:7d}  "
            f"{rep.per_segment_precision[seg]:9.3f}  "
            f"{rep.per_segment_recall[seg]:6.3f}  "
            f"{rep.per_segment_f1[seg]:5.3f}"
        )
    if rep.confusion_matrix:
        labels = sorted(rep.confusion_matrix.keys())
        lines.append("")
        lines.append("  confusion (rows=truth, cols=predicted):")
        header = "    " + " ".join(f"{l[:10]:>10s}" for l in labels)
        lines.append(header)
        for truth in labels:
            row = "    " + " ".join(
                f"{rep.confusion_matrix[truth].get(pred, 0):>10d}" for pred in labels
            )
            lines.append(row)
    return "\n".join(lines)
