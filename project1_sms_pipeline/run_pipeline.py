"""End-to-end runner: pipeline -> dual judge -> cohort + segmentation + cost report."""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import time
from pathlib import Path

from shared import AsyncLLMClient, CostLedger, LLMClient

from .agents import AsyncSMSPipeline, SMSPipeline, output_to_dict
from .eval import (
    Judge,
    run_cross_judge,
    slice_by_segment,
    write_report,
    find_worst_cohort,
    format_failure_breakdown,
    evaluate_segmentation,
    format_report,
)

ROOT = Path(__file__).parent
DATA_CSV = ROOT / "data" / "subscribers.csv"
LOG_DIR = ROOT / "logs"


def load_subscribers(n: int) -> list[dict]:
    with DATA_CSV.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    for r in rows:
        r["days_since_last_visit"] = int(r["days_since_last_visit"])
        r["purchase_count_90d"] = int(r["purchase_count_90d"])
        r["avg_order_value"] = float(r["avg_order_value"])
        r["is_loyalty_member"] = r["is_loyalty_member"] == "True"
    return rows[:n]


def _run_sync(subs: list[dict], gen_model: str, ledger: CostLedger) -> list[dict]:
    client = LLMClient(model=gen_model, ledger=ledger)
    pipe = SMSPipeline(client=client, log_path=str(LOG_DIR / "pipeline.jsonl"))
    outputs: list[dict] = []
    for i, sub in enumerate(subs):
        print(f"  [{i+1}/{len(subs)}] sub={sub['subscriber_id']} ...", flush=True)
        outputs.append(pipe.run_one_to_dict(sub))
    return outputs


async def _run_async(
    subs: list[dict],
    gen_model: str,
    ledger: CostLedger,
    concurrency: int,
) -> list[dict]:
    client = AsyncLLMClient(model=gen_model, ledger=ledger)
    pipe = AsyncSMSPipeline(client=client, log_path=str(LOG_DIR / "pipeline.jsonl"))
    print(f"  running {len(subs)} subscribers with concurrency={concurrency} ...", flush=True)
    pipeline_outputs = await pipe.run_many(subs, max_concurrency=concurrency)
    return [output_to_dict(o) for o in pipeline_outputs]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--gen-model", default="claude-sonnet-4-6")
    parser.add_argument("--judge-a", default="claude-sonnet-4-6")
    parser.add_argument("--judge-b", default="claude-haiku-4-5-20251001")
    parser.add_argument("--async-mode", action="store_true")
    parser.add_argument("--concurrency", type=int, default=8)
    args = parser.parse_args()

    LOG_DIR.mkdir(exist_ok=True)
    subs = load_subscribers(args.n)
    print(f"Loaded {len(subs)} subscribers")

    ledger = CostLedger()
    t0 = time.perf_counter()
    if args.async_mode:
        outputs = asyncio.run(
            _run_async(subs, args.gen_model, ledger, args.concurrency)
        )
    else:
        outputs = _run_sync(subs, args.gen_model, ledger)
    pipeline_seconds = time.perf_counter() - t0

    (LOG_DIR / "outputs.json").write_text(json.dumps(outputs, indent=2))

    # Segmentation accuracy vs ground truth (no extra API calls)
    seg_report = evaluate_segmentation(outputs)

    print("\nRunning cross-model judge ...")
    judge_a = Judge(LLMClient(model=args.judge_a, ledger=ledger))
    judge_b = Judge(LLMClient(model=args.judge_b, ledger=ledger))
    report = run_cross_judge(outputs, judge_a=judge_a, judge_b=judge_b)

    (LOG_DIR / "judge_rows.json").write_text(
        json.dumps(report.per_subscriber_results, indent=2)
    )

    stats = slice_by_segment(report.per_subscriber_results, judge_key="judge_a")
    write_report(stats, LOG_DIR / "cohort_report.csv")
    worst = find_worst_cohort(stats)

    ledger.write_json(LOG_DIR / "cost_ledger.json")

    summary = []
    summary.append("SMS Pipeline run summary")
    summary.append("=" * 56)
    summary.append(f"n subscribers       : {len(outputs)}")
    summary.append(f"gen model           : {args.gen_model}")
    summary.append(f"judge A             : {args.judge_a}")
    summary.append(f"judge B             : {args.judge_b}")
    mode_str = "async" + (f" (concurrency={args.concurrency})" if args.async_mode else "") if args.async_mode else "sync"
    summary.append(f"execution mode      : {mode_str}")
    summary.append(f"pipeline wall time  : {pipeline_seconds:.1f}s")
    summary.append("")
    summary.append(format_report(seg_report))
    summary.append("")
    summary.append(
        f"judge A pass rate   : {report.judge_a_pass_rate:.1%} "
        f"(95% CI {report.judge_a_pass_rate_ci[0]:.1%}, {report.judge_a_pass_rate_ci[1]:.1%})"
    )
    summary.append(
        f"judge B pass rate   : {report.judge_b_pass_rate:.1%} "
        f"(95% CI {report.judge_b_pass_rate_ci[0]:.1%}, {report.judge_b_pass_rate_ci[1]:.1%})"
    )
    summary.append(f"agreement           : {report.agreement_pct:.1%}")
    summary.append(
        f"Cohen's kappa       : {report.cohens_kappa:.3f} "
        f"(95% CI {report.cohens_kappa_ci[0]:.3f}, {report.cohens_kappa_ci[1]:.3f})"
    )
    summary.append("")
    summary.append("Per-cohort pass rate (judge A):")
    for s in stats:
        summary.append(
            f"  {s.segment:24s} n={s.n:3d}  pass={s.pass_rate:5.1%}  "
            f"tone={s.avg_tone_match:.2f}  cta={s.avg_cta_clarity:.2f}  "
            f"rel={s.avg_segment_relevance:.2f}"
        )
    if worst:
        summary.append("")
        summary.append(f"Worst cohort: {worst.segment} ({worst.pass_rate:.1%})")
        top = worst.top_failure()
        if top:
            summary.append(f"  top failure category: {top[0]} ({top[1]:.0%})")
    summary.append("")
    summary.append(format_failure_breakdown(stats))
    summary.append("")
    summary.append(ledger.format_summary())

    txt = "\n".join(summary)
    (LOG_DIR / "summary.txt").write_text(txt)
    print()
    print(txt)


if __name__ == "__main__":
    main()
