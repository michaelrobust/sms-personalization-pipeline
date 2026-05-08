# SMS Pipeline

A 3-stage agent pipeline for generating SMS campaign variants:

```
subscriber features
        │
        ▼
   segmentation        (1 of 6 micro-segments + confidence)
        │
        ▼
   persona framing     (motivation / barrier / voice pillars)
        │              ← optional cohort-learning snippet from context_retrieval
        ▼
   variant generation  (3 distinct SMS variants, ≤160 chars each, with CTA)
```

Plus an evaluation harness:

- segmentation accuracy vs ground truth (`segment_truth` from the synthetic data)
- LLM-as-judge over the variants, run with two different Claude tiers
- Cohen's kappa with bootstrap 95% CI for inter-rater agreement
- pass-rate sliced by segment (cohort error analysis)

## Files

```
data/
  generate_subscribers.py    5,000 synthetic subscribers across 6 segments
  subscribers.csv            (regenerable; checked in for convenience)
schemas.py                   tool schemas (segmentation / persona / variants / judge)
context_retrieval.py         per-cohort copy guidance keyed by (segment, last_category)
agents.py                    SegmentationAgent / PersonaFramingAgent /
                             VariantGenerationAgent / SMSPipeline
eval/
  judge.py                   Judge, run_cross_judge, cohens_kappa, bootstrap CIs
  segmentation_accuracy.py   confusion matrix, per-segment precision/recall/F1
  cohort_analysis.py         pass-rate by segment, worst-cohort lookup
run_pipeline.py              end-to-end runner; writes logs/
```

## Run

```bash
export ANTHROPIC_API_KEY=sk-ant-...

python -m project1_sms_pipeline.data.generate_subscribers
python -m project1_sms_pipeline.run_pipeline --n 30
```

Outputs land in `project1_sms_pipeline/logs/`:

- `pipeline.jsonl` — every agent span with `elapsed_ms`
- `outputs.json` — per-subscriber pipeline results
- `judge_rows.json` — every variant's score from both judges
- `cohort_report.csv` — pass rate sliced by segment
- `summary.txt` — text summary including segmentation accuracy and CIs

## Acceptability rubric

Every variant gets scored on three 0-3 axes plus two booleans:

| axis | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| `tone_match` | wrong tone | weak match | acceptable | locked-in |
| `cta_clarity` | missing | implied | adequate | single-tap obvious |
| `segment_relevance` | could be any segment | weak hook | reasonable | unmistakably for this cohort |

Booleans:
- `char_limit_ok` — `len(body) <= 160`
- `would_send_unedited` — strict pass/fail; one edit needed → False

`overall_pass` is true if at least one of the 3 variants gets
`would_send_unedited = True` from the judge.

## Notes

- The synthetic CRM is generated from a per-segment Gaussian mixture. Real
  subscriber distributions are heavier-tailed — treat metric values as
  comparative, not absolute.
- The cohort-learning library has 8 entries. In production this would be a
  vector-store lookup over hundreds of past-campaign learnings.
- Bootstrap CIs at n=30 are wide. For real evaluation push n past 200.
