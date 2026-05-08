# SMS Personalization Pipeline

A 3-stage agent pipeline that turns subscriber features into SMS variants,
plus a cross-model judging harness, plus a Thompson-sampling variant
selector, plus a contextual bandit (LinUCB) with off-policy evaluation
(IPS / SNIPS / **doubly-robust** with cross-fitting), plus
**always-valid sequential A/B testing** that survives unlimited peeking,
plus **counterfactual policy learning** (POEM) that trains a new policy
directly from logged bandit data.

```
subscriber features
        │
        ▼
   segmentation        →  one of 6 micro-segments + confidence
        │
        ▼
   persona framing     →  motivation / barrier / voice pillars
        │                 (cohort-learning snippet injected when available)
        ▼
   variant generation  →  3 distinct SMS variants, ≤160 chars each, with CTA
        │
        ▼
   cross-model judge   →  per-variant rubric + failure taxonomy + Cohen's kappa
        │
        ▼
   variant selection   →  Thompson sampling (cohort-level)
                          OR LinUCB (per-user, given feature vector)
        │
        ▼
   off-policy eval     →  IPS / SNIPS estimate of counterfactual policies
                          on logged decisions, with bootstrap 95% CIs
```

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
```

## Four entry points

### 1. End-to-end pipeline + judge

```bash
python -m project1_sms_pipeline.data.generate_subscribers          # one-off

# sequential
python -m project1_sms_pipeline.run_pipeline --n 30

# concurrent (≈ 8x faster on 30 subscribers)
python -m project1_sms_pipeline.run_pipeline --n 30 --async-mode --concurrency 8
```

Outputs (`project1_sms_pipeline/logs/`):

- `pipeline.jsonl` — every agent span with `elapsed_ms`, cache hits
- `outputs.json` — per-subscriber pipeline result (segment + persona + 3 variants)
- `judge_rows.json` — every variant's score from both judges
- `cohort_report.csv` — pass rate sliced by segment, with failure-category columns
- `cost_ledger.json` — input / cached / output tokens + USD cost per model
- `summary.txt` — text summary including:
  - segmentation confusion matrix vs ground truth
  - judge A / B pass rates with 95% bootstrap CIs
  - Cohen's kappa with 95% CI
  - per-cohort pass rate
  - failure breakdown per cohort (`tone_off`, `cta_unclear`, `off_brand`, ...)
  - cost summary (tokens, cache hit rate, USD)

### 2. Thompson sampling variant selector (cohort-level)

```bash
python -m project1_sms_pipeline.run_optimizer --cohort winback_dormant --rounds 500
```

Beta-Bernoulli MAB. One bandit per cohort. Pick variants by Thompson sampling,
update posteriors from binary click rewards. With `--persist`, state is kept
in `logs/bandit_state.json` and shared across runs.

### 3. Contextual bandit (LinUCB) + off-policy evaluation

```bash
python -m project1_sms_pipeline.run_contextual --n 1500 --epsilon 0.1
```

Per-user variant selection. The bandit fits a separate linear regression per
arm on the user's feature vector (10 dims: bias + recency + frequency +
monetary + segment one-hot). Picks the arm with the highest UCB-style score.
Wrapped in epsilon-greedy so every arm has non-zero propensity, which is
required for inverse-propensity scoring.

After the online loop, three counterfactual policies are evaluated against
the logged dataset:

- the trained LinUCB itself (its epsilon-greedy distribution)
- a uniform-random baseline
- (extensible) any deterministic policy via `deterministic_policy(...)`

Both **IPS** (Horvitz-Thompson) and **SNIPS** (self-normalized) estimators
are computed, each with a 500-iteration bootstrap 95% CI. Effective sample
size is reported alongside so you can spot when the IPS estimate is
dominated by a few high-weight samples.

Output (excerpt):

```
Online performance (logging policy):
  cumulative reward   : 203
  empirical mean ctr  : 0.1353

Off-policy estimates (95% bootstrap CI):
  IPS    on logged LinUCB policy : 0.1697 [0.0937, 0.2664]
  SNIPS  on logged LinUCB policy : 0.1576 [0.0953, 0.2410]    ESS=88/1500
  SNIPS  on uniform-random target: 0.0666 [0.0416, 0.0973]    ESS=203/1500

  estimated lift LinUCB vs uniform: +0.0909  (+136.4%)
```

This is the standard production-RL gating pattern: log decisions with their
propensities, then evaluate counterfactual policies offline before
committing to a rollout.

### 4. Advanced OPE + sequential testing + counterfactual policy learning

```bash
python -m project1_sms_pipeline.run_advanced --n 2000
```

Three sections in one run, each printed and saved to `logs/advanced_run.json`:

**Section 1 — variance comparison.** IPS, SNIPS, and cross-fitted DR all
estimate the same target (the trained LinUCB's epsilon-greedy policy)
on the same logged data. Cross-fitting splits the log into K=5 folds,
fits the per-arm ridge reward model on K-1 folds, evaluates the DR sum
on the held-out fold, averages. The doubly-robust property — agreement
across estimators — is the safety check you actually want.

**Section 2 — peeking-safe sequential testing.** Stress test under H0
(p_A == p_B): the always-valid Hoeffding confidence sequence holds
type-I error at 0% (target 5%) regardless of how many times the
marketer peeks. The fixed-horizon t-test with peeking — which is what
many production marketing platforms still use under the hood — gets
type-I error around **30%** in the same simulation. That's the bug.

**Section 3 — POEM counterfactual policy learning.** Collect logs under
a uniform-random logging policy, train a softmax-linear policy by
minimizing the variance-regularized IPS objective (L-BFGS), then
evaluate the trained policy on a held-out fold via SNIPS *and* DR.
The two estimators agree, and the trained policy's value estimate
beats the logging policy's value estimate.

### 5. Static checks (no API key)

```bash
python verify.py
```

42 checks covering imports, data generation, context retrieval, tool
schemas, Cohen's kappa, bootstrap CIs, segmentation accuracy, cohort
slicing with failure categories, cost ledger arithmetic, Thompson
sampling math + convergence + JSON store, LinUCB closed-form update +
JSON state roundtrip + theta-recovery, IPS/SNIPS unbiasedness, ridge
reward-model recovery, DR-estimator consistency, Hoeffding CS type-I
control under H0 + power under H1, and POEM beating uniform when
context matters.

## Layout

```
sms-personalization-pipeline/
├── shared/
│   ├── llm_client.py            sync + async Anthropic client
│   ├── cost_tracker.py          token / USD ledger with per-model pricing
│   └── observability.py         JSONL spans
├── project1_sms_pipeline/
│   ├── data/
│   │   └── generate_subscribers.py  5,000 synthetic subscribers, 6 segments
│   ├── schemas.py                tool schemas + FAILURE_CATEGORIES enum
│   ├── context_retrieval.py      cohort-learning lookup
│   ├── agents.py                 sync + async agents and orchestrators
│   ├── eval/
│   │   ├── judge.py                  LLM judge + Cohen's kappa + bootstrap CI
│   │   ├── segmentation_accuracy.py  classification metrics vs ground truth
│   │   └── cohort_analysis.py        pass rate + failure-category aggregation
│   ├── optimization/
│   │   ├── bandit.py                  Beta-Bernoulli + Thompson sampling
│   │   ├── simulator.py               per-cohort CTR ground truth + bandit loop runner
│   │   ├── contextual_bandit.py       LinUCB + feature extractor + JSON store
│   │   ├── off_policy_eval.py         IPS / SNIPS + bootstrap CI + ESS
│   │   ├── reward_model.py            per-arm ridge regression Q_hat
│   │   ├── doubly_robust.py           cross-fitted DR estimator + bootstrap CI
│   │   ├── sequential_test.py         Hoeffding confidence sequence + A/B runner
│   │   └── counterfactual_learning.py POEM softmax-linear policy + L-BFGS
│   ├── run_pipeline.py
│   ├── run_optimizer.py
│   ├── run_contextual.py
│   └── run_advanced.py
├── verify.py
├── requirements.txt
└── .gitignore
```

## Acceptability rubric

Every variant gets scored by the judge on three 0-3 axes plus two booleans
plus a list of failure categories.

Axes:

| axis | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| `tone_match` | wrong tone | weak | acceptable | locked-in |
| `cta_clarity` | missing | implied | adequate | single-tap obvious |
| `segment_relevance` | could be any segment | weak | reasonable | unmistakably for cohort |

Booleans:
- `char_limit_ok` — `len(body) <= 160`
- `would_send_unedited` — strict pass/fail; one edit needed → False

Failure categories (populated when `would_send_unedited` is False, empty otherwise):

```
tone_off          voice doesn't match the persona's voice_pillars
cta_unclear       CTA missing, fuzzy, or asks for >1 action
off_brand         claims that need legal review, or breaks brand voice
segment_mismatch  could be for any cohort; not specific to this segment
char_limit        exceeds 160 chars
duplicate_angle   not meaningfully different from another variant
other             reasons not covered above
```

The `summary.txt` failure breakdown then lets you say things like "winback
fails 70% on `tone_off` and 20% on `cta_unclear`, so the next prompt
iteration should focus on tone calibration for that cohort."

## Two bandit modes — when to use which

| Mode | What it learns | Best for |
|---|---|---|
| Thompson sampling (cohort-level) | best variant per cohort | small variant slates, fast convergence, no user features |
| LinUCB (contextual) | best variant per **user**, given a feature vector | personalization at scale, when subscriber features matter |

Thompson sampling treats every winback user the same. LinUCB lets the
"high-AOV winback in apparel" branch off from the "low-AOV winback in
beauty" branch — same cohort, different best variant.

## Off-policy evaluation, briefly

Given a log of decisions `(context, action, propensity, reward)` from any
logging policy `mu`, the IPS estimator computes the expected reward of any
target policy `pi`:

```
V_pi = (1/N) * Σ_i  pi(a_i | x_i) / mu(a_i | x_i)  *  r_i
```

The self-normalized estimator (SNIPS) divides by the sum of importance
weights, which trades a small bias for substantially lower variance — it
is the more practical estimator most of the time.

The doubly-robust estimator combines IPS with a learned reward model
`Q_hat(x, a)`:

```
V_DR = (1/N) * Σ_i [ Σ_a pi(a|x_i) Q_hat(x_i, a)
                  + (pi(a_i|x_i)/mu(a_i|x_i)) * (r_i - Q_hat(x_i, a_i)) ]
```

DR is unbiased if EITHER the propensities OR the reward model is correct
(hence "doubly robust"). With cross-fitting (Chernozhukov 2018), the
reward model is trained on disjoint folds from the data used to evaluate
the DR sum, removing the bias from training and evaluating on the same
data. This is what production OPE looks like.

All three estimators (IPS, SNIPS, DR) require `mu(a_i | x_i) > 0` for every
logged action — the LinUCB here is wrapped in epsilon-greedy specifically
to satisfy this.

This is how production teams ship a new bandit safely: run the current
policy with logged propensities → estimate the value of the new policy
offline with IPS/SNIPS/DR → if the offline estimate is convincing, run
a small peek-safe online sequential test → roll out.

## Counterfactual policy learning (POEM)

Instead of running a new bandit online to learn a policy, POEM
(Swaminathan & Joachims 2015) trains the policy directly from a logged
dataset. Given logs from any logging policy `mu`, find policy
parameters `θ` that maximize:

```
L(θ) = V_IPS(π_θ)  -  λ · sqrt( Var_IPS(π_θ) / N )
```

The first term is the IPS estimate of the new policy's expected reward.
The variance term penalizes policies that put high mass on rarely-logged
actions — these would have huge importance weights and unstable IPS
estimates. We parametrize π_θ as a softmax over linear scores and
optimize with L-BFGS-B (closed-form gradient on the IPS term).

The deliverable: a policy trained offline that is verifiably better than
the logging policy, evaluated independently with DR + SNIPS before
rollout. No online A/B, no risk to live traffic until you have a CI.

## Always-valid sequential testing

Standard fixed-horizon t-tests fail when a marketer peeks at running
results — the type-I error rate compounds with every peek. A
*confidence sequence* is a sequence of intervals `(CI_t)` such that:

```
P(  for all t >= 1,  μ ∈ CI_t  )  >=  1 - α
```

Equivalently: **stop at any data-dependent rule, the coverage holds**.
Peek as much as you want.

We use a Hoeffding-type uniform CS for [0, 1]-bounded rewards
(Howard, Ramdas, McAuliffe, Sekhon 2021), then run two independent CSs
at level α/2 for the two arms and intersect to get a CS for Δ = p_A - p_B.
The marketer can stop the experiment the instant the CS for Δ excludes 0.

The `simulate_type_i_error` helper stress-tests this: under H0 (p_A == p_B),
the CS rejects ≤ α of the time regardless of horizon or peeking frequency.
The fixed-horizon-with-peeking baseline rejects roughly 30% of the time at
α=0.05 (massively inflated type-I error). That contrast is the section 2
demo's headline result.

## Notes

- The synthetic CRM is generated from per-segment Gaussian mixtures. Real
  subscriber distributions are heavier-tailed; treat metric values as
  comparative.
- The cohort-learning library has 8 entries. Production would retrieve via
  embedding search over hundreds of past-campaign learnings.
- Bandit state persists across runs; deleting `logs/bandit_state.json` or
  `logs/linucb_state.json` resets it.
- The CTR simulator is illustrative. In production the bandit consumes real
  click events from your delivery webhook.
- Per-model pricing in `shared/cost_tracker.py` reflects published rates at
  build time. Update if Anthropic publishes new pricing.


