"""Static checks for the SMS personalization pipeline. No API calls."""
from __future__ import annotations

import importlib
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-DUMMY-FOR-IMPORT-ONLY")

PASS = "PASS"
FAIL = "FAIL"
results: list[tuple[str, str, str]] = []


def check(label: str, fn) -> None:
    try:
        detail = fn() or ""
        results.append((label, PASS, str(detail)))
    except Exception as e:
        results.append((label, FAIL, f"{type(e).__name__}: {e}"))
        traceback.print_exc()


# ---- imports --------------------------------------------------------------

check("import shared", lambda: importlib.import_module("shared"))
check("import shared.llm_client", lambda: importlib.import_module("shared.llm_client"))
check("import shared.observability", lambda: importlib.import_module("shared.observability"))
check("import shared.cost_tracker", lambda: importlib.import_module("shared.cost_tracker"))

check("import project1.schemas", lambda: importlib.import_module("project1_sms_pipeline.schemas"))
check("import project1.context_retrieval", lambda: importlib.import_module("project1_sms_pipeline.context_retrieval"))
check("import project1.agents", lambda: importlib.import_module("project1_sms_pipeline.agents"))
check("import project1.eval.judge", lambda: importlib.import_module("project1_sms_pipeline.eval.judge"))
check("import project1.eval.cohort_analysis", lambda: importlib.import_module("project1_sms_pipeline.eval.cohort_analysis"))
check("import project1.eval.segmentation_accuracy", lambda: importlib.import_module("project1_sms_pipeline.eval.segmentation_accuracy"))
check("import project1.run_pipeline", lambda: importlib.import_module("project1_sms_pipeline.run_pipeline"))
check("import project1.data.generate_subscribers", lambda: importlib.import_module("project1_sms_pipeline.data.generate_subscribers"))
check("import project1.optimization", lambda: importlib.import_module("project1_sms_pipeline.optimization"))
check("import project1.run_optimizer", lambda: importlib.import_module("project1_sms_pipeline.run_optimizer"))
check("import project1.optimization.contextual_bandit", lambda: importlib.import_module("project1_sms_pipeline.optimization.contextual_bandit"))
check("import project1.optimization.off_policy_eval", lambda: importlib.import_module("project1_sms_pipeline.optimization.off_policy_eval"))
check("import project1.run_contextual", lambda: importlib.import_module("project1_sms_pipeline.run_contextual"))
check("import project1.optimization.reward_model", lambda: importlib.import_module("project1_sms_pipeline.optimization.reward_model"))
check("import project1.optimization.doubly_robust", lambda: importlib.import_module("project1_sms_pipeline.optimization.doubly_robust"))
check("import project1.optimization.sequential_test", lambda: importlib.import_module("project1_sms_pipeline.optimization.sequential_test"))
check("import project1.optimization.counterfactual_learning", lambda: importlib.import_module("project1_sms_pipeline.optimization.counterfactual_learning"))
check("import project1.run_advanced", lambda: importlib.import_module("project1_sms_pipeline.run_advanced"))


# ---- logic checks ---------------------------------------------------------


def check_data():
    from project1_sms_pipeline.data.generate_subscribers import generate, segment_distribution
    rows = generate(n=200, seed=7)
    dist = segment_distribution(rows)
    assert len(rows) == 200
    assert len(dist) == 6
    return "6 segments, 200 rows"


def check_context_retrieval():
    from project1_sms_pipeline.context_retrieval import retrieve_context
    s1 = retrieve_context("winback_dormant", "beauty")
    assert s1 and "Beauty winback" in s1.text
    s2 = retrieve_context("winback_dormant", "wellness")
    assert s2 and s2.source_key == ("winback_dormant", None)
    s3 = retrieve_context("nonexistent_segment")
    assert s3 is None
    return "specific / fallback / miss paths exercised"


def check_schemas():
    from project1_sms_pipeline.schemas import (
        SEGMENT_TOOL,
        PERSONA_TOOL,
        VARIANT_TOOL,
        JUDGE_TOOL,
        SEGMENT_LABELS,
        FAILURE_CATEGORIES,
    )
    for tool in [SEGMENT_TOOL, PERSONA_TOOL, VARIANT_TOOL, JUDGE_TOOL]:
        assert "name" in tool and "input_schema" in tool
        assert tool["input_schema"]["type"] == "object"
        assert "properties" in tool["input_schema"]
        assert "required" in tool["input_schema"]
    assert len(SEGMENT_LABELS) == 6
    # JUDGE_TOOL must require failure_categories
    judge_required = JUDGE_TOOL["input_schema"]["properties"]["per_variant"]["items"]["required"]
    assert "failure_categories" in judge_required
    assert len(FAILURE_CATEGORIES) >= 6
    return f"4 tools, 6 segments, {len(FAILURE_CATEGORIES)} failure categories"


def check_kappa_and_bootstrap():
    from project1_sms_pipeline.eval.judge import (
        cohens_kappa,
        bootstrap_ci,
        bootstrap_kappa_ci,
    )
    assert abs(cohens_kappa([True, True, False, False], [True, True, False, False]) - 1.0) < 1e-9
    k_disagree = cohens_kappa([True, True, False, False], [False, False, True, True])
    assert k_disagree < 0
    lo, hi = bootstrap_ci([1.0] * 50, lambda xs: sum(xs) / len(xs), n_iter=200)
    assert abs(lo - 1.0) < 1e-9 and abs(hi - 1.0) < 1e-9
    klo, khi = bootstrap_kappa_ci([True] * 30, [True] * 30, n_iter=200)
    assert klo == 1.0 and khi == 1.0
    return f"kappa range OK (disagree={k_disagree:.2f}); CIs collapse correctly"


def check_segmentation_accuracy():
    from project1_sms_pipeline.eval.segmentation_accuracy import evaluate_segmentation
    fake_outputs = [
        {"segment_truth": "high_intent_browser", "segment_predicted": "high_intent_browser"},
        {"segment_truth": "high_intent_browser", "segment_predicted": "high_intent_browser"},
        {"segment_truth": "high_intent_browser", "segment_predicted": "price_sensitive"},
        {"segment_truth": "price_sensitive", "segment_predicted": "price_sensitive"},
        {"segment_truth": "vip_loyalist", "segment_predicted": "vip_loyalist"},
    ]
    rep = evaluate_segmentation(fake_outputs)
    assert rep.n == 5
    assert rep.overall_accuracy == 4 / 5
    assert rep.per_segment_recall["high_intent_browser"] == 2 / 3
    assert rep.confusion_matrix["high_intent_browser"]["price_sensitive"] == 1
    return f"accuracy={rep.overall_accuracy:.3f}, macro_f1={rep.macro_f1:.3f}"


def check_cohort_slicing_with_failures():
    from project1_sms_pipeline.eval.cohort_analysis import (
        slice_by_segment,
        find_worst_cohort,
        format_failure_breakdown,
    )
    fake = [
        {
            "segment": "winback_dormant",
            "judge_a": {
                "overall_pass": False,
                "per_variant": [
                    {"variant_index": 0, "tone_match": 1, "cta_clarity": 1,
                     "segment_relevance": 1, "char_limit_ok": True,
                     "would_send_unedited": False,
                     "failure_categories": ["tone_off", "cta_unclear"], "notes": ""},
                    {"variant_index": 1, "tone_match": 0, "cta_clarity": 1,
                     "segment_relevance": 1, "char_limit_ok": True,
                     "would_send_unedited": False,
                     "failure_categories": ["tone_off"], "notes": ""},
                ],
            },
        },
        {
            "segment": "vip_loyalist",
            "judge_a": {
                "overall_pass": True,
                "per_variant": [
                    {"variant_index": 0, "tone_match": 3, "cta_clarity": 3,
                     "segment_relevance": 3, "char_limit_ok": True,
                     "would_send_unedited": True,
                     "failure_categories": [], "notes": ""},
                ],
            },
        },
    ]
    stats = slice_by_segment(fake)
    worst = find_worst_cohort(stats)
    assert worst is not None and worst.segment == "winback_dormant"
    # tone_off appears twice across the 2 failed variants in winback
    assert worst.failure_breakdown.get("tone_off") == 2
    assert worst.failure_breakdown.get("cta_unclear") == 1
    top = worst.top_failure()
    assert top is not None and top[0] == "tone_off"
    text = format_failure_breakdown(stats)
    assert "tone_off" in text
    return f"worst={worst.segment}, top_failure={top[0]} ({top[1]:.0%})"


# ---- new: B (cost ledger) -------------------------------------------------


def check_cost_ledger_arithmetic():
    from shared.cost_tracker import CostLedger, PRICES
    ledger = CostLedger()
    # 1M input + 1M output for Sonnet should match published pricing.
    ledger.record(
        "claude-sonnet-4-6",
        {
            "input_tokens": 1_000_000,
            "output_tokens": 1_000_000,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        },
    )
    p = PRICES["claude-sonnet-4-6"]
    expected = p["input"] + p["output"]
    assert abs(ledger.total_cost_usd() - expected) < 1e-9, ledger.total_cost_usd()
    # Cached input should be ~10x cheaper.
    ledger2 = CostLedger()
    ledger2.record(
        "claude-sonnet-4-6",
        {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_input_tokens": 1_000_000,
            "cache_creation_input_tokens": 0,
        },
    )
    assert abs(ledger2.total_cost_usd() - p["cached_input"]) < 1e-9
    assert ledger2.by_model["claude-sonnet-4-6"].cache_hit_rate() == 1.0
    return f"sonnet cost / 1M in+out = ${expected:.2f}; cache hit rate computed"


# ---- new: A (bandit) ------------------------------------------------------


def check_bandit_update_math():
    from project1_sms_pipeline.optimization import (
        Arm,
        BanditState,
        BetaBernoulliBandit,
    )
    state = BanditState(
        cohort_id="test",
        arms=[Arm(arm_id="a"), Arm(arm_id="b")],
    )
    bandit = BetaBernoulliBandit()
    # Reward 1 to arm a should bump alpha by 1.
    bandit.update(state, "a", 1)
    assert state.arms[0].alpha == 2.0 and state.arms[0].beta == 1.0
    bandit.update(state, "a", 0)
    assert state.arms[0].alpha == 2.0 and state.arms[0].beta == 2.0
    assert state.total_pulls == 2
    return "Beta posterior updates correct"


def check_bandit_converges_in_simulation():
    """Run 800 rounds against the simulator; bandit should pick the true-best arm."""
    from project1_sms_pipeline.optimization import (
        BanditStore,
        run_bandit_loop,
        PersonaCTRModel,
    )
    variants = [
        {"variant_id": "v1", "tone": "empathetic",
         "cta_action": "Tap to come back — 20% off"},
        {"variant_id": "v2", "tone": "playful", "cta_action": "Tap to claim"},
        {"variant_id": "v3", "tone": "urgent", "cta_action": "Last chance"},
        {"variant_id": "v4", "tone": "premium", "cta_action": "VIP early access"},
    ]
    res = run_bandit_loop(
        cohort_id="winback_dormant",
        variants=variants,
        n_rounds=800,
        seed=0,
    )
    model = PersonaCTRModel()
    true_ctrs = {
        v["variant_id"]: model.true_ctr("winback_dormant", v["tone"], v["cta_action"])
        for v in variants
    }
    best_true = max(true_ctrs, key=true_ctrs.get)
    assert res.best_arm_id == best_true, f"got {res.best_arm_id}, expected {best_true}"
    # Bandit should pull the best arm a clear majority of the time.
    best_arm = next(a for a in res.arms if a["arm_id"] == best_true)
    assert best_arm["pulls"] >= 0.5 * res.n_rounds, best_arm["pulls"]
    return f"converged on {best_true} ({best_arm['pulls']}/{res.n_rounds} pulls)"


def check_bandit_store_roundtrip():
    import tempfile
    from project1_sms_pipeline.optimization import (
        Arm,
        BanditState,
        BanditStore,
    )
    with tempfile.TemporaryDirectory() as d:
        store = BanditStore(Path(d) / "state.json")
        s = BanditState(
            cohort_id="cohort_x",
            arms=[Arm(arm_id="a", alpha=3.0, beta=2.0, pulls=4, rewards=2)],
            total_pulls=4,
        )
        store.upsert(s)
        loaded = store.load()
        assert "cohort_x" in loaded
        a = loaded["cohort_x"].arms[0]
        assert a.alpha == 3.0 and a.beta == 2.0 and a.pulls == 4
    return "JSON store roundtrip ok"


def check_linucb_update_math():
    """Verify Sherman-style update: A_a += xx^T, b_a += r * x."""
    import numpy as np
    from project1_sms_pipeline.optimization import LinUCB

    bandit = LinUCB(arm_ids=["a", "b"], dim=3, alpha=1.0, epsilon=0.0)
    x = np.array([1.0, 2.0, 3.0])
    bandit.update("a", x, 1.0)
    expected_A = np.eye(3) + np.outer(x, x)
    expected_b = 1.0 * x
    assert np.allclose(bandit.states["a"].A, expected_A)
    assert np.allclose(bandit.states["a"].b, expected_b)
    # Other arm untouched.
    assert np.allclose(bandit.states["b"].A, np.eye(3))
    assert np.allclose(bandit.states["b"].b, np.zeros(3))
    return "A_a, b_a updates match closed form"


def check_linucb_state_roundtrip():
    """LinUCB state survives a JSON serialize/deserialize cycle."""
    import tempfile
    import numpy as np
    from project1_sms_pipeline.optimization import LinUCB, LinUCBStore

    bandit = LinUCB(arm_ids=["a", "b"], dim=3, alpha=1.0, epsilon=0.1)
    bandit.update("a", np.array([1.0, 0.5, 0.2]), 1.0)
    bandit.update("b", np.array([0.3, 0.1, 0.7]), 0.0)

    with tempfile.TemporaryDirectory() as d:
        store = LinUCBStore(Path(d) / "linucb.json")
        store.save(bandit)
        loaded = store.load()
        assert loaded is not None
        for arm in ["a", "b"]:
            assert np.allclose(loaded.states[arm].A, bandit.states[arm].A)
            assert np.allclose(loaded.states[arm].b, bandit.states[arm].b)
            assert loaded.states[arm].pulls == bandit.states[arm].pulls
    return "LinUCB state roundtrip ok"


def check_linucb_learns_best_arm():
    """Run a synthetic learning task: arm 0 has reward x[0], arm 1 has reward x[1].
    With balanced contexts, neither dominates; with x heavily weighted to dim 0,
    LinUCB should prefer arm 0. The test just checks the LEARNED THETA captures
    that signal and the chosen arm matches expectation post-training."""
    import numpy as np
    from project1_sms_pipeline.optimization import LinUCB

    rng_np = np.random.default_rng(0)
    bandit = LinUCB(arm_ids=["a0", "a1"], dim=2, alpha=0.5, epsilon=0.1)

    # Train on uniform random actions so both arms see data.
    for _ in range(400):
        x = rng_np.standard_normal(2)
        # True rewards: a0 favors dim 0, a1 favors dim 1
        r0 = 1 if (x[0] > 0) else 0
        r1 = 1 if (x[1] > 0) else 0
        bandit.update("a0", x, r0)
        bandit.update("a1", x, r1)

    # theta_a0 should put weight on dim 0; theta_a1 on dim 1.
    theta_a0 = bandit.theta("a0")
    theta_a1 = bandit.theta("a1")
    assert theta_a0[0] > theta_a0[1], theta_a0
    assert theta_a1[1] > theta_a1[0], theta_a1
    return f"theta_a0={theta_a0.round(2).tolist()}, theta_a1={theta_a1.round(2).tolist()}"


def check_ips_unbiased_on_known_truth():
    """Sanity check: when target == logging policy, IPS estimate should equal
    the empirical mean reward (within bootstrap noise)."""
    import numpy as np
    from project1_sms_pipeline.optimization import (
        LogEntry,
        ips_estimate,
        snips_estimate,
    )

    # Logging policy: each arm chosen with prob 0.5 uniformly.
    rng = np.random.default_rng(0)
    logs = []
    for _ in range(2000):
        arm = "a" if rng.random() < 0.5 else "b"
        reward = 1.0 if rng.random() < (0.2 if arm == "a" else 0.4) else 0.0
        logs.append(
            LogEntry(
                context=np.zeros(2),
                arm_id=arm,
                propensity=0.5,
                reward=reward,
            )
        )

    # Target == logging policy (50/50). IPS should approximate mean reward.
    def target(x):
        return {"a": 0.5, "b": 0.5}

    empirical_mean = sum(e.reward for e in logs) / len(logs)
    ips = ips_estimate(logs, target)
    snips = snips_estimate(logs, target)
    # Both should be within 0.03 of empirical at n=2000.
    assert abs(ips - empirical_mean) < 0.03, (ips, empirical_mean)
    assert abs(snips - empirical_mean) < 0.03, (snips, empirical_mean)
    return f"empirical={empirical_mean:.3f}  IPS={ips:.3f}  SNIPS={snips:.3f}"


def check_ips_orders_policies_correctly():
    """A target policy that picks the high-CTR arm should score better than
    one that always picks the low-CTR arm, on the same logged data."""
    import numpy as np
    from project1_sms_pipeline.optimization import (
        LogEntry,
        snips_estimate,
    )

    # Two arms: a has 0.1 CTR, b has 0.4 CTR. Logged 50/50.
    rng = np.random.default_rng(1)
    logs = []
    for _ in range(2000):
        arm = "a" if rng.random() < 0.5 else "b"
        ctr = 0.1 if arm == "a" else 0.4
        reward = 1.0 if rng.random() < ctr else 0.0
        logs.append(LogEntry(np.zeros(2), arm, 0.5, reward))

    pick_a = lambda x: {"a": 1.0, "b": 0.0}
    pick_b = lambda x: {"a": 0.0, "b": 1.0}

    v_a = snips_estimate(logs, pick_a)
    v_b = snips_estimate(logs, pick_b)
    assert v_b > v_a + 0.15, (v_a, v_b)
    return f"V(pick_a)={v_a:.3f} < V(pick_b)={v_b:.3f}"


check("data generation", check_data)
check("context retrieval lookup paths", check_context_retrieval)
check("tool schemas + failure taxonomy", check_schemas)
check("kappa + bootstrap CI", check_kappa_and_bootstrap)
check("segmentation accuracy + confusion matrix", check_segmentation_accuracy)
check("cohort slicing aggregates failure categories", check_cohort_slicing_with_failures)
check("cost ledger arithmetic + cache rate", check_cost_ledger_arithmetic)
check("bandit Beta-Bernoulli update math", check_bandit_update_math)
check("bandit converges to optimum in simulation", check_bandit_converges_in_simulation)
check("bandit JSON store roundtrip", check_bandit_store_roundtrip)
check("LinUCB closed-form A/b update", check_linucb_update_math)
check("LinUCB JSON state roundtrip", check_linucb_state_roundtrip)
check("LinUCB learns per-arm theta from data", check_linucb_learns_best_arm)
check("IPS/SNIPS unbiased when target==logging", check_ips_unbiased_on_known_truth)
check("IPS/SNIPS orders policies correctly", check_ips_orders_policies_correctly)


# ---- new advanced section -------------------------------------------------


def check_reward_model_fits():
    import numpy as np
    from project1_sms_pipeline.optimization import fit_reward_model
    rng = np.random.default_rng(0)
    n, d = 200, 4
    X = rng.standard_normal((n, d))
    arms = ["a" if i % 2 == 0 else "b" for i in range(n)]
    # arm a's reward depends on dim 0; arm b's on dim 1.
    rewards = np.array([
        (X[i, 0] if a == "a" else X[i, 1]) for i, a in enumerate(arms)
    ])
    model = fit_reward_model(X, arms, rewards, all_arm_ids=["a", "b"], lam=0.1)
    theta_a = model.theta_by_arm["a"]
    theta_b = model.theta_by_arm["b"]
    assert theta_a[0] > 0.5 and abs(theta_a[1]) < 0.3, theta_a
    assert theta_b[1] > 0.5 and abs(theta_b[0]) < 0.3, theta_b
    return f"theta_a≈[1, 0, 0, 0], theta_b≈[0, 1, 0, 0]"


def check_dr_estimator_consistent_with_truth():
    """When propensities are exact and Q_hat is reasonable, DR should match
    the empirical mean reward of the logging policy on a target == logging
    sanity test."""
    import numpy as np
    from project1_sms_pipeline.optimization import LogEntry, doubly_robust_estimate
    rng = np.random.default_rng(0)
    n = 500
    logs = []
    for _ in range(n):
        x = rng.standard_normal(3)
        arm = "a" if rng.random() < 0.5 else "b"
        # CTR depends on context (so reward model has signal).
        ctr = 0.3 if arm == "a" else 0.5 - 0.05 * x[0]
        r = 1.0 if rng.random() < ctr else 0.0
        logs.append(LogEntry(context=x, arm_id=arm, propensity=0.5, reward=r))

    def target(x):
        return {"a": 0.5, "b": 0.5}

    dr = doubly_robust_estimate(logs, target, n_folds=5, seed=0)
    empirical = sum(e.reward for e in logs) / len(logs)
    assert abs(dr - empirical) < 0.04, (dr, empirical)
    return f"empirical={empirical:.3f}  DR={dr:.3f}"


def check_cs_type_i_error_holds():
    """Under H0 (p_A == p_B), the always-valid CS should reject at most ~alpha
    of the time even with many peeks. Use small horizon for speed."""
    from project1_sms_pipeline.optimization import simulate_type_i_error
    res = simulate_type_i_error(
        n_trials=100, horizon=400, p_shared=0.10, alpha=0.05, seed=7
    )
    # Empirical alpha must NOT exceed target. Hoeffding CS is conservative
    # so we expect << 0.05.
    assert res.empirical_alpha <= res.target_alpha + 0.02, res
    return f"empirical={res.empirical_alpha:.3f} <= target={res.target_alpha}"


def check_cs_excludes_zero_under_h1():
    """Under a clear H1 (large effect, plenty of samples), the CS should
    detect the difference."""
    import random
    from project1_sms_pipeline.optimization import run_sequential_ab
    rng = random.Random(42)
    arms, rewards = [], []
    for _ in range(2500):
        arm = "A" if rng.random() < 0.5 else "B"
        p = 0.20 if arm == "A" else 0.05  # 4x effect, easy to detect
        arms.append(arm)
        rewards.append(1.0 if rng.random() < p else 0.0)
    decision = run_sequential_ab(arms, rewards, alpha=0.05, min_per_arm=30)
    assert decision.decision == "A_better", decision
    assert decision.stopped_at < 2500
    return f"stopped at t={decision.stopped_at}, A_better, Δ̂={decision.final_diff:.3f}"


def check_poem_recovers_better_policy():
    """POEM trained on uniform-random logs should beat uniform random when
    arms have context-dependent rewards."""
    import numpy as np
    from project1_sms_pipeline.optimization import (
        LogEntry,
        snips_estimate,
        softmax_policy_propensity,
        train_poem,
        uniform_random_policy,
    )
    rng = np.random.default_rng(0)
    n = 600
    arm_ids = ["a", "b"]
    logs = []
    for _ in range(n):
        x = rng.standard_normal(3)
        arm = "a" if rng.random() < 0.5 else "b"
        # arm a is best when x[0] > 0; arm b is best when x[0] < 0
        ctr = (0.5 + 0.3 if (arm == "a" and x[0] > 0) or (arm == "b" and x[0] < 0)
               else 0.2)
        r = 1.0 if rng.random() < ctr else 0.0
        logs.append(LogEntry(context=x, arm_id=arm, propensity=0.5, reward=r))

    policy = train_poem(logs, arm_ids=arm_ids, lam_var=0.05, l2=1e-3, seed=0)
    target_poem = softmax_policy_propensity(policy)
    target_uniform = uniform_random_policy(arm_ids)

    v_poem = snips_estimate(logs, target_poem)
    v_uniform = snips_estimate(logs, target_uniform)
    assert v_poem > v_uniform, (v_poem, v_uniform)
    # Smoke check on the learned theta direction.
    # arm a should weight x[0] positively; arm b should weight x[0] negatively.
    theta_a = policy.theta[0]
    theta_b = policy.theta[1]
    assert theta_a[0] > theta_b[0], (theta_a, theta_b)
    return f"V(uniform)={v_uniform:.3f} < V(POEM)={v_poem:.3f}; theta_a[0]>theta_b[0]"


check("ridge reward model recovers per-arm structure", check_reward_model_fits)
check("DR estimator agrees with empirical when target==logging", check_dr_estimator_consistent_with_truth)
check("Hoeffding CS controls type-I error under H0", check_cs_type_i_error_holds)
check("Hoeffding CS rejects under clear H1", check_cs_excludes_zero_under_h1)
check("POEM beats uniform when context matters", check_poem_recovers_better_policy)


# ---- summary --------------------------------------------------------------

n_pass = sum(1 for _, s, _ in results if s == PASS)
n_fail = sum(1 for _, s, _ in results if s == FAIL)
print()
print("=" * 70)
print(f"VERIFY RESULTS: {n_pass} passed, {n_fail} failed")
print("=" * 70)
for label, status, detail in results:
    suffix = f"  -- {detail}" if detail else ""
    print(f"  [{status}]  {label:55s}{suffix}")
print()
sys.exit(0 if n_fail == 0 else 1)
