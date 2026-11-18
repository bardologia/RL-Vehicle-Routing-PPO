import json

import pytest

from findings import RunFindings
from project_paths import RunLayout
from results_browser import RunBrowser
from web_errors import ApiError
from tests.webui.conftest import StubCatalog, agent_block, write_report


def digest(**overrides):
    base = {
        "total_reward"           : -1.0,
        "reward_by_term"         : {"distance_reward": -2.0, "unassigned_reward": 2.0, "idle_reward": 0.0, "priority_reward": 1.0, "action_reward": -0.5, "holding_reward": -1.0},
        "edits_per_round"        : [2, 2, 2, 2],
        "commits_with_work_left" : 0,
        "operator_counts"        : [20, 0, 4],
        "wasted_edits"           : 0,
        "abandonments"           : 0,
        "profitable_left"        : 0,
    }
    base.update(overrides)
    return base


def decision_rows(count, probability=0.9):
    return [{"chosen_joint_prob": probability} for _ in range(count)]


def write_index(run_dir, model, critic=None, ranking=None, decisions=None, sampled=None, seed=1000):
    agents = {"model": model}
    if sampled is not None:
        agents["model_sampled"] = sampled

    index = {
        "episodes"   : [{"index": 0, "agents": agents}],
        "decisions"  : decision_rows(40) if decisions is None else decisions,
        "aggregates" : {
            "critic"      : critic or {"explained_variance": 0.6, "bias": 0.1, "mae": 0.5, "n": 40},
            "job_ranking" : ranking or {"spearman": -0.5, "n": 60},
        },
    }

    target = run_dir / "diagnostics" / f"seed{seed}"
    target.mkdir(parents=True, exist_ok=True)
    (target / "index.json").write_text(json.dumps(index))


def healthy_agents(episodes=40):
    return {
        "model"          : agent_block([2.0] * episodes, [1] * episodes, [3] * episodes, [100.0] * episodes, [0.3, 0.0, 0.7]),
        "teacher"        : agent_block([1.0, 3.0] * (episodes // 2), [1] * episodes, [3] * episodes, [100.0] * episodes, [0.3, 0.1, 0.6]),
        "insertion_only" : agent_block([1.5, 2.5] * (episodes // 2), [1] * episodes, [3] * episodes, [100.0] * episodes, [0.4, 0.0, 0.6]),
        "do_nothing"     : agent_block([0.0] * episodes, [4] * episodes, [2] * episodes, [80.0] * episodes, [0.0, 0.0, 1.0]),
    }


def evaluate(paths, name, curves=None, catalog=None, seed=1000, monkeypatch=None):
    runs = RunBrowser(paths, RunLayout(paths), catalog or StubCatalog())
    runs.run_curves = lambda run_name: {"tags": curves or {}}

    return {finding["id"]: finding for finding in RunFindings(runs, runs.catalog).findings(name, seed)}


def test_sixteen_rules_in_order_and_all_pass_on_a_healthy_run(paths):
    run_dir = paths.run_dir("healthy")
    write_report(run_dir, healthy_agents())
    write_index(run_dir, digest())

    curves = {
        "episode/operator_frequency/op2"          : {"steps": list(range(20)), "values": [0.5] * 20},
        "batch/baseline_stats/explained_variance" : {"steps": list(range(20)), "values": [0.6] * 20},
        "batch/anchor_coefficient"                : {"steps": [0, 1], "values": [0.5, 0.0]},
        "batch/gradients_stats/critic/has_nan"    : {"steps": [0, 1], "values": [0.0, 0.0]},
    }

    runs = RunBrowser(paths, RunLayout(paths), StubCatalog(mtime=0.0))
    runs.run_curves = lambda name: {"tags": curves}
    findings        = RunFindings(runs, runs.catalog).findings("healthy", 1000)

    assert [finding["id"] for finding in findings] == [f"F{number}" for number in range(1, 17)]
    assert {finding["status"] for finding in findings} == {"PASS"}
    assert all(set(finding) == {"id", "status", "title", "metric", "value", "threshold", "n", "text", "needs", "evidence"} for finding in findings)
    assert "Not separable" in findings[1]["text"]


def test_report_rules_fire_on_a_degenerate_model(paths):
    agents          = healthy_agents(10)
    agents["model"] = agent_block([0.0] * 10, [4] * 10, [2] * 10, [80.0] * 10, [0.0, 0.0, 1.0])
    agents["insertion_only"] = agent_block([1.5, 2.5] * 5, [1] * 10, [3] * 10, [100.0] * 10, [0.4, 0.0, 0.6])
    write_report(paths.run_dir("flat"), agents)

    findings = evaluate(paths, "flat")

    assert findings["F1"]["status"] == "FAIL"
    assert findings["F2"]["status"] == "FAIL"
    assert "insertion_only" in findings["F2"]["text"]
    assert findings["F3"]["status"] == "WARN"
    assert findings["F3"]["threshold"] == pytest.approx(0.15)
    assert findings["F4"]["status"] == "WARN"
    assert findings["F1"]["evidence"] == [{"label": "Agent table", "hash": "#/diagnose/flat/1000"}]


def test_checkpoint_rule_reports_mismatch_and_stale_reports(paths):
    write_report(paths.run_dir("run"), healthy_agents())

    broken = evaluate(paths, "run", catalog=StubCatalog(loadable=False, reason="architecture mismatch: 2 missing keys"))
    stale  = evaluate(paths, "run", catalog=StubCatalog(mtime=9e12))

    assert broken["F5"]["status"] == "FAIL"
    assert "architecture mismatch" in broken["F5"]["text"]
    assert stale["F5"]["status"] == "WARN"


def test_rules_without_data_say_what_they_need(paths):
    paths.run_dir("bare").mkdir()

    findings = evaluate(paths, "bare", seed=None)

    assert [findings[f"F{number}"]["needs"] for number in (1, 2, 3, 4)] == ["report"] * 4
    assert [findings[f"F{number}"]["needs"] for number in range(6, 13)] == ["traces"] * 7
    assert [findings[f"F{number}"]["needs"] for number in range(13, 17)] == ["curves"] * 4
    assert all(findings[f"F{number}"]["status"] == "NO DATA" for number in (1, 6, 13))


def test_trace_rules_fire_at_their_thresholds(paths):
    run_dir = paths.run_dir("traced")
    write_report(run_dir, healthy_agents())
    write_index(
        run_dir,
        digest(commits_with_work_left=2, wasted_edits=2, abandonments=1, profitable_left=1, reward_by_term={"distance_reward": -1.0, "unassigned_reward": 1.0, "idle_reward": 0.0, "priority_reward": 0.0, "action_reward": 0.0, "holding_reward": -6.0}),
        critic    = {"explained_variance": 0.05, "bias": -1.0, "mae": 2.0, "n": 40},
        ranking   = {"spearman": 0.02, "n": 60},
        decisions = decision_rows(40, 0.3),
    )

    findings = evaluate(paths, "traced")

    assert all(findings[f"F{number}"]["status"] == "WARN" for number in range(6, 13))
    assert findings["F7"]["value"] == pytest.approx(0.5)
    assert findings["F8"]["value"] == pytest.approx(0.1)
    assert findings["F9"]["value"] == pytest.approx(0.75)
    assert findings["F12"]["value"] == pytest.approx(0.3)


def test_trace_rules_need_enough_decisions_and_react_to_the_sampled_twin(paths):
    run_dir = paths.run_dir("thin")
    write_report(run_dir, healthy_agents())
    write_index(run_dir, digest(total_reward=-2.0), critic={"explained_variance": 0.0, "bias": 0.0, "mae": 1.0, "n": 12}, ranking={"spearman": None, "n": 3}, sampled=digest(total_reward=1.0))

    findings = evaluate(paths, "thin")

    assert findings["F6"]["status"] == "NO DATA"
    assert findings["F6"]["n"] == 12
    assert findings["F10"]["status"] == "NO DATA"
    assert findings["F12"]["status"] == "WARN"
    assert "Sampled twin" in findings["F12"]["text"]


def test_curve_rules_fire_on_drift_weak_critic_anchor_and_nan(paths):
    write_report(paths.run_dir("curved"), healthy_agents())

    curves = {
        "episode/operator_frequency/op2"          : {"steps": list(range(20)), "values": [0.6] * 10 + [0.8] * 10},
        "batch/baseline_stats/explained_variance" : {"steps": list(range(20)), "values": [0.5] * 18 + [0.1, 0.1]},
        "batch/anchor_coefficient"                : {"steps": [0, 1], "values": [0.5, 0.25]},
        "batch/gradients_stats/critic/has_nan"    : {"steps": [0, 1], "values": [0.0, 1.0]},
        "batch/gradients_stats/critic/has_inf"    : {"steps": [0, 1], "values": [0.0, 0.0]},
    }

    findings = evaluate(paths, "curved", curves=curves)

    assert findings["F13"]["status"] == "WARN"
    assert findings["F13"]["value"] == pytest.approx(0.2)
    assert findings["F14"]["status"] == "WARN"
    assert findings["F15"]["status"] == "WARN"
    assert findings["F16"]["status"] == "FAIL"
    assert findings["F16"]["value"] == 1


def test_unknown_run_answers_404(paths):
    runs = RunBrowser(paths, RunLayout(paths), StubCatalog())

    with pytest.raises(ApiError) as raised:
        RunFindings(runs, runs.catalog).findings("nope")

    assert raised.value.status == 404
