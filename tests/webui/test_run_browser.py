import json

import pytest

from project_paths import RunLayout
from results_browser import RunBrowser
from web_errors import ApiError
from tests.webui.conftest import StubCatalog, agent_block, write_report


def browser(paths):
    return RunBrowser(paths, RunLayout(paths), StubCatalog())


def standard_agents():
    return {
        "model"          : agent_block([1.0, 2.0, 3.0, 0.0], [2, 1, 0, 3], [3, 3, 4, 2], [100.0, 110.0, 120.0, 90.0], [0.2, 0.0, 0.8]),
        "teacher"        : agent_block([2.0, 2.0, 1.0, 4.0], [1, 1, 1, 1], [4, 3, 3, 4], [90.0, 100.0, 130.0, 80.0], [0.4, 0.1, 0.5]),
        "insertion_only" : agent_block([0.0, 3.0, 3.0, 1.0], [2, 0, 0, 2], [3, 4, 4, 3], [95.0, 100.0, 120.0, 85.0], [0.5, 0.0, 0.5]),
        "do_nothing"     : agent_block([-1.0, -1.0, 0.0, 0.0], [4, 4, 4, 4], [2, 2, 2, 2], [80.0, 80.0, 80.0, 80.0], [0.0, 0.0, 1.0]),
    }


def test_list_runs_reads_only_the_new_report_schema(paths):
    write_report(paths.run_dir("fresh"), standard_agents())

    old = paths.run_dir("old")
    old.mkdir()
    (old / "evaluation.json").write_text(json.dumps({"model": {"mean_reward": 1.0}}))
    (old / "training.log").write_text("log")

    runs = {run["name"]: run for run in browser(paths).list_runs()}

    assert runs["old"]["reports"] == []
    assert runs["old"]["kind"] == "ppo"
    assert runs["fresh"]["reports"][0]["seed"] == 1000
    assert runs["fresh"]["reports"][0]["episodes"] == 4
    assert runs["fresh"]["reports"][0]["agents"]["model"]["mean_reward"] == pytest.approx(1.5)
    assert runs["fresh"]["loadable"] is True


def test_evaluation_adds_intervals_paired_statistics_and_regret(paths):
    write_report(paths.run_dir("fresh"), standard_agents())

    report = browser(paths).evaluation("fresh", 1000)
    paired = report["paired"]["teacher"]

    assert report["missing_agents"] == []
    assert paired["mean_diff"] == pytest.approx(-0.75)
    assert (paired["wins"], paired["ties"], paired["losses"]) == (1, 1, 2)
    assert paired["ci95"][0] < paired["mean_diff"] < paired["ci95"][1]
    assert paired["ci95"][1] - paired["mean_diff"] == pytest.approx(1.96 * paired["sem_diff"])

    model = report["agents"]["model"]
    assert model["ci95"] == pytest.approx([model["mean_reward"] - 1.96 * model["sem_reward"], model["mean_reward"] + 1.96 * model["sem_reward"]])
    assert model["identical_to"] is None
    assert "per_episode" not in model

    first = report["episodes"][0]
    assert first["seed"] == 1000
    assert first["regret_teacher"] == pytest.approx(1.0)
    assert first["regret_best_baseline"] == pytest.approx(1.0)
    assert first["unassigned_delta"] == 1
    assert first["served_delta"] == -1
    assert first["has_trace"]["model"] is False
    assert [row["regret_best_baseline"] for row in report["episodes"]] == pytest.approx([1.0, 1.0, 0.0, 4.0])


def test_evaluation_flags_a_model_identical_to_do_nothing(paths):
    agents          = standard_agents()
    agents["model"] = agent_block([-1.0, -1.0, 0.0, 0.0], [4, 4, 4, 4], [2, 2, 2, 2], [80.0, 80.0, 80.0, 80.0], [0.0, 0.0, 1.0])
    write_report(paths.run_dir("flat"), agents)

    report = browser(paths).evaluation("flat", 1000)

    assert report["agents"]["model"]["identical_to"] == "do_nothing"
    assert report["agents"]["do_nothing"]["identical_to"] is None
    assert report["paired"]["do_nothing"]["ties"] == 4


def test_partial_report_names_the_missing_agents(paths):
    agents = standard_agents()
    write_report(paths.run_dir("partial"), {"model": agents["model"], "do_nothing": agents["do_nothing"]})

    report = browser(paths).evaluation("partial", 1000)

    assert report["missing_agents"] == ["teacher", "insertion_only"]
    assert report["episodes"][0]["regret_teacher"] is None
    assert report["episodes"][0]["regret_best_baseline"] == pytest.approx(-2.0)
    assert set(report["paired"]) == {"do_nothing"}


def test_has_trace_follows_the_diagnostics_files(paths):
    run_dir = paths.run_dir("fresh")
    write_report(run_dir, standard_agents())

    traces = run_dir / "diagnostics" / "seed1000"
    traces.mkdir(parents=True)
    (traces / "ep2_model.json.gz").write_bytes(b"")

    rows = browser(paths).evaluation("fresh", 1000)["episodes"]

    assert rows[2]["has_trace"]["model"] is True
    assert rows[2]["has_trace"]["teacher"] is False
    assert rows[0]["has_trace"]["model"] is False


def test_unknown_run_and_report_answer_404(paths):
    write_report(paths.run_dir("fresh"), standard_agents())

    for call in (lambda: browser(paths).run_detail("nope"), lambda: browser(paths).evaluation("fresh", 7), lambda: browser(paths).diagnostics("fresh", 1000), lambda: browser(paths).run_curves("../etc")):
        with pytest.raises(ApiError) as raised:
            call()
        assert raised.value.status == 404


def test_run_detail_lists_reports_markdown_logs_and_files(paths):
    run_dir = paths.run_dir("fresh")
    write_report(run_dir, standard_agents())
    (run_dir / "model_summary.md").write_text("| a | b |")
    (run_dir / "pretraining.log").write_text("\x1b[32mgreen\x1b[0m line")

    detail = browser(paths).run_detail("fresh")

    assert detail["reports"] == [1000]
    assert detail["diagnostics"] == []
    assert detail["kind"] == "pretrain"
    assert detail["markdown"] == {"model_summary.md": "| a | b |"}
    assert detail["logs"] == [{"name": "pretraining.log", "text": "green line"}]
    assert {entry["rel"] for entry in detail["files"]} == {"evaluation_seed1000.json", "model_summary.md", "pretraining.log"}


def test_health_returns_the_fixed_charts_with_missing_tags_empty(paths, monkeypatch):
    run_dir = paths.run_dir("fresh")
    run_dir.mkdir()
    (run_dir / "training.log").write_text("log")

    runs = browser(paths)
    monkeypatch.setattr(runs, "run_curves", lambda name: {"tags": {"episode/total_reward": {"steps": [1, 2], "values": [0.5, 1.5]}, "episode/operator_frequency/op2": {"steps": [1], "values": [0.9]}}})

    health = runs.health("fresh")
    charts = {chart["key"]: chart for chart in health["charts"]}

    assert health["kind"] == "ppo"
    assert [chart["key"] for chart in health["charts"]] == ["episode_reward", "operator_frequency", "entropy", "mean_kl", "clip_fraction", "explained_variance", "value_loss", "grad_norm", "anchor_coefficient", "learning_rate"]
    assert charts["episode_reward"]["series"] == [{"label": "episode return", "tag": "episode/total_reward", "steps": [1, 2], "values": [0.5, 1.5]}]
    assert [series["label"] for series in charts["operator_frequency"]["series"]] == ["DO_NOTHING"]
    assert charts["entropy"]["series"] == []
    assert all(chart["unit"] and chart["title"] for chart in health["charts"])
