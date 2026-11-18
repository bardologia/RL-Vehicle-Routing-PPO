import torch

from checkpoint_catalog import CheckpointCatalog
from console_runtime import ConsoleRuntime
from model.policy_model import Policy, PolicyCheckpoint
from tests.webui.conftest import write_report


def save(paths, name, mutate=None):
    state = Policy(ConsoleRuntime(paths).config).state_dict()
    if mutate is not None:
        mutate(state)

    paths.run_dir(name).mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": state, "training_state": None}, paths.run_dir(name) / "graph_ppo_policy.pt")


def drop_and_resize(state):
    state.pop("graph_embedding.context_encoder.weight")
    state["operator_actor.0.weight"] = torch.zeros(4, 4)
    state["extra.weight"]            = torch.zeros(1)


def test_matching_checkpoint_is_loadable_and_lists_its_reports(paths):
    save(paths, "good")
    write_report(paths.run_dir("good"), {}, seed=77)
    (paths.run_dir("good") / "diagnostics" / "seed77").mkdir(parents=True)
    (paths.run_dir("good") / "diagnostics" / "seed77" / "index.json").write_text("{}")
    (paths.run_dir("good") / "pretraining.log").write_text("log")
    paths.run_dir("empty").mkdir()

    entries = CheckpointCatalog(ConsoleRuntime(paths)).list()

    assert [entry["run"] for entry in entries] == ["good"]
    assert entries[0]["loadable"] is True
    assert entries[0]["reason"] is None
    assert entries[0]["kind"] == "pretrain"
    assert entries[0]["reports"] == [77]
    assert entries[0]["diagnostics"] == [77]


def test_mismatched_checkpoint_states_the_reason(paths):
    save(paths, "bad", drop_and_resize)

    catalog = CheckpointCatalog(ConsoleRuntime(paths))
    status  = catalog.status(paths.run_dir("bad"))

    assert status["loadable"] is False
    assert status["reason"] == "architecture mismatch: 1 missing keys, 1 unexpected keys, 1 shape mismatches"


def test_status_is_cached_until_the_file_changes(paths, monkeypatch):
    save(paths, "good")

    catalog = CheckpointCatalog(ConsoleRuntime(paths))
    calls   = []
    compare = catalog._compare
    monkeypatch.setattr(catalog, "_compare", lambda path: calls.append(path) or compare(path))

    catalog.status(paths.run_dir("good"))
    catalog.status(paths.run_dir("good"))
    assert len(calls) == 1

    save(paths, "good", drop_and_resize)
    assert catalog.status(paths.run_dir("good"))["loadable"] is False
    assert len(calls) == 2


def test_run_without_checkpoint_has_no_verdict(paths):
    paths.run_dir("empty").mkdir()

    status = CheckpointCatalog(ConsoleRuntime(paths)).status(paths.run_dir("empty"))

    assert status == {"has_checkpoint": False, "loadable": None, "reason": None, "mtime": None}


def test_runtime_model_loads_once_per_mtime_and_rejects_unknown_runs(paths):
    save(paths, "good")
    runtime = ConsoleRuntime(paths)

    first = runtime.model("good")
    assert runtime.model("good") is first
    assert not first.training

    for name in ("nope", "../good", None):
        try:
            runtime.model(name)
        except ValueError as error:
            assert "unknown run" in str(error)
        else:
            raise AssertionError("expected a ValueError")
