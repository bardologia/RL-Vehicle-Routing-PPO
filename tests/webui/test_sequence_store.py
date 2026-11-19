"""Tests for the on-disk sequence store.

Covers creation with the isolated and detached defaults, saving a validated
block graph with fresh block ids and normalised fields, the refusal of unknown
scripts, dangling and self edges, duplicate ids and cycles, the decision axes a
block declares and the verdicts they freeze downstream, the topological order,
newest-first listing with adoption of a misnamed file, and deletion.
"""

from __future__ import annotations

import json

import pytest

from sequence_store import SequenceStore
from web_logger     import WebLogger

from tests.webui.conftest import LEAVES, StubPaths, decision_engine


@pytest.fixture
def store(tmp_path):
    """Returns a store over a temporary root whose main directory holds train and evaluate."""
    (tmp_path / "main").mkdir()
    for key in ("train", "evaluate"):
        (tmp_path / "main" / f"{key}.py").write_text("")
    return SequenceStore(StubPaths(tmp_path), WebLogger(), decision_engine())


def _block(block_id: str, script: str = "train", predecessors: tuple = (), **extra) -> dict:
    """Returns one block payload."""
    return {"block_id": block_id, "script": script, "predecessors": list(predecessors), **extra}


def test_a_new_sequence_isolates_and_detaches_by_default(store):
    """A created sequence is empty, isolated and detached, with a name or a stamp-based one."""
    entry = store.create("  ladder  ")["entry"]

    assert entry["name"] == "ladder"
    assert (entry["blocks"], entry["isolate"], entry["detach"]) == ([], True, True)
    assert store.create("")["entry"]["name"].startswith("sequence ")


def test_a_saved_graph_is_normalised(store):
    """Blocks are normalised: a missing id is assigned, overrides stringified, flags booleans."""
    sequence_id = store.create("graph")["entry"]["sequence_id"]
    result      = store.save(sequence_id, {"name": "graph", "detach": False, "isolate": True, "blocks": [
        {"script": "train", "overrides": {"training.num_updates": 4}, "checkpoint": 1},
        _block("b2", "evaluate"),
    ]})

    blocks = result["entry"]["blocks"]
    assert result["ok"] is True
    assert len(blocks[0]["block_id"]) == 8
    assert blocks[0]["overrides"] == {"training.num_updates": "4"}
    assert (blocks[0]["checkpoint"], blocks[0]["skip"], blocks[0]["title"]) == (True, False, "train")
    assert result["entry"]["detach"] is False


@pytest.mark.parametrize("blocks, error", [
    ([_block("a", "nope")],                               "unknown script 'nope'"),
    ([_block("a", predecessors=("ghost",))],               "links to missing block(s): ghost"),
    ([_block("a", predecessors=("a",))],                   "links to itself"),
    ([_block("a"), _block("a")],                           "duplicate block ids"),
    ([_block("a", predecessors=("b",)), _block("b", predecessors=("a",))], "the sequence has a cycle"),
])
def test_an_invalid_graph_is_refused(store, blocks, error):
    """Unknown scripts, dangling or self edges, duplicate ids and cycles are refused by name."""
    sequence_id = store.create("bad")["entry"]["sequence_id"]

    result = store.save(sequence_id, {"blocks": blocks})

    assert result["ok"] is False and error in result["error"]


def test_the_order_is_topological(store):
    """The order lists every block after its predecessors, ties sorted by id."""
    blocks = [_block("c", predecessors=("a", "b")), _block("b", predecessors=("a",)), _block("a")]

    assert store.order(blocks) == ["a", "b", "c"]


def test_listing_is_newest_first_and_adopts_a_misnamed_file(store, tmp_path):
    """The listing sorts by update time and renames a definition dropped in under any name."""
    first  = store.create("first")["entry"]
    second = store.create("second")["entry"]
    stray  = {**first, "sequence_id": "abcdefabcdef", "updated_at": "2099-01-01T00:00:00"}
    (store.directory / "dropped.json").write_text(json.dumps(stray))

    listed = [entry["sequence_id"] for entry in store.list()["sequences"]]

    assert listed[0] == "abcdefabcdef"
    assert set(listed) == {first["sequence_id"], second["sequence_id"], "abcdefabcdef"}
    assert (store.directory / "abcdefabcdef.json").is_file()


def test_delete_and_malformed_ids(store):
    """Deleting removes the file; malformed and unknown ids are reported as not found."""
    sequence_id = store.create("gone")["entry"]["sequence_id"]

    assert store.delete(sequence_id) == {"ok": True}
    assert store.get(sequence_id) is None
    assert store.delete("../etc") == {"ok": False, "error": "sequence not found"}
    assert store.save("nope", {})["ok"] is False


def _decided(store: SequenceStore, blocks: list) -> dict:
    """Saves the blocks into a new sequence and returns the stored blocks keyed by id."""
    sequence_id = store.create("decisions")["entry"]["sequence_id"]
    result      = store.save(sequence_id, {"name": "decisions", "blocks": blocks})

    assert result["ok"], result.get("error")
    return {block["block_id"]: block for block in result["entry"]["blocks"]}


def test_save_discards_field_notes_from_the_payload(store):
    """Field notes are never taken from the caller: they are rewritten from the declared decisions."""
    saved = _decided(store, [_block("a1", field_notes={"training.device": "frozen"})])

    assert saved["a1"]["field_notes"] == {}
    assert saved["a1"]["decisions"] == []


def test_a_verdict_freezes_the_blocks_after_the_deciding_one(store):
    """A decided axis is stamped as the verdict downstream and as the pre-verdict value on the decider."""
    decisions = [{"path": "training.num_updates", "options": ["50", "200"], "waiting": "50", "choice": "200"}]
    saved     = _decided(store, [_block("a1", decisions=decisions), _block("b2", predecessors=("a1",)), _block("c3", predecessors=("b2",))])

    assert saved["a1"]["overrides"] == {"training.num_updates": "50"}
    assert saved["a1"]["field_notes"] == {"training.num_updates": "waiting"}
    assert saved["b2"]["overrides"] == {"training.num_updates": "200"}
    assert saved["c3"]["field_notes"] == {"training.num_updates": "frozen"}


def test_an_undecided_axis_leaves_every_block_waiting(store):
    """Without a verdict, blocks after the deciding one keep the pre-verdict value."""
    decisions = [{"path": "training.num_updates", "options": ["50", "200"], "waiting": "50", "choice": ""}]
    saved     = _decided(store, [_block("a1", decisions=decisions), _block("b2", predecessors=("a1",))])

    assert saved["b2"]["overrides"] == {"training.num_updates": "50"}
    assert saved["b2"]["field_notes"] == {"training.num_updates": "waiting"}


def test_a_value_matching_the_entry_default_is_stamped_without_an_override(store):
    """A stamped value equal to the entry's own default leaves the note but no override."""
    decisions = [{"path": "training.num_updates", "options": ["50", "100"], "waiting": "50", "choice": "100"}]
    saved     = _decided(store, [_block("a1", decisions=decisions), _block("b2", predecessors=("a1",))])

    assert saved["b2"]["overrides"] == {}
    assert saved["b2"]["field_notes"] == {"training.num_updates": "frozen"}


def test_a_decision_skips_blocks_whose_entry_does_not_expose_the_path(store):
    """Only blocks whose entry point exposes the axis are stamped with it."""
    decisions = [{"path": "training.max_disruptions_per_episode", "options": ["3", "5"], "waiting": "5", "choice": "3"}]
    saved     = _decided(store, [_block("a1", decisions=decisions), _block("b2", "evaluate", predecessors=("a1",)), _block("c3", predecessors=("b2",))])

    assert (saved["b2"]["overrides"], saved["b2"]["field_notes"]) == ({}, {})
    assert saved["c3"]["overrides"] == {"training.max_disruptions_per_episode": "3"}


def test_an_empty_pre_verdict_value_stamps_nothing(store):
    """A declared axis whose pre-verdict value is still blank leaves the exposing blocks' overrides untouched."""
    decisions = [{"path": "training.device", "options": ["cpu", "cuda"], "waiting": "", "choice": ""}]
    saved     = _decided(store, [_block("a1", decisions=decisions), _block("b2", "evaluate", predecessors=("a1",), overrides={"training.device": "cpu"})])

    assert saved["a1"]["field_notes"] == {}
    assert saved["b2"]["field_notes"] == {}
    assert saved["b2"]["overrides"] == {"training.device": "cpu"}


def test_the_later_decision_on_the_same_axis_wins_downstream(store):
    """When two blocks decide one axis, the topologically later verdict governs the blocks after it."""
    first  = [{"path": "training.num_updates", "options": ["50", "200"], "waiting": "50", "choice": "200"}]
    second = [{"path": "training.num_updates", "options": ["200", "400"], "waiting": "200", "choice": "400"}]
    saved  = _decided(store, [_block("a1", decisions=first), _block("b2", predecessors=("a1",), decisions=second), _block("c3", predecessors=("b2",))])

    assert saved["b2"]["overrides"] == {"training.num_updates": "200"}
    assert saved["c3"]["overrides"] == {"training.num_updates": "400"}


def test_save_refuses_a_decision_it_cannot_resolve_the_entries_for(store):
    """A decision on a sequence holding an unresolvable entry point is refused by name."""
    decisions   = [{"path": "training.device", "options": ["cpu"], "waiting": "cpu", "choice": ""}]
    narrow      = SequenceStore(store.paths, WebLogger(), decision_engine({"train": LEAVES["train"]}))
    sequence_id = narrow.create("narrow")["entry"]["sequence_id"]

    result = narrow.save(sequence_id, {"blocks": [_block("a1", decisions=decisions), _block("b2", "evaluate", predecessors=("a1",))]})

    assert result["ok"] is False and "evaluate" in result["error"]


def test_decide_records_a_verdict_and_restamps_the_sequence(store):
    """Recording a verdict through the store freezes it into the blocks after the deciding one."""
    sequence_id = store.create("decide")["entry"]["sequence_id"]
    decisions   = [{"path": "training.num_updates", "options": ["50", "200"], "waiting": "50", "choice": ""}]
    store.save(sequence_id, {"blocks": [_block("a1", decisions=decisions), _block("b2", predecessors=("a1",))]})

    result = store.decide(sequence_id, "a1", "training.num_updates", "200")

    saved = {block["block_id"]: block for block in result["entry"]["blocks"]}
    assert result["ok"] is True
    assert saved["a1"]["decisions"][0]["choice"] == "200"
    assert saved["b2"]["overrides"] == {"training.num_updates": "200"}


def test_decide_adds_a_verdict_outside_the_options_and_reopens_on_an_empty_choice(store):
    """A verdict no option anticipated joins the options, and clearing it puts the blocks back on the pre-verdict value."""
    sequence_id = store.create("decide")["entry"]["sequence_id"]
    decisions   = [{"path": "training.num_updates", "options": ["50"], "waiting": "50", "choice": ""}]
    store.save(sequence_id, {"blocks": [_block("a1", decisions=decisions), _block("b2", predecessors=("a1",))]})

    decided  = store.decide(sequence_id, "a1", "training.num_updates", "300")
    reopened = store.decide(sequence_id, "a1", "training.num_updates", "")

    assert decided["entry"]["blocks"][0]["decisions"][0]["options"] == ["50", "300"]
    assert reopened["entry"]["blocks"][1]["overrides"] == {"training.num_updates": "50"}
    assert reopened["entry"]["blocks"][1]["field_notes"] == {"training.num_updates": "waiting"}


def test_decide_refuses_an_axis_the_block_does_not_decide(store):
    """Recording a verdict for an axis the block never declared is refused."""
    sequence_id = store.create("decide")["entry"]["sequence_id"]
    store.save(sequence_id, {"blocks": [_block("a1")]})

    result = store.decide(sequence_id, "a1", "training.num_updates", "7")

    assert result["ok"] is False and "does not decide" in result["error"]
