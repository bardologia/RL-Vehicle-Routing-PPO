"""Tests for the on-disk sequence store.

Covers creation with the isolated and detached defaults, saving a validated
block graph with fresh block ids and normalised fields, the refusal of unknown
scripts, dangling and self edges, duplicate ids and cycles, the topological
order, newest-first listing with adoption of a misnamed file, and deletion.
"""

from __future__ import annotations

import json

import pytest

from sequence_store import SequenceStore
from web_logger     import WebLogger

from tests.webui.conftest import StubPaths


@pytest.fixture
def store(tmp_path):
    """Returns a store over a temporary root whose main directory holds train and evaluate."""
    (tmp_path / "main").mkdir()
    for key in ("train", "evaluate"):
        (tmp_path / "main" / f"{key}.py").write_text("")
    return SequenceStore(StubPaths(tmp_path), WebLogger())


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
