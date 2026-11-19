"""Tests for the sequence route that records a decision verdict.

Covers the decide route answering the re-stamped sequence through the router's
own table, and its refusal of an axis the block does not declare.
"""

from __future__ import annotations

import json

from routers.dispatch        import HttpExchange
from routers.sequence_router import SequenceRouter
from sequence_store          import SequenceStore
from web_logger              import WebLogger

from tests.webui.conftest import FakeHandler, StubPaths, decision_engine


class IdleRunner:
    """Sequence-runner stand-in reporting no live execution."""

    def states(self) -> dict:
        """Returns an empty state map."""
        return {}

    def state(self, sequence_id: str) -> None:
        """Returns no state for any sequence."""
        return None


def _router(tmp_path) -> tuple:
    """Returns a router over a store whose main directory holds train and evaluate, with one deciding sequence saved."""
    (tmp_path / "main").mkdir()
    for key in ("train", "evaluate"):
        (tmp_path / "main" / f"{key}.py").write_text("")

    store       = SequenceStore(StubPaths(tmp_path), WebLogger(), decision_engine())
    sequence_id = store.create("decide")["entry"]["sequence_id"]
    decisions   = [{"path": "training.num_updates", "options": ["50", "200"], "waiting": "50", "choice": ""}]
    store.save(sequence_id, {"blocks": [{"block_id": "a1", "script": "train", "decisions": decisions}, {"block_id": "b2", "script": "train", "predecessors": ["a1"]}]})

    return SequenceRouter(store, IdleRunner()), sequence_id


def _post(router: SequenceRouter, path: str, body: dict) -> FakeHandler:
    """Dispatches one JSON POST through the router's table and returns the captured handler."""
    handler  = FakeHandler("POST", path, json.dumps(body).encode("utf-8"), {"Content-Type": "application/json"})
    exchange = HttpExchange.of(handler)
    exchange.read_body()

    assert router.handle(exchange)
    return handler


def test_the_decide_route_freezes_the_verdict_downstream(tmp_path):
    """A verdict posted to the decide route is answered with the blocks after the decider frozen on it."""
    router, sequence_id = _router(tmp_path)

    handler = _post(router, f"/api/sequences/{sequence_id}/decide", {"block_id": "a1", "path": "training.num_updates", "choice": "200"})
    blocks  = {block["block_id"]: block for block in handler.payload()["entry"]["blocks"]}

    assert handler.status == 200
    assert blocks["b2"]["overrides"] == {"training.num_updates": "200"}
    assert blocks["b2"]["field_notes"] == {"training.num_updates": "frozen"}


def test_the_decide_route_refuses_an_undeclared_axis(tmp_path):
    """A verdict for an axis the block never declared is answered as a 400 naming the block."""
    router, sequence_id = _router(tmp_path)

    handler = _post(router, f"/api/sequences/{sequence_id}/decide", {"block_id": "b2", "path": "training.num_updates", "choice": "200"})

    assert handler.status == 400
    assert "does not decide" in handler.payload()["error"]
