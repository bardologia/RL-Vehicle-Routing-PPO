from tools.tracker import NullTracker, Tracker
from tests.conftest import FakeWriter


def tags_of(writer):
    return [entry[0] for entry in writer.scalars]


def test_tracker_active_property():
    assert Tracker(writer=FakeWriter()).active is True
    assert Tracker().active is False
    assert NullTracker().active is False


def test_tracker_set_step_advance_and_current_step():
    tracker = Tracker(writer=FakeWriter())

    tracker.set_step(10)
    assert tracker.current_step == 10
    assert tracker.advance() == 11
    assert tracker.advance(4) == 15
    assert tracker.current_step == 15


def test_tracker_uses_resolved_step_when_none_passed():
    writer  = FakeWriter()
    tracker = Tracker(writer=writer)

    tracker.set_step(42)
    tracker.log_scalar("loss", 1.5)

    assert writer.scalars[-1] == ("loss", 1.5, 42)


def test_tracker_explicit_step_overrides_resolved():
    writer  = FakeWriter()
    tracker = Tracker(writer=writer)

    tracker.set_step(42)
    tracker.log_scalar("loss", 1.5, step=7)

    assert writer.scalars[-1] == ("loss", 1.5, 7)


def test_tracker_scope_prefixes_tags_and_pops_cleanly():
    writer  = FakeWriter()
    tracker = Tracker(writer=writer)

    with tracker.scope("system"):
        tracker.log_scalar("cpu", 1.0, step=0)
        with tracker.scope("gpu0"):
            tracker.log_scalar("vram", 2.0, step=0)

    tracker.log_scalar("plain", 3.0, step=0)

    assert ("system/cpu", 1.0, 0) in writer.scalars
    assert ("system/gpu0/vram", 2.0, 0) in writer.scalars
    assert ("plain", 3.0, 0) in writer.scalars
    assert tracker._scopes == []


def test_tracker_log_metrics_skips_non_numeric_and_keeps_prefix():
    writer  = FakeWriter()
    tracker = Tracker(writer=writer)

    tracker.log_metrics("m", {"ok": 1.0, "bad": "x"}, step=3)

    tags = tags_of(writer)
    assert "m/ok" in tags
    assert "m/bad" not in tags


def test_tracker_histogram_and_null_tracker_are_safe():
    writer  = FakeWriter()
    tracker = Tracker(writer=writer)

    tracker.log_histogram("weights", [0.1, 0.2, 0.3], step=1)
    assert writer.histograms == [("weights", 1)]

    null = NullTracker()
    null.log_scalar("x", 1.0)
    null.log_metrics("m", {"a": 1.0})
    null.log_histogram("h", [1.0, 2.0])
    assert null.current_step == 0


def test_tracker_flush_and_close_delegate_to_writer():
    writer  = FakeWriter()
    tracker = Tracker(writer=writer)

    tracker.flush()
    tracker.close()

    assert tracker.active is True
