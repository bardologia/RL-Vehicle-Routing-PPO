import cProfile
import pstats

import pytest

from configuration.monitor import MonitorConfig
from tools.logger import Logger, NullLogger
from tools.resource_monitor import ResourceMonitor
from tools.tracker import NullTracker, Tracker


class FakeWriter:
    def __init__(self):
        self.scalars    = []
        self.histograms = []

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, value, step))

    def add_histogram(self, tag, values, step, bins="auto"):
        self.histograms.append((tag, step))

    def flush(self):
        pass

    def close(self):
        pass


class CapturingLogger:
    def __init__(self):
        self.warnings = []

    def warning(self, message):
        self.warnings.append(message)

    def __getattr__(self, name):
        return lambda *args, **kwargs: None


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


def build_monitor(config=None, logger=None, tracker=None):
    return ResourceMonitor(config or MonitorConfig(), logger=logger or NullLogger(), tracker=tracker or NullTracker())


def test_resource_monitor_sample_reports_core_metrics():
    metrics = build_monitor().sample()

    for key in ("ram_pct", "ram_used_gb", "ram_total_gb", "cpu_pct", "proc_cpu_pct", "proc_rss_gb", "swap_pct", "shm_pct", "vram_used_gb", "vram_pct"):
        assert key in metrics


def test_resource_monitor_publish_routes_to_system_tags_at_current_step():
    writer  = FakeWriter()
    tracker = Tracker(writer=writer)
    monitor = build_monitor(tracker=tracker)

    tracker.set_step(9)
    monitor._publish(monitor.sample())

    tags = tags_of(writer)
    assert tags
    assert all(tag.startswith("system/") for tag in tags)
    assert "system/ram_pct" in tags
    assert all(step == 9 for _, _, step in writer.scalars)


def test_resource_monitor_publish_disabled_when_tb_off():
    writer  = FakeWriter()
    tracker = Tracker(writer=writer)
    monitor = build_monitor(config=MonitorConfig(log_to_tensorboard=False), tracker=tracker)

    monitor._publish(monitor.sample())

    assert writer.scalars == []


def test_resource_monitor_peak_publish_uses_scope():
    writer  = FakeWriter()
    tracker = Tracker(writer=writer)
    monitor = build_monitor(tracker=tracker)

    monitor.sample()
    monitor._publish_peaks()

    assert any(tag.startswith("system/peak/") for tag in tags_of(writer))


def test_resource_monitor_warns_when_threshold_exceeded():
    logger  = CapturingLogger()
    monitor = build_monitor(config=MonitorConfig(warn_ram_pct=0.0, warn_cooldown_sec=0.0), logger=logger)

    monitor.sample()

    assert any("RAM" in message for message in logger.warnings)


def test_resource_monitor_disabled_config_does_not_start_thread():
    monitor = build_monitor(config=MonitorConfig(enabled=False))

    monitor.start()
    assert monitor._thread is None
    monitor.stop()


def test_resource_monitor_start_stop_lifecycle_spawns_and_joins_thread():
    monitor = build_monitor(config=MonitorConfig(poll_interval_sec=0.01))

    monitor.start()
    assert monitor._thread is not None
    assert monitor._thread.is_alive()

    monitor.stop()
    assert monitor._thread is None


def _profile_stats():
    def work():
        return sum(index * index for index in range(2000))

    profiler = cProfile.Profile()
    profiler.enable()
    work()
    profiler.disable()

    stats        = pstats.Stats(profiler)
    stats.stream = None
    return stats


def test_logger_save_profiler_results_writes_markdown_table(tmp_path):
    logger = Logger(log_dir=str(tmp_path), name="prof")
    output = tmp_path / "profile.md"

    logger.save_profiler_results(_profile_stats(), str(output))
    logger.close()

    text = output.read_text(encoding="utf-8")
    assert text.startswith("# Profiler Results")
    assert "| Function" in text
    assert "| Cumulative Time (s)" in text
    assert "Generated:" in text
