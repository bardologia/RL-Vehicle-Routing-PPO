from configuration.monitor import MonitorConfig
from tools.logger import NullLogger
from tools.resource_monitor import ResourceMonitor
from tools.tracker import NullTracker, Tracker
from tests.conftest import CapturingLogger, FakeWriter


def tags_of(writer):
    return [entry[0] for entry in writer.scalars]


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
