"""Tests for the CPU, RAM and disk watchdog guarding the machine.

Covers the sustain and clear-margin logic of alerts, the RAM kill switch with
its sustain, cooldown and memory-attribution guard, the auto-throttle fired by
a raised CPU or load alert, the auto-nuke switch with its arming, sustain,
attribution and cooldown guards, CPU percentage from counter deltas, the
attributed memory share, and the bounded event log.
"""
from __future__ import annotations

import time

import pytest

from politeness        import JobPoliteness
from proc_stats        import CpuCounters, MemInfo, ProcSweep
from resource_watchdog import ResourceWatchdog
from web_logger        import WebLogger

from tests.webui.conftest import StubPaths


class StubProcesses:
    """Process manager stand-in counting stop_all calls.

    Attributes:
        running : Number of jobs the next stop_all reports as terminated.
        calls   : How many times stop_all has been called.
    """

    def __init__(self, running: int = 2) -> None:
        """Starts with `running` jobs and no recorded calls."""
        self.running = running
        self.calls   = 0

    def stop_all(self) -> int:
        """Records the call and returns the number of jobs terminated."""
        self.calls  += 1
        stopped      = self.running
        self.running = 0
        return stopped


class StubThrottle:
    """Throttle stand-in counting how often it is applied.

    Attributes:
        calls : How many times apply has been called.
    """

    NICE = 19

    def __init__(self) -> None:
        """Starts with no recorded calls."""
        self.calls = 0

    def apply(self) -> dict:
        """Records the call and reports two lowered processes."""
        self.calls += 1
        return {"ok": True, "processes": 2, "renice": 0, "ionice": 0, "error": ""}


class StubNuke:
    """Process nuke stand-in counting how often it fires.

    Attributes:
        calls : How many times nuke has been called.
    """

    def __init__(self) -> None:
        """Starts with no recorded calls."""
        self.calls = 0

    def nuke(self) -> dict:
        """Records the call and reports one cancelled queue entry, three signalled and one killed process."""
        self.calls += 1
        return {"ok": True, "cancelled": 1, "signalled": 3, "killed": 1}


class SharedWatchdog(ResourceWatchdog):
    """Watchdog whose attributed memory share is a settable attribute instead of a /proc sweep.

    Attributes:
        share : Fraction of used memory attributed to this user, or None when it cannot be read.
    """

    share = 0.9

    def mine_memory_share(self) -> float | None:
        """Returns the configured share."""
        return self.share


@pytest.fixture
def watchdog(tmp_path):
    """Returns a watchdog wired to stub processes reporting two running jobs, a 0.9 memory share, auto-throttle off and auto-nuke disarmed."""
    paths = StubPaths(tmp_path)
    return SharedWatchdog(StubProcesses(), JobPoliteness(paths, WebLogger()), StubThrottle(), StubNuke(), WebLogger(), paths)


def _alert(watchdog: ResourceWatchdog, kind: str) -> dict | None:
    """Returns the active alert of the given kind, or None when it is not raised."""
    return next((entry for entry in watchdog.state()["active"] if entry["kind"] == kind), None)


def _breach(watchdog: ResourceWatchdog, kind: str, times: int, level: str = "warn", mine: bool = False) -> None:
    """Feeds `times` consecutive breaching samples of the given kind, level and attribution."""
    for _ in range(times):
        watchdog._track(kind, level, True, False, mine, f"{kind} is over the line")


def test_an_alert_needs_a_sustained_breach(watchdog):
    """An alert is raised only once the breach has lasted the configured sustain."""
    _breach(watchdog, "cpu", ResourceWatchdog.SUSTAIN["cpu"] - 1)
    assert _alert(watchdog, "cpu") is None

    _breach(watchdog, "cpu", 1)
    assert _alert(watchdog, "cpu")["message"] == "cpu is over the line"


def test_a_single_calm_sample_resets_the_streak(watchdog):
    """One non-breaching sample resets the streak, so the alert never fires."""
    _breach(watchdog, "cpu", ResourceWatchdog.SUSTAIN["cpu"] - 1)
    watchdog._track("cpu", "warn", False, False, False, "calm")
    _breach(watchdog, "cpu", ResourceWatchdog.SUSTAIN["cpu"] - 1)

    assert _alert(watchdog, "cpu") is None


def test_an_active_alert_is_updated_in_place(watchdog):
    """A raised alert has its level and message updated while keeping its original timestamp."""
    _breach(watchdog, "ram", ResourceWatchdog.SUSTAIN["ram"])
    since = _alert(watchdog, "ram")["since"]

    watchdog._track("ram", "danger", True, False, False, "ram is critical")
    updated = _alert(watchdog, "ram")

    assert len(watchdog.state()["active"]) == 1
    assert updated["level"]   == "danger"
    assert updated["message"] == "ram is critical"
    assert updated["since"]   == since


def test_an_alert_survives_until_the_clear_margin_is_crossed(watchdog):
    """An alert clears only when the sample falls below the clear margin."""
    _breach(watchdog, "ram", ResourceWatchdog.SUSTAIN["ram"])

    watchdog._track("ram", "warn", False, False, False, "still elevated")
    assert _alert(watchdog, "ram") is not None

    watchdog._track("ram", "warn", False, True, False, "back to normal")
    assert _alert(watchdog, "ram") is None


def test_the_ram_kill_needs_the_sustain_before_it_fires(watchdog):
    """The RAM kill fires only after the breach has lasted the ram_kill sustain."""
    watchdog._check_kill(ResourceWatchdog.RAM_KILL + 1.0)

    assert watchdog.processes.calls == 0
    assert watchdog.state()["events"] == []

    watchdog._check_kill(ResourceWatchdog.RAM_KILL + 1.0)
    events = watchdog.state()["events"]

    assert watchdog.processes.calls == 1
    assert len(events)       == 1
    assert events[0]["kind"] == "ram_kill"
    assert "terminated 2 running job(s)" in events[0]["message"]


def test_ram_dropping_below_the_kill_line_resets_the_streak(watchdog):
    """A sample below the kill line resets the kill streak."""
    watchdog._check_kill(ResourceWatchdog.RAM_KILL + 1.0)
    watchdog._check_kill(ResourceWatchdog.RAM_KILL - 10.0)
    watchdog._check_kill(ResourceWatchdog.RAM_KILL + 1.0)

    assert watchdog.processes.calls == 0


def test_a_second_kill_waits_for_the_cooldown(watchdog):
    """A second kill fires only after the cooldown has elapsed."""
    for _ in range(ResourceWatchdog.SUSTAIN["ram_kill"] + 4):
        watchdog._check_kill(ResourceWatchdog.RAM_KILL + 1.0)

    assert watchdog.processes.calls == 1

    watchdog.last_kill = time.monotonic() - ResourceWatchdog.KILL_COOLDOWN - 1.0
    watchdog._check_kill(ResourceWatchdog.RAM_KILL + 1.0)

    assert watchdog.processes.calls == 2


def test_a_kill_with_nothing_running_still_records_the_event(watchdog):
    """A kill with no running jobs still records an event saying so."""
    watchdog.processes.running = 0

    for _ in range(ResourceWatchdog.SUSTAIN["ram_kill"]):
        watchdog._check_kill(ResourceWatchdog.RAM_KILL + 1.0)

    assert "no console jobs were running" in watchdog.state()["events"][0]["message"]


def test_a_neighbours_ram_never_kills_this_users_jobs(watchdog):
    """Memory owned mostly by other users raises a foreign-memory alert instead of killing."""
    watchdog.share = ResourceWatchdog.MINE_SHARE - 0.2

    for _ in range(ResourceWatchdog.SUSTAIN["ram_kill"] + 3):
        watchdog._check_kill(ResourceWatchdog.RAM_KILL + 5.0)

    assert watchdog.processes.calls == 0
    assert watchdog.state()["events"] == []
    assert "attributed to this user" in _alert(watchdog, "ram_foreign")["message"]


def test_the_kill_fires_once_this_user_owns_the_memory(watchdog):
    """The kill fires and the foreign-memory alert clears once this user owns the memory."""
    watchdog.share = ResourceWatchdog.MINE_SHARE - 0.2
    for _ in range(ResourceWatchdog.SUSTAIN["ram_kill"]):
        watchdog._check_kill(ResourceWatchdog.RAM_KILL + 5.0)

    watchdog.share = ResourceWatchdog.MINE_SHARE + 0.1
    watchdog._check_kill(ResourceWatchdog.RAM_KILL + 5.0)

    assert watchdog.processes.calls == 1
    assert _alert(watchdog, "ram_foreign") is None
    assert "attributed to this user" in watchdog.state()["events"][0]["message"]


def test_an_unsampled_attribution_holds_the_kill_back(watchdog):
    """A missing attribution sample holds the kill back and is reported in the alert."""
    watchdog.share = None

    for _ in range(ResourceWatchdog.SUSTAIN["ram_kill"] + 1):
        watchdog._check_kill(ResourceWatchdog.RAM_KILL + 5.0)

    assert watchdog.processes.calls == 0
    assert "no attribution sample" in _alert(watchdog, "ram_foreign")["message"]


def test_calm_ram_clears_the_foreign_memory_alert(watchdog):
    """RAM falling below the kill line clears the foreign-memory alert."""
    watchdog.share = ResourceWatchdog.MINE_SHARE - 0.2
    for _ in range(ResourceWatchdog.SUSTAIN["ram_kill"]):
        watchdog._check_kill(ResourceWatchdog.RAM_KILL + 5.0)

    watchdog._check_kill(ResourceWatchdog.RAM_KILL - 10.0)

    assert _alert(watchdog, "ram_foreign") is None


def test_cpu_percent_is_a_delta_between_two_samples(watchdog, monkeypatch):
    """CPU percentage is None on the first sample and the busy/total delta afterwards."""
    samples = [{"cpu": (100, 200)}, {"cpu": (150, 300)}]
    monkeypatch.setattr(CpuCounters, "read", staticmethod(lambda: samples.pop(0)))

    assert watchdog._cpu_percent() is None
    assert watchdog._cpu_percent() == pytest.approx(50.0)


def test_cpu_percent_ignores_a_stalled_counter(watchdog, monkeypatch):
    """Identical counters between samples yield no CPU percentage."""
    monkeypatch.setattr(CpuCounters, "read", staticmethod(lambda: {"cpu": (100, 200)}))

    assert watchdog._cpu_percent() is None
    assert watchdog._cpu_percent() is None


def test_evaluate_escalates_the_ram_alert_and_kills(watchdog, monkeypatch):
    """A sustained RAM breach warns, then escalates to danger and triggers the kill."""
    monkeypatch.setattr(ResourceWatchdog, "_cpu_percent", lambda self: 10.0)
    monkeypatch.setattr(ResourceWatchdog, "_ram_percent", lambda self: ResourceWatchdog.RAM_WARN + 1.0)

    for _ in range(ResourceWatchdog.SUSTAIN["ram"]):
        watchdog._evaluate()

    assert _alert(watchdog, "ram")["level"] == "warn"
    assert _alert(watchdog, "cpu") is None
    assert watchdog.processes.calls == 0

    monkeypatch.setattr(ResourceWatchdog, "_ram_percent", lambda self: ResourceWatchdog.RAM_KILL + 1.0)
    for _ in range(ResourceWatchdog.SUSTAIN["ram_kill"]):
        watchdog._evaluate()

    assert _alert(watchdog, "ram")["level"] == "danger"
    assert watchdog.processes.calls == 1


def test_state_reports_the_limits_it_enforces(watchdog):
    """State reports the watchdog as unarmed and echoes the thresholds it enforces."""
    limits = watchdog.state()["limits"]

    assert watchdog.state()["armed"] is False
    assert limits["ram_kill"]  == ResourceWatchdog.RAM_KILL
    assert limits["ram_warn"]  == ResourceWatchdog.RAM_WARN
    assert limits["cpu_alert"]  == ResourceWatchdog.CPU_ALERT
    assert limits["cooldown"]   == ResourceWatchdog.KILL_COOLDOWN
    assert limits["mine_share"] == ResourceWatchdog.MINE_SHARE


def test_the_event_log_is_bounded(watchdog):
    """The event log keeps only the most recent 20 entries."""
    for _ in range(60):
        watchdog.last_kill = 0.0
        watchdog.streaks["ram_kill"] = ResourceWatchdog.SUSTAIN["ram_kill"] - 1
        watchdog._check_kill(ResourceWatchdog.RAM_KILL + 1.0)

    assert len(watchdog.state()["events"]) == 20


def test_a_raised_cpu_alert_throttles_the_jobs_once_when_auto_throttle_is_on(watchdog):
    """A newly raised CPU alert applies the throttle once and records the event; a refresh stays silent."""
    watchdog.politeness.configure({"auto_throttle": True})
    _breach(watchdog, "cpu", ResourceWatchdog.SUSTAIN["cpu"] + 2)

    assert watchdog.throttle.calls == 1
    assert "auto-throttle on the cpu alert: 2 job process(es)" in watchdog.state()["events"][0]["message"]


def test_auto_throttle_off_leaves_the_jobs_alone(watchdog):
    """With auto-throttle off a raised CPU or load alert applies no throttle."""
    _breach(watchdog, "cpu", ResourceWatchdog.SUSTAIN["cpu"])
    _breach(watchdog, "load", ResourceWatchdog.SUSTAIN["load"])

    assert watchdog.throttle.calls == 0
    assert watchdog.state()["events"] == []


def test_a_ram_alert_never_throttles(watchdog):
    """Only CPU and load alerts throttle; a RAM alert does not."""
    watchdog.politeness.configure({"auto_throttle": True})
    _breach(watchdog, "ram", ResourceWatchdog.SUSTAIN["ram"])

    assert watchdog.throttle.calls == 0


def test_the_attributed_share_divides_this_users_memory_by_the_used_memory(tmp_path, monkeypatch):
    """The memory share sums the attributed memory of this user's processes over the host's used memory."""
    paths    = StubPaths(tmp_path)
    watchdog = ResourceWatchdog(StubProcesses(), JobPoliteness(paths, WebLogger()), StubThrottle(), StubNuke(), WebLogger(), paths)
    rows     = [{"uid": watchdog.uid, "attributed": 300}, {"uid": watchdog.uid, "attributed": None}, {"uid": watchdog.uid + 1, "attributed": 500}]

    monkeypatch.setattr(MemInfo, "fields", staticmethod(lambda: {"MemTotal": 2000, "MemAvailable": 1000}))
    monkeypatch.setattr(ProcSweep, "rows", staticmethod(lambda: rows))

    assert watchdog.mine_memory_share() == pytest.approx(0.3)


def test_low_disk_space_raises_a_sustained_alert(watchdog, monkeypatch):
    """Free space under the warn line raises a disk alert once sustained."""
    monkeypatch.setattr(watchdog, "_disk_free", lambda: (ResourceWatchdog.DISK_WARN - 2.0, 40.0))
    monkeypatch.setattr(watchdog, "_cpu_percent", lambda: 0.0)
    monkeypatch.setattr(watchdog, "_ram_percent", lambda: 0.0)

    for _ in range(ResourceWatchdog.SUSTAIN["disk"]):
        watchdog._evaluate()

    alert = _alert(watchdog, "disk")
    assert alert is not None
    assert alert["level"] == "warn"
    assert "disk almost full" in alert["message"]


def test_critically_low_disk_space_is_a_danger_alert(watchdog, monkeypatch):
    """Free space under the danger line raises the disk alert at danger level."""
    monkeypatch.setattr(watchdog, "_disk_free", lambda: (ResourceWatchdog.DISK_DANGER - 1.0, 8.0))
    monkeypatch.setattr(watchdog, "_cpu_percent", lambda: 0.0)
    monkeypatch.setattr(watchdog, "_ram_percent", lambda: 0.0)

    for _ in range(ResourceWatchdog.SUSTAIN["disk"]):
        watchdog._evaluate()

    assert _alert(watchdog, "disk")["level"] == "danger"


def test_ample_disk_space_raises_nothing(watchdog, monkeypatch):
    """Free space above the warn line never raises the disk alert."""
    monkeypatch.setattr(watchdog, "_disk_free", lambda: (50.0, 800.0))
    monkeypatch.setattr(watchdog, "_cpu_percent", lambda: 0.0)
    monkeypatch.setattr(watchdog, "_ram_percent", lambda: 0.0)

    for _ in range(4):
        watchdog._evaluate()

    assert _alert(watchdog, "disk") is None


def _sample_nuke(watchdog: ResourceWatchdog, times: int) -> None:
    """Runs the auto-nuke check `times` times against the standing alerts."""
    for _ in range(times):
        watchdog._check_nuke()


def test_arming_auto_nuke_is_reported_in_the_state(watchdog):
    """Arming and disarming flip the auto-nuke flag the state reports, and disarming resets the streak."""
    assert watchdog.state()["auto_nuke"] is False

    assert watchdog.arm(True)["auto_nuke"] is True
    watchdog.nuke_streak = 3

    assert watchdog.arm(False)["auto_nuke"] is False
    assert watchdog.nuke_streak == 0


def test_an_armed_auto_nuke_fires_after_the_sustain_on_a_danger_alert_of_this_user(watchdog):
    """A danger alert this user causes fires the nuke once after NUKE_SUSTAIN samples and records the event."""
    watchdog.arm(True)
    _breach(watchdog, "ram", ResourceWatchdog.SUSTAIN["ram"], "danger", True)

    _sample_nuke(watchdog, ResourceWatchdog.NUKE_SUSTAIN - 1)
    assert watchdog.nuke.calls == 0

    _sample_nuke(watchdog, 1)
    events = watchdog.state()["events"]

    assert watchdog.nuke.calls == 1
    assert events[-1]["kind"]  == "auto_nuke"
    assert events[-1]["level"] == "danger"
    assert "cancelled 1 queued, terminated 3, force-killed 1" in events[-1]["message"]


def test_a_disarmed_auto_nuke_never_fires(watchdog):
    """With auto-nuke disarmed a standing danger alert of this user's never fires the nuke."""
    _breach(watchdog, "ram", ResourceWatchdog.SUSTAIN["ram"], "danger", True)
    _sample_nuke(watchdog, ResourceWatchdog.NUKE_SUSTAIN * 3)

    assert watchdog.nuke.calls == 0
    assert watchdog.nuke_streak == 0


def test_a_danger_alert_this_user_does_not_cause_never_fires_auto_nuke(watchdog):
    """A danger alert not attributed to this user, such as a full disk, never fires the nuke."""
    watchdog.arm(True)
    _breach(watchdog, "disk", ResourceWatchdog.SUSTAIN["disk"], "danger", False)
    _sample_nuke(watchdog, ResourceWatchdog.NUKE_SUSTAIN * 2)

    assert watchdog.nuke.calls == 0


def test_a_warn_alert_of_this_user_never_fires_auto_nuke(watchdog):
    """A warn-level alert this user causes never fires the nuke."""
    watchdog.arm(True)
    _breach(watchdog, "ram", ResourceWatchdog.SUSTAIN["ram"], "warn", True)
    _sample_nuke(watchdog, ResourceWatchdog.NUKE_SUSTAIN * 2)

    assert watchdog.nuke.calls == 0


def test_a_cleared_alert_resets_the_auto_nuke_streak(watchdog):
    """A sample without a severe alert resets the streak, so the nuke needs a fresh sustained run."""
    watchdog.arm(True)
    _breach(watchdog, "ram", ResourceWatchdog.SUSTAIN["ram"], "danger", True)
    _sample_nuke(watchdog, ResourceWatchdog.NUKE_SUSTAIN - 1)

    watchdog._track("ram", "warn", False, True, False, "back to normal")
    _sample_nuke(watchdog, 1)
    _breach(watchdog, "ram", ResourceWatchdog.SUSTAIN["ram"], "danger", True)
    _sample_nuke(watchdog, ResourceWatchdog.NUKE_SUSTAIN - 1)

    assert watchdog.nuke.calls == 0


def test_a_second_auto_nuke_waits_for_its_cooldown(watchdog):
    """After firing, the nuke fires again only once its cooldown has elapsed."""
    watchdog.arm(True)
    _breach(watchdog, "ram", ResourceWatchdog.SUSTAIN["ram"], "danger", True)
    _sample_nuke(watchdog, ResourceWatchdog.NUKE_SUSTAIN * 3)

    assert watchdog.nuke.calls == 1

    watchdog.last_nuke = time.monotonic() - ResourceWatchdog.NUKE_COOLDOWN - 1.0
    _sample_nuke(watchdog, ResourceWatchdog.NUKE_SUSTAIN)

    assert watchdog.nuke.calls == 2


def test_evaluate_attributes_the_ram_danger_alert_and_fires_an_armed_auto_nuke(watchdog, monkeypatch):
    """RAM at the kill limit with this user's share marks the RAM alert as this user's, and an armed auto-nuke fires once it persists."""
    monkeypatch.setattr(watchdog, "_cpu_percent", lambda: 0.0)
    monkeypatch.setattr(watchdog, "_ram_percent", lambda: ResourceWatchdog.RAM_KILL + 2.0)
    monkeypatch.setattr(watchdog, "_disk_free", lambda: (50.0, 800.0))
    watchdog.arm(True)

    for _ in range(ResourceWatchdog.SUSTAIN["ram"] + ResourceWatchdog.NUKE_SUSTAIN - 1):
        watchdog._evaluate()

    assert _alert(watchdog, "ram")["mine"] is True
    assert watchdog.nuke.calls == 1


def test_evaluate_leaves_a_neighbours_ram_alert_unattributed(watchdog, monkeypatch):
    """RAM at the kill limit held mostly by other users leaves the RAM alert unattributed and auto-nuke idle."""
    monkeypatch.setattr(watchdog, "_cpu_percent", lambda: 0.0)
    monkeypatch.setattr(watchdog, "_ram_percent", lambda: ResourceWatchdog.RAM_KILL + 2.0)
    monkeypatch.setattr(watchdog, "_disk_free", lambda: (50.0, 800.0))
    watchdog.share = ResourceWatchdog.MINE_SHARE - 0.2
    watchdog.arm(True)

    for _ in range(ResourceWatchdog.SUSTAIN["ram"] + ResourceWatchdog.NUKE_SUSTAIN * 2):
        watchdog._evaluate()

    assert _alert(watchdog, "ram")["mine"] is False
    assert watchdog.nuke.calls == 0
