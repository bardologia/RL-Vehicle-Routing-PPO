"""Host resource watchdog that alerts on CPU, load, RAM and disk-space pressure.

Samples /proc on a fixed interval, raises and clears sustained alerts with
hysteresis, warns when the filesystem holding the repository runs out of free
space, lowers the priority of the console's jobs when the processor stays
saturated and auto-throttle is on, terminates every running console job when
memory pressure crosses the kill limit with a large enough share attributed to
this user, and, once auto-nuke is armed, fires the process nuke when a danger
alert this user causes persists.
"""

from __future__ import annotations

import os
import shutil
import threading
import time
from collections import deque
from datetime    import datetime

from politeness      import JobPoliteness, JobThrottle
from proc_stats      import CpuCounters, MemInfo, ProcSweep
from process_manager import ProcessManager, ProcessNuke
from web_logger      import WebLogger


class ResourceWatchdog:
    """Watches host CPU, load and RAM, raising alerts and killing jobs on sustained RAM pressure.

    Alerts must hold for a per-kind number of consecutive samples before they
    are raised, and clear only once the metric drops below the threshold by a
    margin. Jobs are terminated only when the host RAM crosses the kill limit
    and enough of that memory is attributed to this user. Each alert records
    whether this user causes it; a danger alert of this user's that holds for
    NUKE_SUSTAIN samples fires the process nuke when auto-nuke is armed.

    Attributes:
        processes   : Process manager whose running jobs are stopped on a kill.
        politeness  : Politeness settings holding the auto-throttle switch.
        throttle    : Throttle lowering the jobs' priority when a CPU or load alert is raised.
        nuke        : Process nuke that clears the queue and stops every job process when auto-nuke fires.
        logger      : Console logger for alert transitions and kills.
        paths       : Project paths whose repository root selects the watched filesystem.
        armed       : Whether the background sampling thread has been started.
        active      : Currently raised alerts, keyed by alert kind.
        events      : Recent kill, throttle and auto-nuke events, newest last, capped at 20.
        streaks     : Consecutive breached-sample counts per alert kind.
        prev_cpu    : Previous aggregate CPU counter pair, or None before the first sample.
        last_kill   : Monotonic timestamp of the last kill, used for the cooldown.
        auto_nuke   : Whether the automatic nuke is armed.
        nuke_streak : Consecutive samples with a standing danger alert this user causes.
        last_nuke   : Monotonic timestamp of the last auto-nuke, used for its cooldown.
    """

    INTERVAL      = 5.0
    CPU_ALERT     = 90.0
    LOAD_RATIO    = 1.0
    RAM_WARN      = 80.0
    RAM_KILL      = 90.0
    MINE_SHARE    = 0.5
    DISK_WARN     = 10.0
    DISK_DANGER   = 5.0
    CLEAR_MARGIN  = 5.0
    SUSTAIN       = {"cpu": 3, "load": 3, "ram": 2, "ram_kill": 2, "disk": 2}
    KILL_COOLDOWN = 60.0
    NUKE_SUSTAIN  = 6
    NUKE_COOLDOWN = 120.0

    THROTTLED_BY  = ("cpu", "load")

    def __init__(self, processes: ProcessManager, politeness: JobPoliteness, throttle: JobThrottle, nuke: ProcessNuke, logger: WebLogger, paths) -> None:
        """Stores the collaborators and initialises empty alert and streak state with auto-nuke disarmed.

        Args:
            processes  : Process manager whose jobs are stopped on a RAM kill.
            politeness : Politeness settings holding the auto-throttle switch.
            throttle   : Throttle applied when a CPU or load alert is raised with auto-throttle on.
            nuke       : Process nuke fired when auto-nuke is armed and a danger alert this user causes persists.
            logger     : Console logger for alert transitions.
            paths      : Project paths whose repository root selects the watched filesystem.
        """
        self.processes  = processes
        self.politeness = politeness
        self.throttle   = throttle
        self.nuke       = nuke
        self.logger     = logger
        self.paths      = paths
        self.uid        = os.getuid()
        self.lock       = threading.Lock()
        self.armed      = False
        self.active     = {}
        self.events     = deque(maxlen=20)
        self.streaks    = {key: 0 for key in self.SUSTAIN}
        self.prev_cpu   = None
        self.last_kill  = 0.0

        self.auto_nuke   = False
        self.nuke_streak = 0
        self.last_nuke   = 0.0

    def start(self) -> None:
        """Starts the background sampling thread and marks the watchdog armed."""
        worker = threading.Thread(target=self._watch, daemon=True)
        worker.start()
        self.armed = True
        self.logger.muted(f"resource watchdog armed (cpu {self.CPU_ALERT:.0f}%, ram warn {self.RAM_WARN:.0f}%, ram kill {self.RAM_KILL:.0f}% with at least {self.MINE_SHARE:.0%} of it this user's, disk warn below {self.DISK_WARN:.0f}% free)")

    def state(self) -> dict:
        """Returns the armed and auto-nuke flags, the configured limits, the active alerts and recent events."""
        limits = {"cpu_alert": self.CPU_ALERT, "load_ratio": self.LOAD_RATIO, "ram_warn": self.RAM_WARN, "ram_kill": self.RAM_KILL, "mine_share": self.MINE_SHARE, "disk_warn": self.DISK_WARN, "interval": self.INTERVAL, "cooldown": self.KILL_COOLDOWN, "nuke_after": self.NUKE_SUSTAIN * self.INTERVAL, "nuke_cooldown": self.NUKE_COOLDOWN}

        with self.lock:
            return {"armed": self.armed, "auto_nuke": self.auto_nuke, "limits": limits, "active": list(self.active.values()), "events": list(self.events)}

    def arm(self, value: bool) -> dict:
        """Arms or disarms the automatic nuke and returns the refreshed state.

        Args:
            value : True to arm, False to disarm and reset the streak.

        Returns:
            The watchdog state as returned by `state`.
        """
        with self.lock:
            self.auto_nuke = bool(value)
            if not self.auto_nuke:
                self.nuke_streak = 0
        self.logger.warning(f"auto-nuke {'ARMED' if self.auto_nuke else 'disarmed'} (fires after {self.NUKE_SUSTAIN * self.INTERVAL:.0f}s of a danger alert this user causes)")
        return self.state()

    def _watch(self) -> None:
        """Samples the host on the fixed interval forever, logging any evaluation failure."""
        while True:
            time.sleep(self.INTERVAL)
            try:
                self._evaluate()
            except Exception as exc:
                self.logger.error(f"watchdog error: {exc}")

    def _evaluate(self) -> None:
        """Samples CPU busy percent, one-minute load per core, RAM and disk space, updates every alert, then reconsiders auto-nuke.

        CPU, load and disk alerts are never attributed to this user. The RAM
        alert is, once RAM reaches the kill limit with at least MINE_SHARE of
        the used memory attributed to this user.
        """
        cpu   = self._cpu_percent()
        ram   = self._ram_percent()
        cores = os.cpu_count() or 1
        load1 = os.getloadavg()[0]
        ratio = load1 / cores

        if cpu is not None:
            self._track("cpu", "warn", cpu >= self.CPU_ALERT, cpu < self.CPU_ALERT - self.CLEAR_MARGIN, False,
                        f"CPU at {cpu:.0f}%, above the {self.CPU_ALERT:.0f}% threshold")

        self._track("load", "warn", ratio >= self.LOAD_RATIO, ratio < self.LOAD_RATIO - 0.1, False,
                    f"load {load1:.1f} saturates all {cores} cores, no spare CPU capacity")

        if ram is not None:
            share = self.mine_memory_share() if ram >= self.RAM_KILL else None
            mine  = share is not None and share >= self.MINE_SHARE
            self._track("ram", "danger" if ram >= self.RAM_KILL else "warn", ram >= self.RAM_WARN, ram < self.RAM_WARN - self.CLEAR_MARGIN, mine,
                        f"RAM at {ram:.0f}%, jobs are terminated at {self.RAM_KILL:.0f}% once at least {self.MINE_SHARE:.0%} of it is this user's")
            self._check_kill(ram)

        disk = self._disk_free()
        if disk is not None:
            free_pct, free_gb = disk
            self._track("disk", "danger" if free_pct <= self.DISK_DANGER else "warn", free_pct <= self.DISK_WARN, free_pct > self.DISK_WARN + self.CLEAR_MARGIN, False,
                        f"disk almost full: {free_pct:.1f}% free ({free_gb:.0f} GB) on the filesystem holding {self.paths.repo_root}, running jobs will die once it fills")

        self._check_nuke()

    def _track(self, kind: str, level: str, breached: bool, cleared: bool, mine: bool, message: str) -> None:
        """Advances one alert's streak, raising it once sustained and dropping it once cleared.

        Args:
            kind     : Alert identifier, also the key into the sustain and streak tables.
            level    : Severity label attached to the alert when raised.
            breached : Whether this sample is above the alert threshold.
            cleared  : Whether this sample is below the threshold by the clear margin.
            mine     : Whether this user causes the alert, which lets a danger alert fire auto-nuke.
            message  : Human-readable alert text.
        """
        if breached:
            self.streaks[kind] += 1
            if self.streaks[kind] >= self.SUSTAIN[kind]:
                self._raise(kind, level, mine, message)
        else:
            self.streaks[kind] = 0
            if cleared:
                self._drop(kind)

    def _check_kill(self, ram: float) -> None:
        """Terminates every running job once RAM pressure is sustained and mostly this user's.

        The kill needs the RAM percentage above the kill limit for the required
        number of consecutive samples, the cooldown since the last kill to have
        elapsed, and at least MINE_SHARE of the used memory attributed to this
        user; otherwise a warning alert is raised and the jobs are left alone.

        Args:
            ram : Fraction of host memory in use, in percent.
        """
        if ram < self.RAM_KILL:
            self.streaks["ram_kill"] = 0
            self._drop("ram_foreign")
            return

        self.streaks["ram_kill"] += 1
        now = time.monotonic()
        if self.streaks["ram_kill"] < self.SUSTAIN["ram_kill"] or now - self.last_kill < self.KILL_COOLDOWN:
            return

        share = self.mine_memory_share()
        if share is None or share < self.MINE_SHARE:
            owned = "no attribution sample has been taken yet" if share is None else f"only {share:.0%} of the used RAM is attributed to this user"
            self._raise("ram_foreign", "warn", False, f"RAM at {ram:.0f}% is above the {self.RAM_KILL:.0f}% limit but {owned}, console jobs are left running")
            return

        self._drop("ram_foreign")
        self.last_kill = now
        killed         = self.processes.stop_all()
        detail         = f"terminated {killed} running job(s)" if killed else "no console jobs were running to terminate"
        message        = f"RAM at {ram:.0f}% crossed the {self.RAM_KILL:.0f}% limit with {share:.0%} of it attributed to this user, {detail}"

        with self.lock:
            self.events.append({"kind": "ram_kill", "level": "danger", "message": message, "time": datetime.now().isoformat(timespec="seconds")})
        self.logger.error(message)

    def _check_nuke(self) -> None:
        """Fires the process nuke once a danger alert this user causes persists.

        Requires auto-nuke armed, NUKE_SUSTAIN consecutive samples with a
        standing danger alert attributed to this user, and NUKE_COOLDOWN
        seconds since the previous firing. The outcome is logged and appended
        to the event log.
        """
        with self.lock:
            if not self.auto_nuke:
                self.nuke_streak = 0
                return

            severe = [alert for alert in self.active.values() if alert["mine"] and alert["level"] == "danger"]
            if severe:
                self.nuke_streak += 1
            else:
                self.nuke_streak = 0
            streak  = self.nuke_streak
            reasons = [alert["message"] for alert in severe]

        if streak < self.NUKE_SUSTAIN:
            return

        now = time.monotonic()
        if now - self.last_nuke < self.NUKE_COOLDOWN:
            return

        self.last_nuke   = now
        self.nuke_streak = 0
        result  = self.nuke.nuke()
        message = f"AUTO-NUKE fired after {self.NUKE_SUSTAIN * self.INTERVAL:.0f}s of a danger alert this user causes ({reasons[0]}); cancelled {result['cancelled']} queued, terminated {result['signalled']}, force-killed {result['killed']}"

        with self.lock:
            self.events.append({"kind": "auto_nuke", "level": "danger", "message": message, "time": datetime.now().isoformat(timespec="seconds")})
        self.logger.error(message)

    def mine_memory_share(self) -> float | None:
        """Returns the share of the used host memory attributed to this user's processes, or None when it cannot be read.

        Memory is attributed by proportional set size, so pages shared with
        other users count only in part.
        """
        info = MemInfo.fields()
        used = info.get("MemTotal", 0) - info.get("MemAvailable", 0)
        if used <= 0:
            return None

        mine = sum(row["attributed"] or 0 for row in ProcSweep.rows() if row["uid"] == self.uid)
        return min(1.0, mine / used)

    def _raise(self, kind: str, level: str, mine: bool, message: str) -> None:
        """Raises a new alert of this kind, or refreshes the level, attribution and text of a standing one.

        A newly raised CPU or load alert throttles the console's jobs when
        auto-throttle is on; a refresh stays silent.
        """
        with self.lock:
            known = self.active.get(kind)
            if known is None:
                self.active[kind] = {"kind": kind, "level": level, "mine": mine, "message": message, "since": datetime.now().isoformat(timespec="seconds")}
            else:
                known["level"]   = level
                known["mine"]    = mine
                known["message"] = message

        if known is None:
            self.logger.warning(f"alert raised: {message}")
            if kind in self.THROTTLED_BY and self.politeness.auto_throttle():
                self._auto_throttle(kind)

    def _auto_throttle(self, kind: str) -> None:
        """Drops the console's jobs to the lowest priority and records the event."""
        result  = self.throttle.apply()
        message = f"auto-throttle on the {kind} alert: {result['processes']} job process(es) dropped to nice {JobThrottle.NICE} and idle I/O" if result["ok"] else f"auto-throttle on the {kind} alert failed: {result['error']}"

        with self.lock:
            self.events.append({"kind": "throttle", "level": "warn", "message": message, "time": datetime.now().isoformat(timespec="seconds")})

    def _drop(self, kind: str) -> None:
        """Clears the alert of this kind, logging only if one was actually standing."""
        with self.lock:
            known = self.active.pop(kind, None)
        if known is not None:
            self.logger.muted(f"alert cleared: {kind}")

    def _disk_free(self) -> tuple[float, float] | None:
        """Returns the free share and free space of the repository's filesystem.

        Returns:
            Tuple of the free percentage and free gigabytes, or None when the
            filesystem cannot be measured.
        """
        try:
            usage = shutil.disk_usage(self.paths.repo_root)
        except OSError:
            return None

        if not usage.total:
            return None
        return 100.0 * usage.free / usage.total, usage.free / (1024.0 ** 3)

    def _cpu_percent(self) -> float | None:
        """Returns busy CPU percent since the previous sample, or None on the first sample."""
        current       = CpuCounters.read().get("cpu")
        prev          = self.prev_cpu
        self.prev_cpu = current

        if current is None or prev is None or current[1] <= prev[1]:
            return None
        return 100.0 * (current[0] - prev[0]) / (current[1] - prev[1])

    def _ram_percent(self) -> float | None:
        """Returns host memory in use as a percentage, or None when /proc/meminfo is unreadable."""
        return MemInfo.used_percent()
