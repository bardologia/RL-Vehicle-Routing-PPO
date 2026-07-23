import os
import threading
import time

import psutil
import torch


class ResourceMonitor:
    TB_SCALARS     = ("ram_pct", "proc_rss_gb", "swap_pct", "shm_pct", "cpu_pct", "proc_cpu_pct", "disk_read_mb_s", "disk_write_mb_s", "vram_used_gb", "vram_pct")
    TB_GPU_SCALARS = ("vram_pct", "vram_used_gb", "alloc_gb", "reserved_gb")

    def __init__(self, config, logger=None, tracker=None, step_getter=None):
        self.config      = config
        self.logger      = logger
        self.tracker     = tracker
        self.step_getter = step_getter or (lambda: self.tracker.current_step if self.tracker is not None else 0)

        self._load_config()
        self._init_process()
        self._init_gpu()
        self._init_disk_tracking()
        self._init_threading()
        self._init_peak_tracking()

    def _load_config(self):
        self.enabled         = bool(self.config.enabled)
        self.interval        = float(self.config.poll_interval_sec)
        self.log_to_tb       = bool(self.config.log_to_tensorboard)
        self.warn_ram_pct    = float(self.config.warn_ram_pct)
        self.warn_vram_pct   = float(self.config.warn_vram_pct)
        self.warn_swap_pct   = float(self.config.warn_swap_pct)
        self.warn_shm_pct    = float(self.config.warn_shm_pct)
        self.warn_cooldown_s = float(self.config.warn_cooldown_sec)

    def _init_process(self):
        self.process = psutil.Process(os.getpid())
        self.process.cpu_percent(None)
        psutil.cpu_percent(None, percpu=False)

    def _init_gpu(self):
        self._cuda_ok   = torch.cuda.is_available()
        self._gpu_count = torch.cuda.device_count() if self._cuda_ok else 0

    def _init_disk_tracking(self):
        self._last_disk_io = psutil.disk_io_counters()
        self._last_disk_t  = time.time()

    def _init_threading(self):
        self._stop_evt    = threading.Event()
        self._thread      = None
        self._sample_idx  = 0
        self._last_warn_t = {}

    def _init_peak_tracking(self):
        self.peak = {
            "ram_used_gb"  : 0.0,
            "ram_pct"      : 0.0,
            "proc_rss_gb"  : 0.0,
            "swap_used_gb" : 0.0,
            "shm_used_gb"  : 0.0,
            "vram_used_gb" : 0.0,
            "vram_pct"     : 0.0,
        }

    @staticmethod
    def _bytes_to_gb(value):
        return float(value) / (1024.0 ** 3)

    def _get_shm_usage(self):
        try:
            usage = psutil.disk_usage("/dev/shm")
            return self._bytes_to_gb(usage.used), float(usage.percent)
        except (FileNotFoundError, PermissionError, OSError):
            return 0.0, 0.0

    def _sample_ram_metrics(self, metrics):
        vm = psutil.virtual_memory()
        metrics["ram_used_gb"]      = self._bytes_to_gb(vm.used)
        metrics["ram_available_gb"] = self._bytes_to_gb(vm.available)
        metrics["ram_total_gb"]     = self._bytes_to_gb(vm.total)
        metrics["ram_pct"]          = float(vm.percent)

    def _sample_swap_metrics(self, metrics):
        sm = psutil.swap_memory()
        metrics["swap_used_gb"] = self._bytes_to_gb(sm.used)
        metrics["swap_pct"]     = float(sm.percent)

    def _sample_process_memory_metrics(self, metrics):
        mi = self.process.memory_info()
        metrics["proc_rss_gb"] = self._bytes_to_gb(mi.rss)
        metrics["proc_vms_gb"] = self._bytes_to_gb(mi.vms)

    def _sample_process_stats(self, metrics):
        try:
            metrics["proc_num_threads"] = float(self.process.num_threads())
            metrics["proc_num_fds"]     = float(self.process.num_fds())
        except (psutil.AccessDenied, AttributeError):
            pass

    def _sample_cpu_metrics(self, metrics):
        metrics["cpu_pct"]      = float(psutil.cpu_percent(None, percpu=False))
        metrics["proc_cpu_pct"] = float(self.process.cpu_percent(None))
        try:
            la1, la5, la15 = os.getloadavg()
            metrics["loadavg_1m"]  = float(la1)
            metrics["loadavg_5m"]  = float(la5)
            metrics["loadavg_15m"] = float(la15)
        except (AttributeError, OSError):
            pass

    def _sample_shm_metrics(self, metrics):
        shm_used_gb, shm_pct = self._get_shm_usage()
        metrics["shm_used_gb"] = shm_used_gb
        metrics["shm_pct"]     = shm_pct

    def _sample_disk_io_metrics(self, metrics):
        now = time.time()
        try:
            io = psutil.disk_io_counters()
            dt = max(now - self._last_disk_t, 1e-6)
            if io is not None and self._last_disk_io is not None:
                metrics["disk_read_mb_s"]  = (io.read_bytes - self._last_disk_io.read_bytes) / dt / (1024.0 ** 2)
                metrics["disk_write_mb_s"] = (io.write_bytes - self._last_disk_io.write_bytes) / dt / (1024.0 ** 2)
            self._last_disk_io = io
            self._last_disk_t  = now
        except (PermissionError, OSError):
            pass

    def _sample_gpu_metrics(self, metrics):
        gpu_used  = 0.0
        gpu_total = 0.0

        for i in range(self._gpu_count):
            free, total = torch.cuda.mem_get_info(i)
            used        = total - free
            used_gb     = self._bytes_to_gb(used)
            total_gb    = self._bytes_to_gb(total)

            metrics[f"gpu{i}_vram_used_gb"]  = used_gb
            metrics[f"gpu{i}_vram_free_gb"]  = self._bytes_to_gb(free)
            metrics[f"gpu{i}_vram_total_gb"] = total_gb
            metrics[f"gpu{i}_vram_pct"]      = 100.0 * used / max(total, 1)
            metrics[f"gpu{i}_alloc_gb"]      = self._bytes_to_gb(torch.cuda.memory_allocated(i))
            metrics[f"gpu{i}_reserved_gb"]   = self._bytes_to_gb(torch.cuda.memory_reserved(i))
            metrics[f"gpu{i}_max_alloc_gb"]  = self._bytes_to_gb(torch.cuda.max_memory_allocated(i))

            gpu_used  += used_gb
            gpu_total += total_gb

        return gpu_used, gpu_total

    def _update_peak_metrics(self, metrics):
        for key in list(self.peak.keys()):
            if key in metrics and metrics[key] > self.peak[key]:
                self.peak[key] = float(metrics[key])

    def _maybe_warn(self, key, message):
        if self.logger is None:
            return

        now  = time.time()
        last = self._last_warn_t.get(key, 0.0)
        if now - last < self.warn_cooldown_s:
            return

        self._last_warn_t[key] = now
        self.logger.warning(f"[ResourceMonitor] {message}")

    def _check_warnings(self, metrics):
        if metrics["ram_pct"] >= self.warn_ram_pct:
            self._maybe_warn("ram", f"RAM usage {metrics['ram_pct']:.1f}% ({metrics['ram_used_gb']:.2f}/{metrics['ram_total_gb']:.2f} GB) >= threshold {self.warn_ram_pct:.1f}% (proc RSS {metrics.get('proc_rss_gb', 0.0):.2f} GB, shm {metrics['shm_used_gb']:.2f} GB)")

        if metrics["vram_pct"] >= self.warn_vram_pct and metrics["vram_used_gb"] > 0.0:
            self._maybe_warn("vram", f"VRAM usage {metrics['vram_pct']:.1f}% ({metrics['vram_used_gb']:.2f} GB) >= threshold {self.warn_vram_pct:.1f}%")

        if metrics["swap_pct"] >= self.warn_swap_pct:
            self._maybe_warn("swap", f"Swap usage {metrics['swap_pct']:.1f}% ({metrics['swap_used_gb']:.2f} GB) >= threshold {self.warn_swap_pct:.1f}%")

        if metrics["shm_pct"] >= self.warn_shm_pct:
            self._maybe_warn("shm", f"/dev/shm usage {metrics['shm_pct']:.1f}% ({metrics['shm_used_gb']:.2f} GB) >= threshold {self.warn_shm_pct:.1f}% (DataLoader workers may exhaust shared memory)")

    def sample(self):
        metrics = {}

        self._sample_ram_metrics(metrics)
        self._sample_swap_metrics(metrics)
        self._sample_process_memory_metrics(metrics)
        self._sample_process_stats(metrics)
        self._sample_cpu_metrics(metrics)
        self._sample_shm_metrics(metrics)
        self._sample_disk_io_metrics(metrics)

        gpu_used, gpu_total = self._sample_gpu_metrics(metrics)

        metrics["vram_used_gb"] = gpu_used
        metrics["vram_pct"]     = (100.0 * gpu_used / gpu_total) if gpu_total > 0 else 0.0

        self._update_peak_metrics(metrics)
        self._check_warnings(metrics)

        return metrics

    def _tb_metrics(self, metrics):
        allowed = set(self.TB_SCALARS)
        for i in range(self._gpu_count):
            allowed.update(f"gpu{i}_{suffix}" for suffix in self.TB_GPU_SCALARS)

        return {key: value for key, value in metrics.items() if key in allowed}

    def _publish(self, metrics):
        if not (self.log_to_tb and self.tracker is not None and self.tracker.active):
            return

        self.tracker.log_metrics("system", self._tb_metrics(metrics), int(self.step_getter() or 0))

    def _run(self):
        while not self._stop_evt.is_set():
            try:
                metrics = self.sample()
                self._publish(metrics)
                self._sample_idx += 1
            except Exception as exc:
                if self.logger is not None:
                    self.logger.warning(f"[ResourceMonitor] sample failed: {exc}")

            self._stop_evt.wait(self.interval)

    def _log_startup_info(self):
        if self.logger is None:
            return

        self.logger.section("[Resource Monitor]")
        self.logger.kv_table({
            "Enabled"         : True,
            "Poll interval"   : f"{self.interval:.1f} s",
            "CUDA available"  : f"{self._cuda_ok} ({self._gpu_count} GPUs)",
            "TB logging"      : self.log_to_tb,
            "Warn thresholds" : f"RAM>={self.warn_ram_pct:.0f}%  VRAM>={self.warn_vram_pct:.0f}%  SWAP>={self.warn_swap_pct:.0f}%  SHM>={self.warn_shm_pct:.0f}%",
        })

    def start(self):
        if not self.enabled:
            if self.logger is not None:
                self.logger.subsection("[ResourceMonitor] disabled by config")
            return

        if self._thread is not None and self._thread.is_alive():
            return

        self._log_startup_info()

        self._stop_evt.clear()
        self._thread = threading.Thread(target=self._run, name="ResourceMonitor", daemon=True)
        self._thread.start()

    def _stop_thread(self):
        if self._thread is None:
            return

        self._stop_evt.set()
        self._thread.join(timeout=max(self.interval * 2, 5.0))
        self._thread = None

    def _publish_peaks(self):
        if not (self.tracker is not None and self.tracker.active):
            return

        with self.tracker.scope("system"):
            self.tracker.log_metrics("peak", self.peak, int(self.step_getter() or 0))

    def _log_peak_metrics(self):
        if self.logger is None:
            return

        peaks = {}
        for key, value in self.peak.items():
            unit                 = "%" if key.endswith("_pct") else "GB"
            peaks[f"peak {key}"] = f"{value:.2f} {unit}"
        peaks["Total samples"] = self._sample_idx

        self.logger.section("[Resource Monitor - Peaks]")
        self.logger.kv_table(peaks)

    def stop(self):
        self._stop_thread()
        self._publish_peaks()
        self._log_peak_metrics()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        return False
