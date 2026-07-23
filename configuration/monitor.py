from dataclasses import dataclass


@dataclass
class MonitorConfig:
    enabled            : bool  = True
    poll_interval_sec  : float = 10.0
    log_to_tensorboard : bool  = True
    warn_ram_pct       : float = 90.0
    warn_vram_pct      : float = 90.0
    warn_swap_pct      : float = 50.0
    warn_shm_pct       : float = 80.0
    warn_cooldown_sec  : float = 30.0
