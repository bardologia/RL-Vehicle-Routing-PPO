from dataclasses import dataclass
from typing      import Tuple


@dataclass
class DiagnosticsConfig:
    seed            : int = 1_000_000
    episodes        : int = 10
    agents          : Tuple[str, ...] = ("model", "insertion_only", "do_nothing")
    greedy          : bool = True
    sampled_twin    : bool = False
    teacher_verdict : str = "insertions"
