from dataclasses import dataclass


@dataclass
class InferenceConfig:
    seed   : int = 2_000_000
    greedy : bool = True
