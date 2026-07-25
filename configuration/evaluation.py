from dataclasses import dataclass


@dataclass
class EvaluationConfig:
    episodes : int = 200
    seed     : int = 1_000_000
