from dataclasses import dataclass


@dataclass
class TelemetryConfig:
    step_every            : int = 8
    episode_every         : int = 4
    sample_every          : int = 32
    layer_gradients_every : int = 50
