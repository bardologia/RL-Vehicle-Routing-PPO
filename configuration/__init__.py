from dataclasses import dataclass, field

from configuration.service     import ServiceConfig
from configuration.io          import IOConfig
from configuration.environment import EnvironmentConfig
from configuration.model       import ModelConfig
from configuration.learning    import LearningRate, Entropy
from configuration.ppo         import PPOConfig
from configuration.training    import TrainingConfig
from configuration.reward      import RewardConfig
from configuration.telemetry   import TelemetryConfig
from configuration.device      import DeviceConfig


@dataclass
class Config:
    service    : ServiceConfig     = field(default_factory=ServiceConfig)
    io         : IOConfig          = field(default_factory=IOConfig)
    env        : EnvironmentConfig = field(default_factory=EnvironmentConfig)
    lr         : LearningRate      = field(default_factory=LearningRate)
    entropy    : Entropy           = field(default_factory=Entropy)
    model      : ModelConfig       = field(default_factory=ModelConfig)
    training   : TrainingConfig    = field(default_factory=TrainingConfig)
    reward     : RewardConfig      = field(default_factory=RewardConfig)
    ppo        : PPOConfig         = field(default_factory=PPOConfig)
    telemetry  : TelemetryConfig   = field(default_factory=TelemetryConfig)
    device     : DeviceConfig      = field(default_factory=DeviceConfig)


config = Config()


__all__ = [
    "ServiceConfig",
    "IOConfig",
    "EnvironmentConfig",
    "ModelConfig",
    "LearningRate",
    "Entropy",
    "PPOConfig",
    "TrainingConfig",
    "RewardConfig",
    "TelemetryConfig",
    "DeviceConfig",
    "Config",
    "config",
]
