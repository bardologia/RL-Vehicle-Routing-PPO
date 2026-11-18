from dataclasses import dataclass, field

from configuration.service     import ServiceConfig
from configuration.io          import IOConfig
from configuration.environment import EnvironmentConfig
from configuration.evaluation  import EvaluationConfig
from configuration.model       import ModelConfig
from configuration.learning    import LearningRate, Entropy
from configuration.ppo         import PPOConfig
from configuration.pretraining import PretrainConfig
from configuration.training    import TrainingConfig
from configuration.reward      import RewardConfig
from configuration.telemetry   import TelemetryConfig
from configuration.inference   import InferenceConfig
from configuration.monitor     import MonitorConfig
from configuration.diagnostics import DiagnosticsConfig


@dataclass
class Config:
    service     : ServiceConfig     = field(default_factory=ServiceConfig)
    io          : IOConfig          = field(default_factory=IOConfig)
    env         : EnvironmentConfig = field(default_factory=EnvironmentConfig)
    lr          : LearningRate      = field(default_factory=LearningRate)
    entropy     : Entropy           = field(default_factory=Entropy)
    model       : ModelConfig       = field(default_factory=ModelConfig)
    training    : TrainingConfig    = field(default_factory=TrainingConfig)
    reward      : RewardConfig      = field(default_factory=RewardConfig)
    ppo         : PPOConfig         = field(default_factory=PPOConfig)
    pretrain    : PretrainConfig    = field(default_factory=PretrainConfig)
    evaluation  : EvaluationConfig  = field(default_factory=EvaluationConfig)
    telemetry   : TelemetryConfig   = field(default_factory=TelemetryConfig)
    inference   : InferenceConfig   = field(default_factory=InferenceConfig)
    monitor     : MonitorConfig     = field(default_factory=MonitorConfig)
    diagnostics : DiagnosticsConfig = field(default_factory=DiagnosticsConfig)


config = Config()


__all__ = [
    "ServiceConfig",
    "IOConfig",
    "EnvironmentConfig",
    "EvaluationConfig",
    "ModelConfig",
    "LearningRate",
    "Entropy",
    "PPOConfig",
    "PretrainConfig",
    "TrainingConfig",
    "RewardConfig",
    "TelemetryConfig",
    "InferenceConfig",
    "MonitorConfig",
    "DiagnosticsConfig",
    "Config",
    "config",
]
