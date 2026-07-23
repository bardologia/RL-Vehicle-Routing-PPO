import configuration

from configuration import Config
from configuration.device      import DeviceConfig
from configuration.environment import EnvironmentConfig
from configuration.io          import IOConfig
from configuration.learning    import Entropy, LearningRate
from configuration.model       import ModelConfig
from configuration.monitor     import MonitorConfig
from configuration.ppo         import PPOConfig
from configuration.reward      import RewardConfig
from configuration.service     import ServiceConfig
from configuration.telemetry   import TelemetryConfig
from configuration.training    import TrainingConfig


def test_config_assembles_every_section_with_correct_type():
    config = Config()

    assert isinstance(config.service, ServiceConfig)
    assert isinstance(config.io, IOConfig)
    assert isinstance(config.env, EnvironmentConfig)
    assert isinstance(config.lr, LearningRate)
    assert isinstance(config.entropy, Entropy)
    assert isinstance(config.model, ModelConfig)
    assert isinstance(config.training, TrainingConfig)
    assert isinstance(config.reward, RewardConfig)
    assert isinstance(config.ppo, PPOConfig)
    assert isinstance(config.telemetry, TelemetryConfig)
    assert isinstance(config.device, DeviceConfig)
    assert isinstance(config.monitor, MonitorConfig)


def test_each_config_instance_owns_independent_sections():
    first  = Config()
    second = Config()

    assert first.env is not second.env
    assert first.training is not second.training

    first.training.device = "cpu"

    assert second.training.device == "cuda"


def test_module_singleton_is_a_config():
    assert isinstance(configuration.config, Config)
    assert isinstance(configuration.config.model, ModelConfig)


def test_exported_names_are_importable():
    for name in configuration.__all__:
        assert hasattr(configuration, name)
