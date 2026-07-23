import requests
import torch

from configuration.device      import DeviceConfig
from configuration.environment import EnvironmentConfig
from configuration.service     import ServiceConfig


def test_device_torch_device_reflects_string():
    assert DeviceConfig(device="cpu").torch_device == torch.device("cpu")
    assert DeviceConfig(device="cuda").torch_device == torch.device("cuda")


def test_outlier_probability_is_reciprocal_of_frequency():
    config = EnvironmentConfig(outlier_frequency=8)

    assert config.outlier_probability == 1.0 / 8


def test_outlier_probability_zero_when_frequency_non_positive():
    assert EnvironmentConfig(outlier_frequency=0).outlier_probability == 0.0
    assert EnvironmentConfig(outlier_frequency=-3).outlier_probability == 0.0


def test_http_session_is_built_lazily_and_cached():
    config = ServiceConfig()

    assert config._http_session is None

    first  = config.http_session
    second = config.http_session

    assert isinstance(first, requests.Session)
    assert first is second
    assert first.headers["Content-Type"] == "application/json"


def test_http_session_mounts_pooled_adapters():
    config  = ServiceConfig()
    session = config.http_session

    assert session.get_adapter("http://localhost") is not None
    assert session.get_adapter("https://localhost") is not None
