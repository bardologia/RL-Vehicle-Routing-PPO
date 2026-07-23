import pytest

from configuration import Config
from configuration.cli import ConfigCli


def test_cli_applies_int_override():
    config = ConfigCli(Config(), ["--io.dataset_seed", "7"]).apply()

    assert config.io.dataset_seed == 7


def test_cli_applies_float_override():
    config = ConfigCli(Config(), ["--ppo.gamma", "0.9"]).apply()

    assert config.ppo.gamma == 0.9


def test_cli_applies_string_override():
    config = ConfigCli(Config(), ["--training.device", "cpu"]).apply()

    assert config.training.device == "cpu"


def test_cli_applies_bool_override():
    config = ConfigCli(Config(), ["--training.use_mixed_precision", "true"]).apply()

    assert config.training.use_mixed_precision is True


def test_cli_applies_optional_override():
    config = ConfigCli(Config(), ["--io.run_name", "run_a"]).apply()

    assert config.io.run_name == "run_a"


def test_cli_applies_optional_none_override():
    config = ConfigCli(Config(), ["--io.run_name", "none"]).apply()

    assert config.io.run_name is None


def test_cli_applies_tuple_override():
    config = ConfigCli(Config(), ["--env.center", "-46.7,-23.6"]).apply()

    assert config.env.center == (-46.7, -23.6)


def test_cli_applies_multiple_overrides():
    config = ConfigCli(Config(), ["--io.dataset_seed", "3", "--env.radius", "10.0"]).apply()

    assert config.io.dataset_seed == 3
    assert config.env.radius == 10.0


def test_cli_rejects_unknown_section():
    with pytest.raises(ValueError):
        ConfigCli(Config(), ["--nope.field", "1"]).apply()


def test_cli_rejects_unknown_field():
    with pytest.raises(ValueError):
        ConfigCli(Config(), ["--io.nope", "1"]).apply()


def test_cli_rejects_bare_path():
    with pytest.raises(ValueError):
        ConfigCli(Config(), ["--radius", "1"]).apply()


def test_cli_rejects_dangling_flag():
    with pytest.raises(ValueError):
        ConfigCli(Config(), ["--env.radius"]).apply()


def test_cli_rejects_bad_bool():
    with pytest.raises(ValueError):
        ConfigCli(Config(), ["--training.use_mixed_precision", "maybe"]).apply()


def test_cli_rejects_dict_field():
    with pytest.raises(ValueError):
        ConfigCli(Config(), ["--service.options", "{}"]).apply()
