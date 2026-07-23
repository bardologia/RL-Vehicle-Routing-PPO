import pytest

from core.inference import InferencePipeline
from tools.logger import NullLogger


def build_pipeline(cpu_config, repo_root):
    return InferencePipeline(config=cpu_config, repo_root=repo_root, logger=NullLogger())


def test_resolve_run_requires_run_name(cpu_config, tmp_path):
    cpu_config.io.run_name = None
    pipeline               = build_pipeline(cpu_config, tmp_path)

    with pytest.raises(ValueError):
        pipeline.resolve_run()


def test_resolve_run_missing_directory_raises(cpu_config, tmp_path):
    cpu_config.io.run_name = "does_not_exist"
    cpu_config.io.runs_dir = str(tmp_path)
    pipeline               = build_pipeline(cpu_config, tmp_path)

    with pytest.raises(FileNotFoundError):
        pipeline.resolve_run()


def test_resolve_run_sets_run_dir_for_existing_run(cpu_config, tmp_path):
    run_dir = tmp_path / "trained_run"
    run_dir.mkdir()

    cpu_config.io.run_name = "trained_run"
    cpu_config.io.runs_dir = str(tmp_path)
    pipeline               = build_pipeline(cpu_config, tmp_path)

    pipeline.resolve_run()

    assert pipeline.run_dir == str(run_dir)
