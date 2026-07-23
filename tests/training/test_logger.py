import cProfile
import pstats

from tools.logger import Logger, NullLogger


def _profile_stats():
    def work():
        return sum(index * index for index in range(2000))

    profiler = cProfile.Profile()
    profiler.enable()
    work()
    profiler.disable()

    stats        = pstats.Stats(profiler)
    stats.stream = None
    return stats


def test_null_logger_returns_silent_callables():
    logger = NullLogger()

    assert logger.info("anything") is None
    assert logger.warning("anything", "extra") is None
    assert logger.section("x") is None


def test_logger_writes_log_file_and_sections(tmp_path):
    logger = Logger(log_dir=str(tmp_path), name="unit")

    logger.section("Setup")
    logger.subsection("detail")
    logger.kv_table({"a": 1, "b": 2}, title="Table")
    logger.info("hello world")
    logger.close()

    log_file = tmp_path / "unit.log"
    assert log_file.exists()

    text = log_file.read_text(encoding="utf-8")
    assert "SETUP" in text
    assert "hello world" in text


def test_logger_save_profiler_results_writes_markdown_table(tmp_path):
    logger = Logger(log_dir=str(tmp_path), name="prof")
    output = tmp_path / "profile.md"

    logger.save_profiler_results(_profile_stats(), str(output))
    logger.close()

    text = output.read_text(encoding="utf-8")
    assert text.startswith("# Profiler Results")
    assert "| Function" in text
    assert "| Cumulative Time (s)" in text
    assert "Generated:" in text
