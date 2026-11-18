from _bootstrap import REPO_ROOT

from configuration import config
from configuration.cli import ConfigCli
from tools.logger import Logger
from core.inference import DiagnosticsPipeline


ConfigCli(config).apply()
logger = Logger(name="diagnostics")
DiagnosticsPipeline(config=config, repo_root=REPO_ROOT, logger=logger).run()
