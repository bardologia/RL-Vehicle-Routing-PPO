from _bootstrap import REPO_ROOT

from configuration import config
from configuration.cli import ConfigCli
from tools.logger import Logger
from core.inference import EvaluationPipeline


ConfigCli(config).apply()
logger = Logger(name="evaluation")
EvaluationPipeline(config=config, repo_root=REPO_ROOT, logger=logger).run()
