from _bootstrap import REPO_ROOT

from configuration import config
from configuration.cli import ConfigCli
from core.training import TrainingPipeline


ConfigCli(config).apply()
TrainingPipeline(config=config, repo_root=REPO_ROOT).run()
