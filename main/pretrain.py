from _bootstrap import REPO_ROOT

from configuration import config
from configuration.cli import ConfigCli
from core.training import PretrainingPipeline


ConfigCli(config).apply()
PretrainingPipeline(config=config, repo_root=REPO_ROOT).run()
