from _bootstrap import REPO_ROOT

from configuration import config
from core.training import TrainingPipeline


TrainingPipeline(config=config, repo_root=REPO_ROOT).run()
