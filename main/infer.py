from _bootstrap import REPO_ROOT

from configuration import config
from tools.logger import Logger
from core.inference import InferencePipeline


logger = Logger(name="inference")
InferencePipeline(config=config, repo_root=REPO_ROOT, logger=logger).run()
