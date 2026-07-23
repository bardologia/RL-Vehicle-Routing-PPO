from _bootstrap import REPO_ROOT

from configuration import config
from tools.logger import Logger
from core.dataset import DatasetGenerator


logger = Logger(name="dataset")
DatasetGenerator(config=config, repo_root=REPO_ROOT, logger=logger).generate()
