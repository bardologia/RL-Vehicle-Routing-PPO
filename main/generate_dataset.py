from _bootstrap import REPO_ROOT

from configuration import config
from configuration.cli import ConfigCli
from tools.logger import Logger
from core.dataset import DatasetGenerator


ConfigCli(config).apply()
logger = Logger(name="dataset")
DatasetGenerator(config=config, repo_root=REPO_ROOT, logger=logger).generate()
