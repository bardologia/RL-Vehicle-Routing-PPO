import os
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

from tools.logger import Logger
from tools.tracker import Tracker
from core.dataset import Dataset
from .training import Trainer


class RunDirectory:
    def __init__(self, config, runs_root):
        self.config    = config
        self.runs_root = runs_root

        self.path    = None
        self.writer  = None
        self.tracker = None

    def resolve_path(self):
        resume_name = self.config.io.resume_from_run

        if resume_name:
            path = os.path.join(self.runs_root, resume_name)
            if not os.path.isdir(path):
                raise FileNotFoundError(f"Cannot resume run '{resume_name}': directory does not exist at {path}")
            return path

        run_name = self.config.io.run_name or datetime.now().strftime("run_%Y%m%d-%H%M%S")
        return os.path.join(self.runs_root, run_name)

    def open_writer(self):
        os.makedirs(self.path, exist_ok=True)

        self.writer  = SummaryWriter(log_dir=self.path)
        self.tracker = Tracker(writer=self.writer)

    def prepare(self):
        self.path             = self.resolve_path()
        self.config.io.logdir = self.path

        self.open_writer()
        return self


class TrainingPipeline:
    def __init__(self, config, repo_root):
        self.config    = config
        self.repo_root = repo_root

        self.runs_root   = None
        self.dataset_dir = None
        self.session     = None
        self.logger      = None
        self.dataset     = None
        self.trainer     = None

    def resolve_paths(self):
        self.runs_root   = self._absolute(self.config.io.runs_dir)
        self.dataset_dir = self._absolute(self.config.io.dataset_dir)

    def _absolute(self, path):
        if os.path.isabs(path):
            return path
        return os.path.join(str(self.repo_root), path)

    def open_session(self):
        self.session = RunDirectory(self.config, self.runs_root).prepare()

    def build_logger(self):
        self.logger = Logger(log_dir=self.config.io.logdir, name="training", level="INFO")

    def load_dataset(self):
        if not os.path.isdir(self.dataset_dir):
            raise FileNotFoundError(f"Dataset directory not found: {self.dataset_dir}")

        self.config.io.dataset_dir = self.dataset_dir
        self.dataset               = Dataset(dataset_dir=self.dataset_dir, config=self.config, shuffle_chunks=False, logger=self.logger)

    def build_trainer(self):
        self.trainer = Trainer(dataset=self.dataset, config=self.config, logger=self.logger, tracker=self.session.tracker)

    def run(self):
        self.resolve_paths()
        self.open_session()
        self.build_logger()
        self.load_dataset()
        self.build_trainer()
        return self.trainer.train()
