import os
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

from tools.logger import Logger
from tools.tracker import Tracker
from .training import Trainer


class RunDirectory:
    def __init__(self, config, runs_root):
        self.config    = config
        self.runs_root = runs_root

        self.path    = None
        self.writer  = None
        self.tracker = None

    def _resolve_path(self):
        resume_name = self.config.io.resume_from_run

        if resume_name:
            path = os.path.join(self.runs_root, resume_name)
            if not os.path.isdir(path):
                raise FileNotFoundError(f"Cannot resume run '{resume_name}': directory does not exist at {path}")
            return path

        run_name = self.config.io.run_name or datetime.now().strftime("run_%Y%m%d-%H%M%S")
        return os.path.join(self.runs_root, run_name)

    def _open_writer(self):
        os.makedirs(self.path, exist_ok=True)

        self.writer  = SummaryWriter(log_dir=self.path)
        self.tracker = Tracker(writer=self.writer)

    def prepare(self):
        self.path             = self._resolve_path()
        self.config.io.logdir = self.path

        self._open_writer()
        return self


class TrainingPipeline:
    def __init__(self, config, repo_root):
        self.config    = config
        self.repo_root = repo_root

        self.runs_root   = None
        self.session     = None
        self.logger      = None
        self.trainer     = None

    def _resolve_paths(self):
        self.runs_root = self._absolute(self.config.io.runs_dir)

        self.config.io.runs_dir = self.runs_root

    def _absolute(self, path):
        if os.path.isabs(path):
            return path
        return os.path.join(str(self.repo_root), path)

    def _open_session(self):
        self.session = RunDirectory(self.config, self.runs_root).prepare()

    def _build_logger(self):
        self.logger = Logger(log_dir=self.config.io.logdir, name="training", level="INFO")

    def _build_trainer(self):
        self.trainer = Trainer(config=self.config, logger=self.logger, tracker=self.session.tracker)

    def run(self):
        self._resolve_paths()
        self._open_session()
        self._build_logger()
        self._build_trainer()
        return self.trainer.train()
