"""Shared runtime of the console services: the base configuration, the environment lock and the model cache."""

import random
import threading

import numpy as np
import torch

from configuration import config
from core.inference import EpisodeSettings
from model.policy_model import Policy, PolicyCheckpoint
from project_paths import RunLayout
from web_errors import ApiError


class RngGuard:
    """Context manager that holds the environment lock and gives its owner a private view of the global RNG streams.

    Entering takes the lock and restores the python, numpy and torch generator states saved at the previous exit; leaving saves them again. Other lock holders may reseed the globals in between without disturbing the owner's episode.
    """

    def __init__(self, lock):
        self.lock  = lock
        self.state = None

    def __enter__(self):
        self.lock.acquire()

        if self.state is not None:
            random.setstate(self.state[0])
            np.random.set_state(self.state[1])
            torch.set_rng_state(self.state[2])

        return self

    def __exit__(self, kind, value, trace):
        self.state = (random.getstate(), np.random.get_state(), torch.get_rng_state())
        self.lock.release()
        return False


class ConsoleRuntime:
    """Process-wide state every console service shares.

    Attributes:
        config   : The global Config, pinned to CPU; episode configs are deep copies of it.
        lock     : Re-entrant lock guarding environment, VROOM and global RNG use.
        settings : EpisodeSettings used to derive episode configs.
        layout   : RunLayout of the runs directory.
    """

    def __init__(self, paths):
        config.training.device = "cpu"

        self.config   = config
        self.lock     = threading.RLock()
        self.settings = EpisodeSettings()
        self.layout   = RunLayout(paths)
        self._models  = {}

    def episode_config(self, environment=None, reward=None, rounds=None, edits=None):
        """Return the Config of one episode.

        Args:
            environment : EnvironmentConfig fields from a report or trace meta, or None for the defaults.
            reward      : RewardConfig fields, or None for the defaults.
            rounds      : Disruption rounds per episode, or None for the default.
            edits       : Edit cap per round, or None for the default.

        Returns:
            A deep copy of the base Config with the overrides applied.

        Raises:
            ApiError : The episode asks for a depot service time other than the one the VROOM client of this process uses (400).
        """
        try:
            built = self.settings.build(self.config, environment, reward, rounds, edits)
        except ValueError as error:
            raise ApiError(400, str(error))

        if built.env.depot_service != self.config.env.depot_service:
            raise ApiError(400, f"episode depot_service {built.env.depot_service} differs from the console's {self.config.env.depot_service}; the VROOM client reads the console value")

        return built

    def rng_guard(self):
        """Return a new RngGuard on the environment lock."""
        return RngGuard(self.lock)

    def model(self, run_name):
        """Return the policy of a run on CPU in eval mode, loading it once per checkpoint mtime.

        Raises:
            ValueError : The run does not exist or holds no checkpoint.
            RuntimeError : The checkpoint does not fit the current architecture.
        """
        run_dir = self.layout.run_dir(run_name or "")
        if run_dir is None:
            raise ValueError(f"unknown run '{run_name}'")

        checkpoint = self.layout.checkpoint(run_dir)
        if not checkpoint.exists():
            raise ValueError(f"run '{run_name}' has no checkpoint")

        with self.lock:
            mtime  = checkpoint.stat().st_mtime
            cached = self._models.get(run_name)
            if cached is not None and cached[0] == mtime:
                return cached[1]

            model = Policy(self.config)
            PolicyCheckpoint().load(model, checkpoint.name, str(run_dir), map_location="cpu")
            model.eval()

            self._models[run_name] = (mtime, model)
            return model
