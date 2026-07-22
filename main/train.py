import os
import sys

# Ensure project root is on sys.path before importing local packages
proj_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from tools.config import config

from core.dataset import Dataset
from core.training import Trainer
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# Resolve dataset directory: prefer configured path, fallback to project `data/` if it exists
dataset_dir_candidate = config.io.dataset_dir
if not os.path.isabs(dataset_dir_candidate):
    dataset_dir_candidate = os.path.join(proj_root, dataset_dir_candidate)

if not os.path.exists(dataset_dir_candidate):
    alt_data_dir = os.path.join(proj_root, "data")
    if os.path.exists(alt_data_dir):
        config.io.dataset_dir = alt_data_dir
    else:
        config.io.dataset_dir = dataset_dir_candidate

dataset = Dataset(dataset_dir=config.io.dataset_dir, config=config, shuffle_chunks=False)
trainer = Trainer(dataset=dataset, config=config)

# Ensure a log directory exists so checkpointing works
if not config.io.logdir:
    timestamp = datetime.now().strftime("run_%Y%m%d-%H%M%S")
    config.io.logdir = os.path.join(proj_root, "runs", timestamp)
    os.makedirs(config.io.logdir, exist_ok=True)
    config.io.writer = SummaryWriter(log_dir=config.io.logdir)

ppo = trainer.train()