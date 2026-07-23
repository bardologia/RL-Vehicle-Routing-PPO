import os
import sys

proj_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
if proj_root not in sys.path:
    sys.path.insert(0, proj_root)

from configuration import config
from tools.logger import Logger
from core.dataset import DatasetGenerator

logger    = Logger(name="dataset")
generator = DatasetGenerator(dataset_dir=os.path.join(proj_root, "datasets/chunked"), config=config, logger=logger)

generator.append(
    num_events=1024000,
    output_dir=os.path.join(proj_root, "datasets/chunked"),
    batch_size=128,
    chunk_size=1024,
    seed=42,
)
