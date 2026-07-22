from dataclasses import dataclass
from typing      import Optional

from torch.utils.tensorboard import SummaryWriter


@dataclass
class IOConfig:
    logdir: Optional[str] = None
    writer: Optional[SummaryWriter] = None
    dataset_dir: str = "datasets/chunked"
    resume_from_run: Optional[str] = None
