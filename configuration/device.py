from dataclasses import dataclass

import torch


@dataclass
class DeviceConfig:
    device: str = "cuda"

    @property
    def torch_device(self):
        return torch.device(self.device)
