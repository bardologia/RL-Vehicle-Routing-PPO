from dataclasses import dataclass


@dataclass
class PretrainConfig:
    episodes               : int = 2000
    bc_epochs              : int = 4
    minibatch_size         : int = 128
    lr                     : float = 3e-4
    value_loss_coef        : float = 0.5
    gradient_clip_max_norm : float = 3.0
    reoptimize_margin      : float = 0.0
