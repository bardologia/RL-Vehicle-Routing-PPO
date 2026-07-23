from dataclasses import dataclass


@dataclass
class PPOConfig:
    gamma      : float = 0.99
    gae_lambda : float = 0.95

    clip_ratio              : float = 0.2
    value_clip_ratio        : float = 0.2
    value_loss_coef         : float = 0.5
    gradient_clip_max_norm  : float = 3.0
    kl_divergence_threshold : float = 0.015

    anchor_kl_start     : float = 0.5
    anchor_kl_end       : float = 0.0
    anchor_anneal_steps : int = 20000
