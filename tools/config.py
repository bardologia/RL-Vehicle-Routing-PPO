import requests
from requests.adapters import HTTPAdapter
from torch.utils.tensorboard import SummaryWriter
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple
import torch


@dataclass
class ServiceConfig:
    vroom_url: str = "http://localhost:3000"
    osrm_url: str = "http://localhost:5000"
    options: Dict[str, Any] = field(default_factory=lambda: {"g": True, "geometry": True, "threads": 8})
    _http_session: Optional[requests.Session] = field(init=False, default=None)

    @property
    def http_session(self) -> requests.Session:
        if self._http_session is None:
            sess = requests.Session()
            sess.headers.update({"Content-Type": "application/json"})
            adapter = HTTPAdapter(pool_connections=32, pool_maxsize=64)
            sess.mount("https://", adapter)
            sess.mount("http://", adapter)
            self._http_session = sess
        return self._http_session

@dataclass
class IOConfig:
    logdir: Optional[str] = None
    writer: Optional[SummaryWriter] = None
    dataset_dir: str = "datasets/chunked"
    resume_from_run: Optional[str] = None  # e.g., "run_20260110-100451"

@dataclass
class EnvironmentConfig:
    center: Tuple[float, float] = (-46.63, -23.55)
    radius: float = 25.0
    
    min_vehicles : int = 2
    min_jobs     : int = 4
    
    mean_jobs     : int = 16
    std_jobs      : int = 4
    mean_vehicles : int = 4
    std_vehicles  : int = 1
    
    outlier_frequency  : int = 8
    outlier_multiplier : int = 2
    
    job_insert_min: int = 2
    job_insert_max: int = 3
    job_remove_min: int = 1
    job_remove_max: int = 2

    vehicle_insert_min: int = 1
    vehicle_insert_max: int = 1
    vehicle_remove_min: int = 1
    vehicle_remove_max: int = 1

    @property
    def outlier_probability(self):
        return 1.0 / self.outlier_frequency if self.outlier_frequency > 0 else 0.0

@dataclass
class ModelConfig:
    job_input_dim     : int = 7
    vehicle_input_dim : int = 5
    edge_attr_dim     : int = 4

    num_operators          : int = 4
    operator_embedding_dim : int = 32

    gnn_num_layers : int = 2

    policy_gnn_hidden_channels : int = 64
    policy_embedding_dim       : int = 64
    policy_actor_hidden_1      : int = 128
    policy_actor_hidden_2      : int = 64

    value_critic_hidden_1 : int = 64
    value_critic_hidden_2 : int = 64

@dataclass
class LearningRate:
    lr_operator_actor : float = 3e-4
    lr_vehicle_actor  : float = 3e-4
    lr_critic         : float = 5e-4
    lr_embedding      : float = 2e-4
    lr_job_actor      : float = 3e-4
    
    lr_warmup_steps: int = 1000
    lr_min: float = 1e-5
    lr_decay_steps: int = 100000

@dataclass
class Entropy:
    entropy_coef         : float = 0.02
    entropy_start        : float = 0.02
    entropy_end          : float = 0.001
    entropy_anneal_steps : int = 50000

@dataclass
class PPOConfig:
    ppo_epochs          : int = 4
    ppo_clip_epsilon    : float = 0.2
    ppo_value_loss_coef : float = 0.5
    ppo_entropy_coef    : float = 0.01
    ppo_max_grad_norm   : float = 0.5

    gamma      : float = 0.99
    gae_lambda : float = 0.95

    clip_ratio              : float = 0.2
    value_clip_ratio        : float = 0.2
    value_loss_coef         : float = 0.5
    gradient_clip_max_norm  : float = 3.0
    kl_divergence_threshold : float = 0.015

@dataclass
class TrainingConfig:
    device                : str = "cuda"
    load_checkpoint       : bool = False
    resume_from_run       : Optional[str] = None
    max_steps_per_episode : int = 5
    batch_size            : int = 1024
    minibatch_size        : int = 128
    num_epochs            : int = 5
    print_frequency       : int = 5
    log_episode_frequency : int = 5
    use_mixed_precision   : bool = False
    large_negative_value  : float = -1e8
    verbose               : bool = False

@dataclass
class RewardConfig:
    distance_weight           : float = 1.5
    unassigned_penalty_weight : float = 1.0
    idle_penalty_weight       : float = 0.5
    priority_penalty_weight   : float = 0.5
    
    invalid_action_penalty    : float = 0.0
    add_job_penalty           : float = 0.5
    remove_job_penalty        : float = 1.5
    reoptimize_penalty        : float = -1.5
    no_action_penalty         : float = 0.0

@dataclass
class DeviceConfig:
    device: str = "cuda"
    
    @property
    def torch_device(self):
        return torch.device(self.device)

@dataclass
class Config:
    service    : ServiceConfig     = field(default_factory=ServiceConfig)
    io         : IOConfig          = field(default_factory=IOConfig)
    env        : EnvironmentConfig = field(default_factory=EnvironmentConfig)
    lr         : LearningRate      = field(default_factory=LearningRate)
    entropy    : Entropy           = field(default_factory=Entropy)
    model      : ModelConfig       = field(default_factory=ModelConfig)
    training   : TrainingConfig    = field(default_factory=TrainingConfig)
    reward     : RewardConfig      = field(default_factory=RewardConfig)
    ppo        : PPOConfig         = field(default_factory=PPOConfig)
    device     : DeviceConfig      = field(default_factory=DeviceConfig)

config = Config()


