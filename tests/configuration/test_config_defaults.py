from configuration.device      import DeviceConfig
from configuration.environment import EnvironmentConfig
from configuration.io          import IOConfig
from configuration.learning    import Entropy, LearningRate
from configuration.model       import ModelConfig
from configuration.monitor     import MonitorConfig
from configuration.ppo         import PPOConfig
from configuration.pretraining import PretrainConfig
from configuration.reward      import RewardConfig
from configuration.service     import ServiceConfig
from configuration.telemetry   import TelemetryConfig
from configuration.training    import TrainingConfig


def test_device_config_defaults():
    config = DeviceConfig()

    assert config.device == "cuda"


def test_environment_config_defaults():
    config = EnvironmentConfig()

    assert config.center == (-46.63, -23.55)
    assert config.radius == 25.0
    assert config.min_vehicles == 2
    assert config.min_jobs == 4
    assert config.mean_jobs == 16
    assert config.mean_vehicles == 4
    assert config.reset_max_attempts == 32
    assert config.outlier_frequency == 8
    assert config.job_insert_min == 2
    assert config.job_insert_max == 3
    assert config.vehicle_remove_min == 1
    assert config.step_event_probability == 0.3


def test_io_config_defaults():
    config = IOConfig()

    assert config.runs_dir == "runs"
    assert config.run_name is None
    assert config.logdir is None
    assert config.dataset_dir == "datasets/chunked"
    assert config.checkpoint_filename == "graph_ppo_policy.pt"
    assert config.resume_from_run is None
    assert config.init_from_run is None
    assert config.dataset_num_events == 1024000
    assert config.dataset_chunk_size == 1024
    assert config.dataset_batch_size == 128
    assert config.dataset_seed == 42


def test_learning_rate_defaults():
    config = LearningRate()

    assert config.lr_operator_actor == 3e-4
    assert config.lr_vehicle_actor == 3e-4
    assert config.lr_critic == 5e-4
    assert config.lr_embedding == 2e-4
    assert config.lr_job_actor == 3e-4
    assert config.lr_warmup_steps == 1000
    assert config.lr_min == 1e-5
    assert config.lr_decay_steps == 100000


def test_entropy_defaults():
    config = Entropy()

    assert config.entropy_coef == 0.02
    assert config.entropy_start == 0.02
    assert config.entropy_end == 0.001
    assert config.entropy_anneal_steps == 50000


def test_model_config_defaults():
    config = ModelConfig()

    assert config.job_input_dim == 7
    assert config.vehicle_input_dim == 7
    assert config.edge_attr_dim == 4
    assert config.num_operators == 4
    assert config.operator_embedding_dim == 32
    assert config.gnn_num_layers == 2
    assert config.policy_gnn_hidden_channels == 64
    assert config.policy_embedding_dim == 64
    assert config.policy_actor_hidden_1 == 128
    assert config.policy_actor_hidden_2 == 64
    assert config.value_critic_hidden_1 == 64
    assert config.value_critic_hidden_2 == 64


def test_monitor_config_defaults():
    config = MonitorConfig()

    assert config.enabled is True
    assert config.poll_interval_sec == 10.0
    assert config.log_to_tensorboard is True
    assert config.warn_ram_pct == 90.0
    assert config.warn_vram_pct == 90.0
    assert config.warn_swap_pct == 50.0
    assert config.warn_shm_pct == 80.0
    assert config.warn_cooldown_sec == 30.0


def test_ppo_config_defaults():
    config = PPOConfig()

    assert config.gamma == 0.99
    assert config.gae_lambda == 0.95
    assert config.clip_ratio == 0.2
    assert config.value_clip_ratio == 0.2
    assert config.value_loss_coef == 0.5
    assert config.gradient_clip_max_norm == 3.0
    assert config.kl_divergence_threshold == 0.015


def test_pretrain_config_defaults():
    config = PretrainConfig()

    assert config.episodes == 2000
    assert config.bc_epochs == 4
    assert config.minibatch_size == 128
    assert config.lr == 3e-4
    assert config.value_loss_coef == 0.5
    assert config.gradient_clip_max_norm == 3.0
    assert config.reoptimize_margin == 0.0


def test_reward_config_defaults():
    config = RewardConfig()

    assert config.distance_weight == 1.5
    assert config.unassigned_penalty_weight == 1.0
    assert config.idle_penalty_weight == 0.5
    assert config.priority_penalty_weight == 0.5
    assert config.add_job_cost == 0.1
    assert config.remove_job_cost == 0.1
    assert config.no_action_cost == 0.0
    assert config.reoptimize_cost == 0.5
    assert config.disruption_cost == 0.3


def test_service_config_defaults():
    config = ServiceConfig()

    assert config.vroom_url == "http://localhost:3000"
    assert config.osrm_url == "http://localhost:5000"
    assert config.options == {"g": True, "geometry": True, "threads": 8}
    assert config._http_session is None


def test_telemetry_config_defaults():
    config = TelemetryConfig()

    assert config.step_every == 8
    assert config.episode_every == 4
    assert config.sample_every == 32
    assert config.layer_gradients_every == 50


def test_training_config_defaults():
    config = TrainingConfig()

    assert config.device == "cuda"
    assert config.max_steps_per_episode == 5
    assert config.minibatch_size == 128
    assert config.num_epochs == 5
    assert config.print_frequency == 5
    assert config.log_episode_frequency == 5
    assert config.use_mixed_precision is False
    assert config.large_negative_value == -1e8
    assert config.verbose is False
