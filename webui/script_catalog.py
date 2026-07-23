from dataclasses import fields, is_dataclass
from typing      import Union, get_args, get_origin

from configuration import Config


class ScriptCatalog:
    META = {
        "training.device"            : {"label": "Device", "choices": ["cuda", "cpu"]},
        "io.run_name"                : {"label": "Run name", "help": "Run directory under runs/. Empty picks a timestamped name."},
        "io.runs_dir"                : {"label": "Runs directory"},
        "io.dataset_dir"             : {"label": "Dataset directory"},
        "io.checkpoint_filename"     : {"label": "Checkpoint file"},
        "io.resume_from_run"         : {"label": "Resume from run", "help": "Continue this run with its optimizer state. Mutually exclusive with init."},
        "io.init_from_run"           : {"label": "Init from run", "help": "Warm-start weights from this run (e.g. a BC pretrain). Mutually exclusive with resume."},
        "io.dataset_num_events"      : {"label": "Total events"},
        "io.dataset_chunk_size"      : {"label": "Chunk size"},
        "io.dataset_batch_size"      : {"label": "Worker batch size"},
        "io.dataset_seed"            : {"label": "Seed"},
        "env.center"                 : {"label": "Center (lon, lat)"},
        "env.step_event_probability" : {"label": "Event probability", "help": "Chance of a disruption event firing on each episode step."},
        "service.vroom_url"          : {"label": "VROOM URL"},
        "service.osrm_url"           : {"label": "OSRM URL"},
        "evaluation.episodes"        : {"label": "Episodes"},
        "evaluation.seed"            : {"label": "Seed"},
        "env.tick_seconds"           : {"label": "Tick (s)", "help": "Simulated execution time between decision steps. Zero freezes time."},
        "env.repossession_success_probability" : {"label": "Repossession success", "help": "Chance a pickup finds the motorcycle."},
        "env.repossession_fraction"  : {"label": "Repossession share"},
        "env.depot_radius"           : {"label": "Depot radius (km)"},
        "env.depot_service"          : {"label": "Depot unload (s)"},
    }

    SCRIPTS = {
        "generate_dataset": {
            "title"      : "Generate Dataset",
            "group"      : "Data",
            "summary"    : "Sample scenarios, apply one disruption event each, and write chunked training data.",
            "essentials" : ["io.dataset_dir", "io.dataset_num_events", "io.dataset_seed"],
            "sections"   : [
                {"title": "Output",   "fields": ["io.dataset_dir", "io.dataset_num_events", "io.dataset_chunk_size", "io.dataset_batch_size", "io.dataset_seed"]},
                {"title": "Sampler",  "fields": ["env.mean_jobs", "env.std_jobs", "env.min_jobs", "env.mean_vehicles", "env.std_vehicles", "env.min_vehicles", "env.radius", "env.center", "env.depot_radius", "env.repossession_fraction", "env.support_service_min", "env.support_service_max", "env.repossession_service_min", "env.repossession_service_max", "env.outlier_frequency", "env.outlier_multiplier", "env.reset_max_attempts"]},
                {"title": "Events",   "fields": ["env.job_insert_min", "env.job_insert_max", "env.job_remove_min", "env.job_remove_max", "env.vehicle_insert_min", "env.vehicle_insert_max", "env.vehicle_remove_min", "env.vehicle_remove_max"]},
                {"title": "Services", "fields": ["service.vroom_url", "service.osrm_url"]},
            ],
        },
        "pretrain": {
            "title"      : "Pretrain (Behavior Cloning)",
            "group"      : "Training",
            "summary"    : "Clone the regret-insertion teacher into the policy as a warm start for PPO.",
            "essentials" : ["io.run_name", "io.dataset_dir", "pretrain.episodes", "pretrain.bc_epochs"],
            "sections"   : [
                {"title": "Run",         "fields": ["io.run_name", "io.runs_dir", "io.dataset_dir"]},
                {"title": "Pretraining", "fields": ["pretrain.episodes", "pretrain.bc_epochs", "pretrain.minibatch_size", "pretrain.lr", "pretrain.value_loss_coef", "pretrain.gradient_clip_max_norm", "pretrain.plan_horizon"]},
                {"title": "Episode",     "fields": ["training.max_steps_per_episode", "env.step_event_probability", "env.tick_seconds", "env.repossession_success_probability", "env.depot_service"]},
                {"title": "Sampler",     "fields": ["env.mean_jobs", "env.std_jobs", "env.mean_vehicles", "env.std_vehicles", "env.radius"]},
                {"title": "Reward",      "fields": ["reward.distance_weight", "reward.unassigned_penalty_weight", "reward.idle_penalty_weight", "reward.priority_penalty_weight", "reward.add_job_cost", "reward.remove_job_cost", "reward.no_action_cost", "reward.disruption_cost"]},
                {"title": "Compute",     "fields": ["training.device"]},
                {"title": "Services",    "fields": ["service.vroom_url", "service.osrm_url"]},
            ],
        },
        "train": {
            "title"      : "Train (PPO)",
            "group"      : "Training",
            "summary"    : "PPO over the chunked dataset with per-head learning rates and anchored fine-tuning.",
            "essentials" : ["io.run_name", "io.dataset_dir", "io.init_from_run"],
            "sections"   : [
                {"title": "Run",            "fields": ["io.run_name", "io.runs_dir", "io.dataset_dir", "io.resume_from_run", "io.init_from_run", "io.checkpoint_filename"]},
                {"title": "Training",       "fields": ["training.device", "training.max_steps_per_episode", "training.minibatch_size", "training.num_epochs", "training.use_mixed_precision", "training.print_frequency", "training.log_episode_frequency", "training.verbose"]},
                {"title": "Learning rates", "fields": ["lr.lr_operator_actor", "lr.lr_vehicle_actor", "lr.lr_job_actor", "lr.lr_critic", "lr.lr_embedding", "lr.lr_warmup_steps", "lr.lr_min", "lr.lr_decay_steps"]},
                {"title": "Entropy",        "fields": ["entropy.entropy_coef", "entropy.entropy_start", "entropy.entropy_end", "entropy.entropy_anneal_steps"]},
                {"title": "PPO",            "fields": ["ppo.gamma", "ppo.gae_lambda", "ppo.clip_ratio", "ppo.value_clip_ratio", "ppo.value_loss_coef", "ppo.gradient_clip_max_norm", "ppo.kl_divergence_threshold", "ppo.anchor_kl_start", "ppo.anchor_kl_end", "ppo.anchor_anneal_steps"]},
                {"title": "Reward",         "fields": ["reward.distance_weight", "reward.unassigned_penalty_weight", "reward.idle_penalty_weight", "reward.priority_penalty_weight", "reward.add_job_cost", "reward.remove_job_cost", "reward.no_action_cost", "reward.disruption_cost"]},
                {"title": "Events",         "fields": ["env.step_event_probability", "env.job_insert_min", "env.job_insert_max", "env.job_remove_min", "env.job_remove_max", "env.vehicle_insert_min", "env.vehicle_insert_max", "env.vehicle_remove_min", "env.vehicle_remove_max"]},
                {"title": "Execution",      "fields": ["env.tick_seconds", "env.repossession_success_probability", "env.depot_service"]},
                {"title": "Architecture",   "fields": ["model.gnn_num_layers", "model.policy_gnn_hidden_channels", "model.policy_embedding_dim", "model.operator_embedding_dim", "model.policy_actor_hidden_1", "model.policy_actor_hidden_2", "model.value_critic_hidden_1", "model.value_critic_hidden_2"]},
                {"title": "Services",       "fields": ["service.vroom_url", "service.osrm_url"]},
            ],
        },
        "evaluate": {
            "title"      : "Evaluate",
            "group"      : "Evaluation",
            "summary"    : "Compare the checkpoint against teacher, insertion-only, and do-nothing baselines.",
            "essentials" : ["io.run_name", "io.dataset_dir", "evaluation.episodes"],
            "sections"   : [
                {"title": "Run",        "fields": ["io.run_name", "io.runs_dir", "io.dataset_dir", "io.checkpoint_filename"]},
                {"title": "Evaluation", "fields": ["evaluation.episodes", "evaluation.seed", "training.max_steps_per_episode", "env.step_event_probability", "env.tick_seconds", "env.repossession_success_probability"]},
                {"title": "Reward",     "fields": ["reward.distance_weight", "reward.unassigned_penalty_weight", "reward.idle_penalty_weight", "reward.priority_penalty_weight", "reward.add_job_cost", "reward.remove_job_cost", "reward.no_action_cost", "reward.disruption_cost"]},
                {"title": "Compute",    "fields": ["training.device"]},
                {"title": "Services",   "fields": ["service.vroom_url", "service.osrm_url"]},
            ],
        },
        "infer": {
            "title"      : "Infer (random scenario)",
            "group"      : "Evaluation",
            "summary"    : "Run the checkpoint once on a random scenario and log the step table.",
            "essentials" : ["io.run_name"],
            "sections"   : [
                {"title": "Run",      "fields": ["io.run_name", "io.runs_dir", "io.checkpoint_filename"]},
                {"title": "Compute",  "fields": ["training.device"]},
                {"title": "Services", "fields": ["service.vroom_url", "service.osrm_url"]},
            ],
        },
    }

    def __init__(self):
        self.defaults = Config()
        self._validate()

    def has_script(self, key):
        return key in self.SCRIPTS

    def known_paths(self, key):
        script = self.SCRIPTS[key]
        return {path for section in script["sections"] for path in section["fields"]}

    def list_scripts(self):
        return [
            {
                "key"        : key,
                "title"      : script["title"],
                "group"      : script["group"],
                "summary"    : script["summary"],
                "essentials" : script["essentials"],
            }
            for key, script in self.SCRIPTS.items()
        ]

    def _field_spec(self, path):
        section_name, _, field_name = path.partition(".")
        section = getattr(self.defaults, section_name)
        spec    = {entry.name: entry for entry in fields(section) if entry.init}[field_name]
        return section, spec

    def _field_kind(self, annotation):
        origin = get_origin(annotation)

        if origin is Union:
            members = [arg for arg in get_args(annotation) if arg is not type(None)]
            return self._field_kind(members[0])

        if origin is tuple:
            return "tuple"
        if annotation is bool:
            return "bool"
        if annotation is int:
            return "int"
        if annotation is float:
            return "float"
        if annotation is str:
            return "str"

        raise ValueError(f"Field type {annotation!r} has no form widget")

    def _nullable(self, annotation):
        return get_origin(annotation) is Union and type(None) in get_args(annotation)

    def _field_entry(self, path):
        section, spec = self._field_spec(path)
        meta          = self.META.get(path, {})
        default       = getattr(section, spec.name)

        if isinstance(default, tuple):
            default = list(default)

        return {
            "path"     : path,
            "label"    : meta.get("label", spec.name.replace("_", " ").capitalize()),
            "help"     : meta.get("help"),
            "kind"     : self._field_kind(spec.type),
            "nullable" : self._nullable(spec.type),
            "choices"  : meta.get("choices"),
            "default"  : default,
        }

    def form(self, key):
        script = self.SCRIPTS[key]

        return {
            "key"        : key,
            "title"      : script["title"],
            "summary"    : script["summary"],
            "essentials" : script["essentials"],
            "command"    : f"main/{key}.py",
            "sections"   : [
                {"title": section["title"], "fields": [self._field_entry(path) for path in section["fields"]]}
                for section in script["sections"]
            ],
        }

    def _validate(self):
        for key in self.SCRIPTS:
            for path in self.known_paths(key):
                self._field_entry(path)

            for path in self.SCRIPTS[key]["essentials"]:
                if path not in self.known_paths(key):
                    raise ValueError(f"Essential field {path!r} of script {key!r} is not in any section")
