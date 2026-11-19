"""Declarative layout of the web UI launch forms, and its expansion into panels.

Holds the single source of truth for how each script's resolved configuration is
presented: which fields are essentials, how they group into sections and panels,
which widget renders each one, and which configuration sections an entry point
does not read. Every routing entry reads the one shared configuration, so a
layout names the sections it hides instead of showing all of them. Validation
refuses any layout that names an unknown field, claims one twice, or leaves a
field of a shown section unclaimed, so a new configuration field fails loudly
until a layout places it.
"""

from __future__ import annotations

import copy
from collections import Counter


class LayoutError(Exception):
    """Raised when a launch layout does not match the script's resolved config."""
    pass


class LaunchLayout:
    """Expands and validates the launch form layout of each routing entry point.

    Attributes:
        LAYOUTS : Declared layout per script key: essentials, sections of panels of field groups, and hidden configuration sections.
    """

    NUM_UPDATES   = {"kind": "number", "int": True, "min": 1, "max": 10000, "presets": [4, 8, 16, 32, 64]}
    NUM_EPISODES  = {"kind": "number", "int": True, "min": 1, "max": 1000000, "presets": [256, 512, 1024, 2048]}
    NUM_BATCH     = {"kind": "number", "int": True, "min": 1, "max": 4096, "presets": [64, 128, 256, 512]}
    NUM_EPOCHS    = {"kind": "number", "int": True, "min": 1, "max": 100, "presets": [1, 2, 4, 5, 8]}
    NUM_BC        = {"kind": "number", "int": True, "min": 1, "max": 1000000, "presets": [200, 1000, 2000, 5000]}
    NUM_EVAL      = {"kind": "number", "int": True, "min": 1, "max": 100000, "presets": [10, 50, 200, 1000]}
    NUM_DIAG      = {"kind": "number", "int": True, "min": 1, "max": 1000, "presets": [2, 5, 10, 20]}
    NUM_SEED      = {"kind": "number", "int": True, "min": 0, "max": 2147483647, "presets": [0, 42, 1000000, 2000000]}
    NUM_WORKERS   = {"kind": "number", "int": True, "min": 0, "max": 64, "presets": [0, 2, 4, 8]}
    NUM_ROUNDS    = {"kind": "number", "int": True, "min": 1, "max": 20, "presets": [1, 3, 5, 8]}
    NUM_EDITS     = {"kind": "number", "int": True, "min": 1, "max": 64, "presets": [4, 8, 16]}
    NUM_LR        = {"kind": "number", "log": True, "min": 1e-7, "max": 1e-1, "presets": [1e-4, 3e-4, 5e-4, 1e-3]}
    NUM_FRACTION  = {"kind": "number", "min": 0, "max": 1, "step": 0.05, "presets": [0, 0.05, 0.25, 0.5, 1]}

    CH_DEVICE     = {"kind": "choice", "options": ["cuda", "cpu"]}
    CH_VERDICT    = {"kind": "choice", "options": ["none", "insertions", "full"]}
    MULTI_AGENTS  = {"kind": "multi", "empty": "select at least one agent", "choices": [["model", "model"], ["teacher", "teacher"], ["insertion_only", "insertion only"], ["do_nothing", "do nothing"]]}
    PICK_RUN      = {"kind": "run"}
    NEED_RUN      = {"kind": "run", "required": True}

    GROUP_RUN       = {"title": "Run", "fields": ["io.runs_dir", "io.logdir", "io.checkpoint_filename"]}
    GROUP_SERVICES  = {"title": "Services", "fields": ["service.vroom_url", "service.osrm_url", "service.solve_cache_limit", "service.options"]}
    GROUP_CAPS      = {"title": "Episode", "fields": [{"path": "training.max_disruptions_per_episode", "widget": NUM_ROUNDS}, {"path": "training.max_edits_per_round", "widget": NUM_EDITS}]}
    GROUP_GAPS      = {"title": "Disruption gaps", "fields": ["env.disruption_gap_mean", "env.disruption_gap_min", "env.disruption_gap_max"]}
    GROUP_EVENTS    = {"title": "Disruption events", "fields": ["env.job_insert_min", "env.job_insert_max", "env.job_remove_min", "env.job_remove_max", "env.vehicle_insert_min", "env.vehicle_insert_max", "env.vehicle_remove_min", "env.vehicle_remove_max"]}
    GROUP_AREA      = {"title": "Area", "fields": ["env.center", "env.radius", "env.depot_radius", "env.depot_service"]}
    GROUP_JOBS      = {"title": "Jobs", "fields": ["env.min_jobs", "env.mean_jobs", "env.std_jobs", "env.outlier_frequency", "env.outlier_multiplier", "env.job_setup_min", "env.job_setup_max"]}
    GROUP_SERVICE_T = {"title": "Service times", "fields": ["env.support_service_min", "env.support_service_max", "env.repossession_fraction", "env.repossession_success_probability", "env.repossession_service_min", "env.repossession_service_max"]}
    GROUP_FLEET     = {"title": "Fleet", "fields": ["env.min_vehicles", "env.mean_vehicles", "env.std_vehicles"]}
    GROUP_SAMPLER   = {"title": "Sampler", "fields": [{"path": "env.scenario_seed", "widget": NUM_SEED}, "env.reset_max_attempts"]}
    GROUP_REWARD    = {"title": "Reward", "fields": ["reward.distance_weight", "reward.unassigned_penalty_weight", "reward.idle_penalty_weight", "reward.priority_penalty_weight", "reward.add_job_cost", "reward.remove_job_cost", "reward.no_action_cost", "reward.disruption_cost", "reward.holding_reference_seconds"]}
    GROUP_ARCH      = {"title": "Architecture", "fields": ["model.gnn_num_layers", "model.policy_gnn_hidden_channels", "model.policy_embedding_dim", "model.operator_embedding_dim", "model.context_embedding_dim", "model.policy_actor_hidden_1", "model.policy_actor_hidden_2", "model.value_critic_hidden_1", "model.value_critic_hidden_2"]}
    GROUP_MONITOR   = {"title": "Resource monitor", "fields": ["monitor.enabled", "monitor.poll_interval_sec", "monitor.log_to_tensorboard", "monitor.warn_ram_pct", "monitor.warn_vram_pct", "monitor.warn_swap_pct", "monitor.warn_shm_pct", "monitor.warn_cooldown_sec"]}
    GROUP_TELEMETRY = {"title": "Telemetry", "fields": ["telemetry.step_every", "telemetry.episode_every", "telemetry.sample_every", "telemetry.layer_gradients_every"]}
    GROUP_LR        = {"title": "Learning rates", "fields": [
        {"path": "lr.lr_operator_actor",     "widget": NUM_LR},
        {"path": "lr.lr_vehicle_actor",      "widget": NUM_LR},
        {"path": "lr.lr_job_actor",          "widget": NUM_LR},
        {"path": "lr.lr_critic",             "widget": NUM_LR},
        {"path": "lr.lr_embedding",          "widget": NUM_LR},
        {"path": "lr.lr_operator_embedding", "widget": NUM_LR},
        {"path": "lr.lr_min",                "widget": NUM_LR},
        "lr.lr_warmup_fraction",
        "lr.lr_decay_fraction",
    ]}
    GROUP_ENTROPY   = {"title": "Entropy", "fields": ["entropy.entropy_start", "entropy.entropy_end", "entropy.entropy_anneal_fraction"]}
    GROUP_PPO       = {"title": "PPO", "fields": ["ppo.gamma", "ppo.gae_lambda", "ppo.clip_ratio", "ppo.value_loss_coef", "ppo.gradient_clip_max_norm", "ppo.kl_divergence_threshold", "ppo.anchor_kl_start", "ppo.anchor_kl_end", "ppo.anchor_anneal_fraction"]}

    MODEL_FIXED    = ["model.job_input_dim", "model.vehicle_input_dim", "model.edge_attr_dim", "model.context_input_dim", "model.num_operators"]
    TRAINING_LOOP  = ["training.num_updates", "training.episodes_per_update", "training.minibatch_size", "training.num_epochs", "training.expected_edits_per_round", "training.use_mixed_precision"]
    TRAINING_FIXED = ["training.large_negative_value"]
    PPO_UNUSED     = ["ppo.gae_lambda", "ppo.clip_ratio", "ppo.value_loss_coef", "ppo.gradient_clip_max_norm", "ppo.kl_divergence_threshold", "ppo.anchor_kl_start", "ppo.anchor_kl_end", "ppo.anchor_anneal_fraction"]

    EPISODE_SECTION   = {"key": "episode", "title": "Episode", "panels": [{"kind": "fields", "groups": [GROUP_CAPS, GROUP_GAPS, GROUP_EVENTS]}]}
    SCENARIO_SECTION  = {"key": "scenarios", "title": "Scenarios", "panels": [{"kind": "fields", "groups": [GROUP_AREA, GROUP_JOBS, GROUP_SERVICE_T, GROUP_FLEET, GROUP_SAMPLER]}]}
    REWARD_SECTION    = {"key": "reward", "title": "Reward", "panels": [{"kind": "fields", "groups": [GROUP_REWARD]}]}
    SERVICES_SECTION  = {"key": "services", "title": "Services", "panels": [{"kind": "fields", "groups": [GROUP_SERVICES]}]}
    MODEL_SECTION     = {"key": "model", "title": "Model", "panels": [{"kind": "fields", "groups": [GROUP_ARCH]}, {"kind": "hidden", "fields": MODEL_FIXED}]}
    COMPUTE_SECTION   = {"key": "compute", "title": "Compute", "panels": [{"kind": "fields", "groups": [{"title": "Device", "fields": [{"path": "training.device", "widget": CH_DEVICE}]}]}, {"kind": "hidden", "fields": TRAINING_LOOP + TRAINING_FIXED}]}
    MONITORING_GROUPS = [GROUP_MONITOR, GROUP_TELEMETRY]

    LAYOUTS = {
        "pretrain": {
            "essentials": ["io.run_name", {"path": "pretrain.episodes", "widget": NUM_BC}, {"path": "pretrain.bc_epochs", "widget": NUM_EPOCHS}],
            "sections": [
                {"key": "run", "title": "Run", "panels": [{"kind": "fields", "groups": [GROUP_RUN]}, {"kind": "hidden", "fields": ["io.resume_from_run", "io.init_from_run"]}]},
                {"key": "pretraining", "title": "Pretraining", "panels": [{"kind": "fields", "groups": [
                    {"title": "Cloning", "fields": [{"path": "pretrain.minibatch_size", "widget": NUM_BATCH}, {"path": "pretrain.lr", "widget": NUM_LR}, "pretrain.value_loss_coef", "pretrain.gradient_clip_max_norm", "pretrain.plan_horizon"]},
                    {"title": "Teacher rollouts", "fields": [{"path": "pretrain.collect_workers", "widget": NUM_WORKERS}, "pretrain.collect_batch_size"]},
                    {"title": "Returns", "fields": ["ppo.gamma"]},
                ]}, {"kind": "hidden", "fields": PPO_UNUSED}]},
                EPISODE_SECTION,
                SCENARIO_SECTION,
                REWARD_SECTION,
                MODEL_SECTION,
                {"key": "compute", "title": "Compute", "panels": [{"kind": "fields", "groups": [{"title": "Device", "fields": [{"path": "training.device", "widget": CH_DEVICE}]}] + MONITORING_GROUPS}, {"kind": "hidden", "fields": TRAINING_LOOP + TRAINING_FIXED}]},
                SERVICES_SECTION,
            ],
            "hidden_sections": ["lr", "entropy", "evaluation", "inference", "diagnostics"],
        },
        "train": {
            "essentials": ["io.run_name", {"path": "training.num_updates", "widget": NUM_UPDATES}, {"path": "io.init_from_run", "widget": PICK_RUN}],
            "sections": [
                {"key": "run", "title": "Run", "panels": [{"kind": "fields", "groups": [GROUP_RUN, {"title": "Start from", "fields": [{"path": "io.resume_from_run", "widget": PICK_RUN}]}]}]},
                {"key": "training", "title": "Training", "panels": [{"kind": "fields", "groups": [
                    {"title": "Loop", "fields": [{"path": "training.episodes_per_update", "widget": NUM_EPISODES}, {"path": "training.minibatch_size", "widget": NUM_BATCH}, {"path": "training.num_epochs", "widget": NUM_EPOCHS}, "training.expected_edits_per_round"]},
                    {"title": "Compute", "fields": [{"path": "training.device", "widget": CH_DEVICE}, "training.use_mixed_precision"]},
                ]}, {"kind": "hidden", "fields": TRAINING_FIXED}]},
                {"key": "optimisation", "title": "Optimisation", "panels": [{"kind": "fields", "groups": [GROUP_LR, GROUP_ENTROPY, GROUP_PPO]}]},
                EPISODE_SECTION,
                SCENARIO_SECTION,
                REWARD_SECTION,
                MODEL_SECTION,
                {"key": "monitoring", "title": "Monitoring", "panels": [{"kind": "fields", "groups": MONITORING_GROUPS}]},
                SERVICES_SECTION,
            ],
            "hidden_sections": ["pretrain", "evaluation", "inference", "diagnostics"],
        },
        "evaluate": {
            "essentials": [{"path": "io.run_name", "widget": NEED_RUN}, {"path": "evaluation.episodes", "widget": NUM_EVAL}],
            "sections": [
                {"key": "run", "title": "Run", "panels": [{"kind": "fields", "groups": [GROUP_RUN]}, {"kind": "hidden", "fields": ["io.resume_from_run", "io.init_from_run"]}]},
                {"key": "evaluation", "title": "Evaluation", "panels": [{"kind": "fields", "groups": [{"title": "Held-out set", "fields": [{"path": "evaluation.seed", "widget": NUM_SEED}, "evaluation.greedy"]}]}]},
                EPISODE_SECTION,
                SCENARIO_SECTION,
                REWARD_SECTION,
                COMPUTE_SECTION,
                SERVICES_SECTION,
            ],
            "hidden_sections": ["lr", "entropy", "model", "ppo", "pretrain", "telemetry", "inference", "monitor", "diagnostics"],
        },
        "infer": {
            "essentials": [{"path": "io.run_name", "widget": NEED_RUN}, {"path": "inference.seed", "widget": NUM_SEED}],
            "sections": [
                {"key": "run", "title": "Run", "panels": [{"kind": "fields", "groups": [GROUP_RUN]}, {"kind": "hidden", "fields": ["io.resume_from_run", "io.init_from_run"]}]},
                {"key": "inference", "title": "Inference", "panels": [{"kind": "fields", "groups": [{"title": "Policy", "fields": ["inference.greedy"]}]}]},
                EPISODE_SECTION,
                SCENARIO_SECTION,
                REWARD_SECTION,
                COMPUTE_SECTION,
                SERVICES_SECTION,
            ],
            "hidden_sections": ["lr", "entropy", "model", "ppo", "pretrain", "evaluation", "telemetry", "monitor", "diagnostics"],
        },
        "diagnose": {
            "essentials": [{"path": "io.run_name", "widget": NEED_RUN}, {"path": "diagnostics.seed", "widget": NUM_SEED}, {"path": "diagnostics.episodes", "widget": NUM_DIAG}],
            "sections": [
                {"key": "run", "title": "Run", "panels": [{"kind": "fields", "groups": [GROUP_RUN]}, {"kind": "hidden", "fields": ["io.resume_from_run", "io.init_from_run"]}]},
                {"key": "diagnostics", "title": "Diagnostics", "panels": [{"kind": "fields", "groups": [
                    {"title": "Agents", "fields": [{"path": "diagnostics.agents", "widget": MULTI_AGENTS}, "diagnostics.greedy", "diagnostics.sampled_twin"]},
                    {"title": "Teacher", "fields": [{"path": "diagnostics.teacher_verdict", "widget": CH_VERDICT}]},
                ]}]},
                EPISODE_SECTION,
                SCENARIO_SECTION,
                REWARD_SECTION,
                COMPUTE_SECTION,
                SERVICES_SECTION,
            ],
            "hidden_sections": ["lr", "entropy", "model", "ppo", "pretrain", "evaluation", "telemetry", "inference", "monitor"],
        },
    }

    def _field_entry(self, item, widgets):
        """Expands one declared field into an entry, registering its widget.

        Args:
            item    : Declared field: a dotted path, a gate wrapper, or a dict with ``path`` and an optional ``widget``.
            widgets : Widget registry, extended in place with path to widget spec.

        Returns:
            The expanded entry dict.
        """
        if isinstance(item, str):
            return {"path": item}

        if "gate" in item:
            return {"gate": item["gate"], "fields": [self._field_entry(sub, widgets) for sub in item["fields"]]}

        if "widget" in item:
            widgets[item["path"]] = item["widget"]
        return {"path": item["path"]}

    def _expand_groups(self, groups, widgets):
        """Expands every field of every group, preserving declaration order."""
        return [{"title": group.get("title"), "fields": [self._field_entry(item, widgets) for item in group["fields"]]} for group in groups]

    def _expand_panel(self, panel, widgets):
        """Expands one panel: hidden panels pass their field list through, field panels expand their groups."""
        if panel["kind"] == "hidden":
            return {"kind": "hidden", "fields": list(panel["fields"])}

        return {"kind": "fields", "title": panel.get("title"), "note": panel.get("note"), "groups": self._expand_groups(panel["groups"], widgets)}

    def _expand(self, key):
        """Expands a declared layout into essentials, sections, the hidden sections and a widget map.

        Args:
            key : Script key whose layout is expanded.

        Returns:
            Layout dict with ``essentials``, ``sections``, ``hidden_sections`` and ``widgets``.
        """
        spec    = self.LAYOUTS[key]
        widgets = {}

        essentials = [self._field_entry(item, widgets) for item in spec.get("essentials", [])]

        sections = []
        for section in spec["sections"]:
            expanded = {"key": section["key"], "title": section["title"], "panels": [self._expand_panel(panel, widgets) for panel in section["panels"]]}
            if "when" in section:
                expanded["when"] = copy.deepcopy(section["when"])
            sections.append(expanded)

        return {"essentials": essentials, "sections": sections, "hidden_sections": list(spec.get("hidden_sections", [])), "widgets": widgets}

    def _entry_claims(self, entry, out):
        """Collects the config paths an expanded entry exposes, gates included."""
        if "gate" in entry:
            out.append(entry["gate"])
            for sub in entry["fields"]:
                self._entry_claims(sub, out)
            return
        out.append(entry["path"])

    def _when_conditions(self, when):
        """Returns a ``when`` clause as a list, accepting a single condition or none."""
        if when is None:
            return []
        return list(when) if isinstance(when, list) else [when]

    def _claims(self, layout):
        """Returns every config path the layout claims, duplicates kept, hidden panels included."""
        claimed = []

        for entry in layout["essentials"]:
            self._entry_claims(entry, claimed)

        for section in layout["sections"]:
            for panel in section["panels"]:
                if panel["kind"] == "hidden":
                    claimed.extend(panel["fields"])
                    continue

                for group in panel["groups"]:
                    for entry in group["fields"]:
                        self._entry_claims(entry, claimed)

        return claimed

    def _validate(self, key, layout, leaves):
        """Checks an expanded layout against the resolved config leaves.

        Every leaf outside the hidden sections must be claimed exactly once,
        no leaf of a hidden section may be claimed, every claimed path and
        hidden section must exist, section gates must reference known fields
        through an ``in`` list, and number widgets must carry bounds.

        Args:
            key    : Script key being validated, used in the error message.
            layout : Expanded layout dict.
            leaves : Resolved config leaves, each carrying a dotted ``path`` and its ``section``.

        Raises:
            LayoutError : With every problem found, one per line.
        """
        paths    = [leaf["path"] for leaf in leaves]
        known    = set(paths)
        sections = {leaf["section"] for leaf in leaves}
        hidden   = set(layout["hidden_sections"])
        claimed  = self._claims(layout)

        counts    = Counter(claimed)
        claim_set = set(counts)

        duplicates = sorted(path for path, count in counts.items() if count > 1)
        unknown    = sorted(claim_set - known)
        concealed  = sorted(leaf["path"] for leaf in leaves if leaf["section"] in hidden and leaf["path"] in claim_set)
        unclaimed  = [leaf["path"] for leaf in leaves if leaf["section"] not in hidden and leaf["path"] not in claim_set]
        missing    = sorted(hidden - sections)

        problems = []
        if unknown:
            problems.append(f"layout for {key} names unknown fields: {', '.join(unknown)}")
        if duplicates:
            problems.append(f"layout for {key} claims fields twice: {', '.join(duplicates)}")
        if concealed:
            problems.append(f"layout for {key} claims fields of hidden sections: {', '.join(concealed)}")
        if unclaimed:
            problems.append(f"layout for {key} leaves fields unclaimed: {', '.join(unclaimed)}")
        if missing:
            problems.append(f"layout for {key} hides unknown sections: {', '.join(missing)}")

        for section in layout["sections"]:
            for condition in self._when_conditions(section.get("when")):
                if condition["field"] not in known:
                    problems.append(f"section {section['key']} gates on unknown field {condition['field']}")
                if "in" not in condition:
                    problems.append(f"section {section['key']} has a when condition without an 'in' list")

        for path, widget in layout["widgets"].items():
            if widget.get("kind") == "number" and not ("min" in widget and "max" in widget):
                problems.append(f"number widget for {path} lacks min/max bounds")

        if problems:
            raise LayoutError("\n".join(problems))

    def build(self, key, leaves):
        """Builds and validates the launch form layout for a script.

        Args:
            key    : Script key to lay out.
            leaves : Resolved config leaves, each carrying a dotted ``path`` and its ``section``.

        Returns:
            The expanded layout with a ``mode`` of ``single`` when it holds one section and no essentials, otherwise ``sections``.

        Raises:
            LayoutError : If no layout is declared for the key, or the declared layout does not match the resolved config.
        """
        if key not in self.LAYOUTS:
            raise LayoutError(f"no launch layout declared for {key}")

        layout = self._expand(key)
        self._validate(key, layout, leaves)

        layout["mode"] = "single" if len(layout["sections"]) == 1 and not layout["essentials"] else "sections"
        return layout
