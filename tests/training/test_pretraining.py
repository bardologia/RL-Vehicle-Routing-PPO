import os

import pytest
import torch

from core.dataset import ChunkStore, Dataset, generate_events
from core.shared import ActionMasker, RoutingState
from core.training import BCTrainer, PPOMemory, PretrainingPipeline, RegretInsertionTeacher, TeacherRolloutCollector, Trainer
from model.policy_model import Action, Policy, PolicyCheckpoint
from tools.logger import NullLogger
from tools.telemetry import PPOTelemetry
from tools.tracker import NullTracker
from tests.conftest import make_jobs, make_route, make_vehicles


def load_scenario(environment, jobs, vehicles, state):
    environment.load_from_dataset({
        "jobs"     : [job.to_dict() for job in jobs],
        "vehicles" : [vehicle.to_dict() for vehicle in vehicles],
        "state"    : state.to_payload(),
    })


def test_teacher_inserts_unassigned_job_into_open_vehicle(environment, fake_vroom):
    jobs     = make_jobs(2)
    vehicles = make_vehicles(1)
    state    = RoutingState(routes=[make_route(vehicles[0], jobs[:1], cost=100)], unassigned_ids={jobs[1].id})

    load_scenario(environment, jobs, vehicles, state)

    teacher = RegretInsertionTeacher(environment.config)
    action  = teacher.select_action(environment, environment.current_state)

    assert action.operator == 0
    assert action.vehicle_index == 0
    assert action.job_index == environment.jobs.index_of(jobs[1].id)


def test_teacher_no_ops_when_everything_is_assigned(environment, fake_vroom):
    jobs     = make_jobs(2)
    vehicles = make_vehicles(1)
    state    = RoutingState(routes=[make_route(vehicles[0], jobs, cost=200)], unassigned_ids=set())

    load_scenario(environment, jobs, vehicles, state)

    teacher = RegretInsertionTeacher(environment.config)
    action  = teacher.select_action(environment, environment.current_state)

    assert action.operator == 2
    assert action.vehicle_index == 0
    assert action.job_index == 0


def test_teacher_reoptimizes_when_margin_allows_it(environment, fake_vroom):
    jobs     = make_jobs(2)
    vehicles = make_vehicles(1)
    state    = RoutingState(routes=[make_route(vehicles[0], jobs, cost=200)], unassigned_ids=set())

    load_scenario(environment, jobs, vehicles, state)
    environment.config.pretrain.reoptimize_margin = -5.0

    teacher = RegretInsertionTeacher(environment.config)
    action  = teacher.select_action(environment, environment.current_state)

    assert action.operator == 3


def test_teacher_margin_override_beats_config_margin(environment, fake_vroom):
    import math

    jobs     = make_jobs(2)
    vehicles = make_vehicles(1)
    state    = RoutingState(routes=[make_route(vehicles[0], jobs, cost=200)], unassigned_ids=set())

    load_scenario(environment, jobs, vehicles, state)
    environment.config.pretrain.reoptimize_margin = -5.0

    teacher = RegretInsertionTeacher(environment.config, reoptimize_margin=math.inf)
    action  = teacher.select_action(environment, environment.current_state)

    assert action.operator == 2


def test_teacher_prefers_higher_priority_job_under_scarce_capacity(environment, fake_vroom):
    jobs     = make_jobs(3)
    vehicles = make_vehicles(1)
    state    = RoutingState(routes=[make_route(vehicles[0], jobs[:1], cost=100)], unassigned_ids={jobs[1].id, jobs[2].id})

    load_scenario(environment, jobs, vehicles, state)

    assert jobs[2].priority > jobs[1].priority

    teacher = RegretInsertionTeacher(environment.config)
    action  = teacher.select_action(environment, environment.current_state)

    assert action.operator == 0
    assert action.job_index == environment.jobs.index_of(jobs[2].id)


def test_best_insertion_ignores_unprofitable_high_regret_jobs(cpu_config):
    teacher = RegretInsertionTeacher(cpu_config)
    options = {7: [(-3.0, 0)], 8: [(2.0, 0), (1.0, 1)]}

    best = teacher.best_insertion(options, baseline=0.0)

    assert best["job_id"] == 8
    assert best["vehicle_id"] == 0
    assert best["reward"] == 2.0


def test_best_insertion_returns_none_when_nothing_profitable(cpu_config):
    teacher = RegretInsertionTeacher(cpu_config)
    options = {7: [(-3.0, 0)], 8: [(-0.5, 0), (-1.0, 1)]}

    assert teacher.best_insertion(options, baseline=0.0) is None


def test_insertion_plan_value_grows_with_horizon(environment, fake_vroom):
    jobs     = make_jobs(2)
    vehicles = make_vehicles(1)
    state    = RoutingState(routes=[], unassigned_ids={jobs[0].id, jobs[1].id})

    load_scenario(environment, jobs, vehicles, state)

    teacher    = RegretInsertionTeacher(environment.config)
    short_plan = teacher.insertion_plan(environment, environment.current_state, horizon=1, baseline=0.0)
    long_plan  = teacher.insertion_plan(environment, environment.current_state, horizon=3, baseline=0.0)

    assert short_plan["action"] is not None
    assert long_plan["value"] > short_plan["value"]
    assert long_plan["action"].operator == short_plan["action"].operator == 0
    assert long_plan["action"].job_index == short_plan["action"].job_index


def test_teacher_prefers_sequential_insertions_over_reoptimize(environment, fake_vroom):
    jobs     = make_jobs(2)
    vehicles = make_vehicles(2)
    state    = RoutingState(routes=[], unassigned_ids={jobs[0].id, jobs[1].id})

    load_scenario(environment, jobs, vehicles, state)

    teacher = RegretInsertionTeacher(environment.config)
    action  = teacher.select_action(environment, environment.current_state)

    assert action.operator == 0


def test_teacher_removes_job_when_removal_pays(environment, fake_vroom):
    import math

    jobs     = make_jobs(2)
    vehicles = make_vehicles(1)
    state    = RoutingState(routes=[make_route(vehicles[0], jobs, cost=5000)], unassigned_ids=set())

    load_scenario(environment, jobs, vehicles, state)

    teacher = RegretInsertionTeacher(environment.config, reoptimize_margin=math.inf)
    action  = teacher.select_action(environment, environment.current_state)

    assert action.operator == 1
    assert action.vehicle_index == 0
    assert action.job_index in {environment.jobs.index_of(jobs[0].id), environment.jobs.index_of(jobs[1].id)}


def test_teacher_removal_disabled_falls_back_to_no_op(environment, fake_vroom):
    import math

    jobs     = make_jobs(2)
    vehicles = make_vehicles(1)
    state    = RoutingState(routes=[make_route(vehicles[0], jobs, cost=5000)], unassigned_ids=set())

    load_scenario(environment, jobs, vehicles, state)

    teacher = RegretInsertionTeacher(environment.config, reoptimize_margin=math.inf, allow_removal=False)
    action  = teacher.select_action(environment, environment.current_state)

    assert action.operator == 2


def test_collector_records_have_discounted_returns(cpu_config, seeded, fake_vroom):
    cpu_config.training.max_steps_per_episode = 2
    cpu_config.env.step_event_probability     = 0.0

    items, _ = generate_events(batch_size=1, seed=7, config=cpu_config)

    from core.shared import Environment
    environment = Environment(cpu_config)
    teacher     = RegretInsertionTeacher(cpu_config)
    collector   = TeacherRolloutCollector(environment, teacher, cpu_config)

    records = collector.rollout(items[0])
    gamma   = cpu_config.ppo.gamma

    assert len(records) == 2
    assert set(records[0].keys()) == {"graph", "mask_info", "action", "reward", "return"}
    assert records[1]["return"] == pytest.approx(records[1]["reward"])
    assert records[0]["return"] == pytest.approx(records[0]["reward"] + gamma * records[1]["reward"])


def test_bc_trainer_reaches_full_accuracy_on_fixed_teacher_action(environment, fake_vroom, cpu_config):
    jobs     = make_jobs(2)
    vehicles = make_vehicles(1)
    state    = RoutingState(routes=[make_route(vehicles[0], jobs, cost=200)], unassigned_ids=set())

    load_scenario(environment, jobs, vehicles, state)
    graph, mask_info = environment.observe()

    records = [
        {
            "graph"     : PPOMemory._clone_detached(graph),
            "mask_info" : mask_info,
            "action"    : Action(operator=2, vehicle_index=0, job_index=0),
            "reward"    : 0.0,
            "return"    : 0.0,
        }
        for _ in range(8)
    ]

    cpu_config.pretrain.bc_epochs      = 30
    cpu_config.pretrain.minibatch_size = 8
    cpu_config.pretrain.lr             = 0.01

    policy  = Policy(cpu_config)
    trainer = BCTrainer(policy, ActionMasker(cpu_config), cpu_config, PPOTelemetry(NullTracker(), cpu_config))
    metrics = trainer.train(records)

    assert metrics["accuracy"]["operator"] == 1.0
    assert metrics["accuracy"]["vehicle"] == 1.0
    assert metrics["accuracy"]["job"] == 1.0
    assert torch.isfinite(torch.tensor(metrics["loss"]))


def test_pretraining_pipeline_saves_policy_checkpoint(cpu_config, seeded, fake_vroom, tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    items, _ = generate_events(batch_size=3, seed=11, config=cpu_config)
    ChunkStore(str(data_dir)).save(items, 0)

    cpu_config.io.runs_dir                    = str(tmp_path / "runs")
    cpu_config.io.run_name                    = "pretrain_run"
    cpu_config.io.dataset_dir                 = str(data_dir)
    cpu_config.pretrain.episodes              = 2
    cpu_config.pretrain.bc_epochs             = 1
    cpu_config.pretrain.minibatch_size        = 8
    cpu_config.training.max_steps_per_episode = 2
    cpu_config.env.step_event_probability     = 0.0

    pipeline = PretrainingPipeline(cpu_config, repo_root=str(tmp_path))
    metrics  = pipeline.run()
    pipeline.session.writer.close()

    checkpoint_path = os.path.join(str(tmp_path / "runs" / "pretrain_run"), cpu_config.io.checkpoint_filename)
    checkpoint      = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    assert checkpoint["training_state"]["phase"] == "pretraining"
    assert checkpoint["training_state"]["episodes"] == 2
    assert checkpoint["training_state"]["records"] == 4
    assert set(metrics.keys()) == {"loss", "accuracy"}


def test_trainer_init_from_run_loads_pretrained_weights(cpu_config, seeded, fake_vroom, tmp_path):
    source = Policy(cpu_config)
    with torch.no_grad():
        for parameter in source.parameters():
            parameter.add_(1.0)

    PolicyCheckpoint().save(source, cpu_config.io.checkpoint_filename, str(tmp_path / "runs" / "pre"), training_state={"phase": "pretraining"})

    cpu_config.io.runs_dir      = str(tmp_path / "runs")
    cpu_config.io.init_from_run = "pre"
    cpu_config.io.logdir        = str(tmp_path / "logs")

    dataset = Dataset(dataset_dir=str(tmp_path / "empty"), config=cpu_config)
    trainer = Trainer(dataset=dataset, config=cpu_config, logger=NullLogger(), tracker=NullTracker())

    for original, loaded in zip(source.state_dict().values(), trainer.ppo.policy.state_dict().values()):
        assert torch.equal(original, loaded)


def test_trainer_rejects_init_and_resume_together(cpu_config, seeded, fake_vroom, tmp_path):
    cpu_config.io.logdir          = str(tmp_path)
    cpu_config.io.init_from_run   = "pre"
    cpu_config.io.resume_from_run = "old"

    dataset = Dataset(dataset_dir=str(tmp_path / "empty"), config=cpu_config)

    with pytest.raises(ValueError):
        Trainer(dataset=dataset, config=cpu_config, logger=NullLogger(), tracker=NullTracker())
