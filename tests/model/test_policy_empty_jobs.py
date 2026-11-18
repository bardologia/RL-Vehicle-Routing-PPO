import math

import torch

from core.shared import ActionMaskBuilder, EntityPool, Graph, RoutingState
from core.training import Trainer
from model.policy_model import Policy
from tools.logger import NullLogger
from tools.tracker import Tracker
from tests.conftest import FakeWriter, make_vehicles


def empty_observation(cpu_config):
    jobs      = EntityPool([])
    vehicles  = EntityPool(make_vehicles(2))
    state     = RoutingState()
    graph     = Graph(cpu_config).build(jobs, vehicles, state, (-46.63, -23.55), 28800)
    mask_info = ActionMaskBuilder().build(jobs, vehicles, state)
    return graph, mask_info


def test_select_action_returns_do_nothing_when_the_job_pool_is_empty(cpu_config, seeded):
    policy           = Policy(cpu_config)
    graph, mask_info = empty_observation(cpu_config)

    for greedy in (True, False):
        for mask in (mask_info, None):
            result = policy.select_action(graph, mask_info=mask, greedy=greedy)

            assert result["action"].operator == 2
            assert result["action"].vehicle_index == 0
            assert result["action"].job_index == 0
            assert float(result["log_prob_vehicle"]) == 0.0
            assert float(result["log_prob_job"]) == 0.0
            assert math.isfinite(float(result["log_prob_operator"]))
            assert math.isfinite(float(result["state_value"]))
            assert result["old_job_logits"].shape == (3, 2, 0)


def test_job_head_logits_passes_non_empty_logits_through():
    logits = torch.tensor([0.5, -1.0])

    assert Policy.job_head_logits(logits) is logits
    assert Policy.job_head_logits(torch.empty(0)).tolist() == [0.0]


def test_ppo_update_rescores_a_stored_empty_pool_step(cpu_config, seeded, fake_vroom, tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    cpu_config.io.logdir                            = str(run_dir)
    cpu_config.training.max_disruptions_per_episode = 2
    cpu_config.training.max_edits_per_round         = 1
    cpu_config.training.minibatch_size              = 4
    cpu_config.training.num_epochs                  = 1
    cpu_config.ppo.kl_divergence_threshold          = 100.0

    trainer          = Trainer(config=cpu_config, logger=NullLogger(), tracker=Tracker(writer=FakeWriter()))
    graph, mask_info = empty_observation(cpu_config)

    trainer._run_update([0, 1])

    output     = trainer.ppo.policy.select_action(graph, mask_info)
    experience = trainer.rollout._build_experience(graph, mask_info, -1.0, output)
    experience["done"] = True
    trainer.ppo.memory.add(**experience)

    before = trainer.ppo.policy.operator_actor[0].weight.detach().clone()
    trainer.ppo.update()

    after = trainer.ppo.policy.operator_actor[0].weight.detach()
    assert torch.isfinite(after).all()
    assert not torch.equal(before, after)
