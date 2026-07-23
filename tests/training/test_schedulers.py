import pytest
import torch

from core.training import EntropyScheduler, EpochEarlyStopping, LRScheduler


def build_optimizer(*lrs):
    groups = [{"params": [torch.nn.Parameter(torch.zeros(1))], "lr": lr} for lr in lrs]
    return torch.optim.SGD(groups)


def test_lr_scheduler_captures_base_lr_per_group():
    scheduler = LRScheduler(build_optimizer(1.0, 0.5), warmup_steps=10, decay_steps=100, lr_min=0.01)

    assert scheduler.lr_max_per_group == [1.0, 0.5]


def test_lr_warmup_interpolates_from_ten_percent():
    optimizer = build_optimizer(1.0)
    scheduler = LRScheduler(optimizer, warmup_steps=10, decay_steps=100, lr_min=0.01)

    scheduler.set_step(8)
    lr = scheduler.step()

    assert lr == pytest.approx(0.1 + 0.9 * 0.9)


def test_lr_reaches_base_at_warmup_boundary():
    optimizer = build_optimizer(1.0)
    scheduler = LRScheduler(optimizer, warmup_steps=10, decay_steps=100, lr_min=0.01)

    scheduler.set_step(9)
    lr = scheduler.step()

    assert lr == pytest.approx(1.0)


def test_lr_decays_to_floor_at_end_of_schedule():
    optimizer = build_optimizer(1.0)
    scheduler = LRScheduler(optimizer, warmup_steps=10, decay_steps=100, lr_min=0.01)

    scheduler.set_step(109)
    lr = scheduler.step()

    assert lr == pytest.approx(0.01)


def test_lr_holds_floor_past_end_of_schedule():
    optimizer = build_optimizer(1.0)
    scheduler = LRScheduler(optimizer, warmup_steps=10, decay_steps=100, lr_min=0.01)

    scheduler.set_step(500)
    lr = scheduler.step()

    assert lr == pytest.approx(0.01)


def test_lr_get_lr_reports_all_groups():
    optimizer = build_optimizer(1.0, 0.5)
    scheduler = LRScheduler(optimizer, warmup_steps=10, decay_steps=100, lr_min=0.01)

    scheduler.step()
    rates = scheduler.get_lr()

    assert set(rates.keys()) == {"group_0", "group_1"}


def test_entropy_returns_start_during_warmup():
    scheduler = EntropyScheduler(start_coef=0.02, end_coef=0.001, anneal_steps=100, warmup_steps=5)

    scheduler.set_step(5)

    assert scheduler.get_coef() == pytest.approx(0.02)


def test_entropy_reaches_end_after_anneal():
    scheduler = EntropyScheduler(start_coef=0.02, end_coef=0.001, anneal_steps=100, warmup_steps=5)

    scheduler.set_step(5 + 100)

    assert scheduler.get_coef() == pytest.approx(0.001)


def test_entropy_default_step_zero_returns_start():
    scheduler = EntropyScheduler(start_coef=0.05, end_coef=0.001, anneal_steps=100)

    assert scheduler.get_coef() == pytest.approx(0.05)


def test_entropy_anneals_monotonically():
    scheduler = EntropyScheduler(start_coef=0.02, end_coef=0.001, anneal_steps=100, warmup_steps=0)

    previous = scheduler.get_coef()
    for _ in range(50):
        current  = scheduler.step()
        assert current <= previous + 1e-9
        previous = current


def test_entropy_step_advances_and_returns_coef():
    scheduler = EntropyScheduler(start_coef=0.02, end_coef=0.001, anneal_steps=100, warmup_steps=0)

    coef = scheduler.step()

    assert scheduler.current_step == 1
    assert coef == scheduler.get_coef()


def test_early_stopping_triggers_strictly_above_threshold():
    stopper = EpochEarlyStopping(threshold=0.015)

    assert stopper.should_stop(0.016) is True
    assert stopper.should_stop(0.015) is False
    assert stopper.should_stop(0.014) is False
