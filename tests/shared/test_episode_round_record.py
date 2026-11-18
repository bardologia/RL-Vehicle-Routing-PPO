from core.shared import Environment, EpisodeDriver, RoundRecord
from model.policy_model import Action


def run_do_nothing(driver, seed):
    for step in driver.episode(seed):
        step.commit(Action(operator=2, vehicle_index=0, job_index=0))


def test_driver_keeps_one_record_per_round(cpu_config, seeded, fake_vroom):
    cpu_config.training.max_disruptions_per_episode = 4

    driver = EpisodeDriver(Environment(cpu_config), cpu_config)
    run_do_nothing(driver, 3)

    assert [record.round_index for record in driver.records] == [0, 1, 2, 3]
    assert driver.records[0].gap == 0
    assert driver.records[0].holding == 0.0
    assert all(record.gap > 0 for record in driver.records[1:])
    assert all(record.event["type"] in ("new_job", "remove_job", "new_vehicle", "remove_vehicle") for record in driver.records)


def test_record_totals_match_the_driver_totals(cpu_config, seeded, fake_vroom):
    cpu_config.training.max_disruptions_per_episode = 5

    driver = EpisodeDriver(Environment(cpu_config), cpu_config)
    run_do_nothing(driver, 5)

    assert sum(len(record.executed["served"]) for record in driver.records) == driver.totals["served"]
    assert sum(len(record.executed["failed"]) for record in driver.records) == driver.totals["failed"]
    assert sum(record.executed["dropped"] for record in driver.records) == driver.totals["dropped"]


def test_record_pools_explain_the_event(cpu_config, seeded, fake_vroom):
    cpu_config.training.max_disruptions_per_episode = 6

    driver = EpisodeDriver(Environment(cpu_config), cpu_config)
    run_do_nothing(driver, 11)

    for record in driver.records:
        advanced  = record.pools["advanced"]
        disrupted = record.pools["disrupted"]
        event     = record.event

        if event["type"] == "new_job":
            assert len(disrupted["jobs"] - advanced["jobs"]) == event["count"]
        if event["type"] == "new_vehicle":
            assert len(disrupted["vehicles"] - advanced["vehicles"]) == event["count"]
        if event["type"] == "remove_vehicle":
            assert len(advanced["vehicles"] - disrupted["vehicles"]) == event["count"]
        if event["type"] == "remove_job":
            assert 0 < len(advanced["jobs"] - disrupted["jobs"]) <= event["count"]


def test_served_jobs_keep_their_attributes_in_the_record(cpu_config, seeded, fake_vroom):
    cpu_config.training.max_disruptions_per_episode = 5

    driver = EpisodeDriver(Environment(cpu_config), cpu_config)
    run_do_nothing(driver, 7)

    served = [(record, job_id) for record in driver.records for job_id in record.executed["served"] + record.executed["failed"]]
    assert served

    for record, job_id in served:
        assert set(record.plan_before[job_id]) == {"vehicle_id", "priority", "kind"}
        assert job_id in record.pools["before"]["jobs"]
        assert job_id not in record.pools["advanced"]["jobs"]


def test_rounds_run_on_a_prepared_environment_without_a_disruption(cpu_config, seeded, fake_vroom):
    cpu_config.training.max_disruptions_per_episode = 2

    environment = Environment(cpu_config)
    jobs_before = list(environment.jobs.ids)
    environment.open_episode(disrupt=False)

    driver = EpisodeDriver(environment, cpu_config)
    steps  = []
    for step in driver.rounds():
        steps.append(step)
        step.commit(Action(operator=2, vehicle_index=0, job_index=0))

    assert isinstance(driver.records[0], RoundRecord)
    assert driver.records[0].event is None
    assert driver.records[0].pools["disrupted"]["jobs"] == set(jobs_before)
    assert [step.round_index for step in steps] == [0, 1]


def test_episode_matches_rounds_after_sample_episode(cpu_config, seeded, fake_vroom):
    cpu_config.training.max_disruptions_per_episode = 3

    def rewards_of(use_rounds):
        environment = Environment(cpu_config)
        driver      = EpisodeDriver(environment, cpu_config)

        if use_rounds:
            environment.sample_episode(21)
            steps = driver.rounds()
        else:
            steps = driver.episode(21)

        totals = []
        for step in steps:
            _, _, rewards, _ = step.commit(Action(operator=2, vehicle_index=0, job_index=0))
            totals.append(sum(rewards.values()))

        return totals

    assert rewards_of(True) == rewards_of(False)
