from core.training.pretraining import RegretInsertionTeacher


def same_action(first, second):
    return (first.operator, first.vehicle_index, first.job_index) == (second.operator, second.vehicle_index, second.job_index)


def unassign(environment, count):
    state = environment.current_state
    for route in list(state.routes):
        for job_id in route.job_ids[:count]:
            state = environment.action_handler.apply_job_removal(environment, state, route.vehicle_id, job_id)
        break

    environment.current_state = state
    return state


def test_explain_agrees_with_select_action(environment, fake_vroom, cpu_config):
    for allow_removal in (True, False):
        teacher = RegretInsertionTeacher(cpu_config, allow_removal=allow_removal)

        for count in (0, 1, 2):
            state       = unassign(environment, count)
            explanation = teacher.explain(environment, state, remaining_steps=3)

            assert same_action(explanation["action"], teacher.select_action(environment, state, remaining_steps=3))


def test_explain_lists_every_feasible_insertion_and_the_baseline(environment, fake_vroom, cpu_config):
    teacher     = RegretInsertionTeacher(cpu_config)
    state       = unassign(environment, 2)
    explanation = teacher.explain(environment, state)

    assert explanation["baseline"] == teacher._no_op_reward(environment, state)
    assert set(explanation["insertions"]) == set(state.unassigned_ids)
    assert explanation["insertions"] == teacher._insertion_options(environment, state)

    for entries in explanation["insertions"].values():
        assert {vehicle_id for _, vehicle_id in entries} <= set(environment.vehicles.ids)


def test_explain_reports_removal_continuations_only_when_removal_is_allowed(environment, fake_vroom, cpu_config):
    state = environment.current_state

    with_removal = RegretInsertionTeacher(cpu_config).explain(environment, state)
    without      = RegretInsertionTeacher(cpu_config, allow_removal=False).explain(environment, state)

    assert without["removals"] is None
    assert len(with_removal["removals"]) == len(state.assigned_job_ids)

    for entry in with_removal["removals"]:
        assert entry["value"] == entry["reward"] + cpu_config.ppo.gamma * entry["continuation"]


def test_explain_leaves_the_environment_untouched(environment, fake_vroom, cpu_config):
    state  = unassign(environment, 1)
    before = (list(environment.jobs.ids), list(environment.vehicles.ids), state.to_payload(), environment.clock)

    RegretInsertionTeacher(cpu_config).explain(environment, state)

    assert before == (list(environment.jobs.ids), list(environment.vehicles.ids), environment.current_state.to_payload(), environment.clock)
