import random
import torch
import numpy as np
import cProfile
import pstats
from io import StringIO

from core.shared import Environment


def generate_events(batch_size, seed, config, enable_profiling=False):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    profile_stats = None
    if enable_profiling:
        profiler = cProfile.Profile()
        profiler.enable()

    simulation_env = Environment(config)
    batch = [None] * batch_size

    for i in range(batch_size):
        simulation_env.reset()
        initial_state = simulation_env.initial_state
        event_type, num_items = simulation_env.generate_event()
        event_state = simulation_env.apply_event(initial_state, event_type, num_items)
        graph, mask_info = simulation_env.observe()

        graph_cpu = graph.clone().detach().cpu()

        batch[i] = {
            "state"     : event_state.to_payload(),
            "graph"     : graph_cpu,
            "mask_info" : mask_info,
            "jobs"      : [job.to_dict() for job in simulation_env.jobs],
            "vehicles"  : [vehicle.to_dict() for vehicle in simulation_env.vehicles],
        }

    if enable_profiling:
        profiler.disable()
        s = StringIO()
        stats = pstats.Stats(profiler, stream=s)
        stats.sort_stats('cumulative')
        stats.print_stats(50)
        profile_stats = s.getvalue()

    return batch, profile_stats


def generate_events_task(task):
    return generate_events(*task)
