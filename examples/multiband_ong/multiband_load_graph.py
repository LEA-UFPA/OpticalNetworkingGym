#!/usr/bin/env python3
# debug_qrmsa_verbose.py

import os
import csv
import time
import random
import numpy as np
import gymnasium as gym

from datetime import datetime
from typing import Tuple

from optical_networking_gym.topology import Modulation, get_topology
from optical_networking_gym.wrappers.qrmsa_gym import QRMSAEnvWrapper
from optical_networking_gym.heuristics.heuristics import shortest_available_path_first_fit_best_modulation_best_band
from optical_networking_gym.core.osnr import calculate_osnr


def define_modulations() -> Tuple[Modulation, ...]:
    return (
        Modulation("QPSK",   200_000, 2, minimum_osnr=6.72, inband_xt=-17),
        Modulation("16QAM",    500, 4, minimum_osnr=13.24, inband_xt=-23),
    )


def create_environment():
    topology_name = "nobel-eu"
    topo_file = os.path.join("examples", "topologies", f"{topology_name}.xml")
    mods = define_modulations()
    topology = get_topology(
        topo_file, topology_name, mods,
        max_span_length=80, default_attenuation=0.2,
        default_noise_figure=4.5, k_paths=5
    )
    seed = 10
    random.seed(seed)
    np.random.seed(seed)
    
    band_specs = [
        {"name": "C", "start_thz": 191.60, "num_slots": 10, "noise_figure_db": 5.0, "attenuation_db_km": 0.20},
        {"name": "S", "start_thz": 197.22, "num_slots": 10, "noise_figure_db": 6.0, "attenuation_db_km": 0.24},
    ]

    env_args = dict(
        topology=topology,
        band_specs=band_specs,
        seed=seed,
        allow_rejection=True,
        load=300,
        episode_length=10,       
        launch_power_dbm=0,
        frequency_slot_bandwidth=12.5e9,
        frequency_start=3e8/1565e-9,
        bandwidth=20*12.5e9,
        bit_rate_selection="discrete",
        bit_rates=(10,40),
        margin=0,
        measure_disruptions=False,
        file_name="",
        k_paths=3,
        modulations_to_consider=4,
        defragmentation=False,
        n_defrag_services=0,
        gen_observation=True,
    )
    return topology, env_args


if __name__ == "__main__":
    topology, args = create_environment()
    env = QRMSAEnvWrapper(**args)

    # Run a single episode
    obs, info = env.reset()
    done = False
    step = 0
    while not done:
        step += 1
        action, _,_ = shortest_available_path_first_fit_best_modulation_best_band(env)
        obs, reward, done, truncated, info = env.step(action)
        print(f"Step: {step}, Action: {action}, Reward: {reward}")