import os
import numpy as np
import random

from optical_networking_gym.topology import Modulation, get_topology
from optical_networking_gym.wrappers.qrmsa_gym import QRMSAEnvWrapper
from optical_networking_gym.heuristics.heuristics import shortest_available_path_first_fit_best_modulation_best_band

def define_modulations():
    return (
        Modulation("QPSK",   200_000, 2, minimum_osnr=6.72, inband_xt=-17),
        Modulation("16QAM",    500, 4, minimum_osnr=13.24, inband_xt=-23),
    )



def create_environment(topology="ring_4"):
    topology_name = topology
    topo_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), "topologies", f"{topology_name}.txt")
    mods = define_modulations()
    topology = get_topology(
        topo_file, topology_name, mods,
        max_span_length=80, default_attenuation=0.2,
        default_noise_figure=4.5, k_paths=2
    )
    
    seed = 10
    random.seed(seed)
    np.random.seed(seed)

    # Configuração multibanda: 10 slots para cada banda
    band_specs = [
        {"name": "L", "start_thz": 186.00, "num_slots": 10, "noise_figure_db": 4.5, "attenuation_db_km": 0.22},
        {"name": "C", "start_thz": 191.60, "num_slots": 10, "noise_figure_db": 5.0, "attenuation_db_km": 0.20},
        {"name": "S", "start_thz": 197.22, "num_slots": 10, "noise_figure_db": 6.0, "attenuation_db_km": 0.24},
    ]

    env_args = dict(
        topology=topology,
        band_specs=band_specs,
        seed=seed,
        allow_rejection=True,
        load=300,
        episode_length=5,
        launch_power_dbm=0,
        frequency_slot_bandwidth=12.5e9,
        bit_rate_selection="discrete",
        bit_rates=(48,120),
        margin=0,
        measure_disruptions=False,
        file_name="",
        k_paths=2,
        modulations_to_consider=2,  
        defragmentation=False,
        n_defrag_services=0,
        gen_observation=True,
    )
    return env_args

def main():
    env_args = create_environment("ring_4")
    env = QRMSAEnvWrapper(**env_args)
    obs, info = env.reset()
    done = False
    while not done:
        print("\n=== LINKS ANTES DA ALOCAÇÃO ===")
        for u, v in env.env.topology.edges():
            idx = env.env.topology[u][v]["index"]
            slots = env.env.topology.graph["available_slots"][idx]
            print(f"Link {u}->{v} slots: {''.join(map(str, slots.tolist()))}")
        print(f"\nCurrent service: {env.env.current_service}")
        action, _, _ = shortest_available_path_first_fit_best_modulation_best_band(env)
        obs, reward, terminated, truncated, info = env.step(action)
        print("\n=== LINKS DEPOIS DA ALOCAÇÃO ===")
        for u, v in env.env.topology.edges():
            idx = env.env.topology[u][v]["index"]
            slots = env.env.topology.graph["available_slots"][idx]
            print(f"Link {u}->{v} slots: {''.join(map(str, slots.tolist()))}")
        # print(f"Action: {action}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}, Info: {info}")

main()