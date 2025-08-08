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

# Imports do optical-networking-gym
from optical_networking_gym.topology import Modulation, get_topology
from optical_networking_gym.wrappers.qrmsa_gym import QRMSAEnvWrapper
from optical_networking_gym.heuristics.heuristics import shortest_available_path_first_fit_best_modulation
from optical_networking_gym.core.osnr import calculate_osnr


# 1) Definição das modulações
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

    env_args = dict(
        topology=topology,
        seed=seed,
        allow_rejection=True,
        load=300,
        episode_length=10,            # poucas iterações
        num_spectrum_resources=20,
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


# 3) Função de debug
def debug_run():
    topology, env_args = create_environment()
    env = QRMSAEnvWrapper(**env_args)

    obs, info = env.reset()
    svc = env.env.current_service

    print("\n=== ESTADO INICIAL DOS LINKS ===")
    for u, v in env.env.topology.edges():
        idx = env.env.topology[u][v]["index"]
        slots = env.env.topology.graph["available_slots"][idx]
        print(f"Link {(u,'->',v)} slots: {''.join(map(str,slots.tolist()))}")

    print("\n--- Serviço atual ---")
    print(f"ID={svc.service_id}, src={svc.source}->{svc.destination}, bit_rate={svc.bit_rate}")
    print("Máscara válida de ações (1=válido):", info["mask"].astype(int))

    # escolhe ação e mostra detalhes
    action = shortest_available_path_first_fit_best_modulation(info["mask"])
    print(f"\n>> Heurística selecionou ação {action}")

    path_i, mod_i, slot_i = env.env.encoded_decimal_to_array(action)
    print(f">> Decoded action: path={path_i}, mod={mod_i}, slot={slot_i}")

    path = env.env.k_shortest_paths[svc.source, svc.destination][path_i]
    modulation = env.env.modulations[mod_i]
    num_slots = env.env.get_number_slots(svc, modulation)
    available = env.env.get_available_slots(path)
    cand = env.env._get_candidates(available, num_slots, env.env.num_spectrum_resources)

    print("\n--- Cálculo de recursos ---")
    print(f"Slots requeridos: {num_slots}")
    print(f"Slots disponíveis na rota: {available.tolist()[:50]}...")  # print parcial
    print(f"Candidatos válidos (inicial slots): {cand}")

    # antes da alocação, mostra links envolvidos
    print("\n=== LINKS ANTES DA ALOCAÇÃO ===")
    for i, (u,v) in enumerate(zip(path.node_list, path.node_list[1:])):
        idx = env.env.topology[u][v]["index"]
        print(f" {u}->{v} slots: {''.join(map(str, env.env.topo_cache.available_slots[idx].tolist()))[:50]}...")

    # configura serviço para calcular OSNR
    svc.path = path
    svc.initial_slot = slot_i
    svc.number_slots = num_slots
    svc.center_frequency = (
        env.env.frequency_start +
        env.env.frequency_slot_bandwidth*(slot_i + num_slots/2)
    )
    svc.bandwidth = env.env.frequency_slot_bandwidth * num_slots
    svc.launch_power = env.env.launch_power

    osnr, ase, nli = calculate_osnr(env.env, svc)
    print(f"\n--- Cálculo de OSNR ---")
    print(f"OSNR = {osnr:.2f}, ASE = {ase:.2f}, NLI = {nli:.2f}")
    print(f"Relação mínima exigida = {modulation.minimum_osnr + env.env.margin:.2f}")

    # executa o step
    obs2, reward, done, trunc, info2 = env.step(action)
    print(f"\n>> Step(action={action}) => reward={reward:.2f}, done={done}")

    # depois da alocação, mostra links envolvidos
    print("\n=== LINKS DEPOIS DA ALOCAÇÃO ===")
    for u, v in zip(path.node_list, path.node_list[1:]):
        idx = env.env.topology[u][v]["index"]
        slots = env.env.topology.graph["available_slots"][idx]
        running = env.env.topo_cache.edge_running_services[idx]
        print(f" {u}->{v} slots: {''.join(map(str,slots.tolist()))[:50]},  #services={[s.service_id for s in running]}")

    print("\n>> Nova máscara:", info2["mask"].astype(int))


if __name__ == "__main__":
    debug_run()

