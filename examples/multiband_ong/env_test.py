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
    topo_file = os.path.join("..", "..", "examples", "topologies", f"{topology_name}.xml")
    mods = define_modulations()
    topology = get_topology(
        topo_file, topology_name, mods,
        max_span_length=80, default_attenuation=0.2,
        default_noise_figure=4.5, k_paths=5
    )
    seed = 10
    random.seed(seed)
    np.random.seed(seed)

    # Configuração multibanda: C-band e S-band
    # Para uma única banda, ainda é necessário especificar via band_specs:
    # band_specs = [{"name": "C", "start_thz": 191.60, "num_slots": 20, "noise_figure_db": 5.0, "attenuation_db_km": 0.20}]
    
    band_specs = [
        {"name": "C", "start_thz": 191.60, "num_slots": 10, "noise_figure_db": 5.0, "attenuation_db_km": 0.20},
        {"name": "S", "start_thz": 197.22, "num_slots": 10, "noise_figure_db": 6.0, "attenuation_db_km": 0.24},
    ]

    env_args = dict(
        topology=topology,
        band_specs=band_specs,  # Obrigatório - especifica as bandas
        seed=seed,
        allow_rejection=True,
        load=300,
        episode_length=10,            # poucas iterações
        # num_spectrum_resources removido - será calculado automaticamente (10+10=20)
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


def create_single_band_environment():
    """Exemplo de como usar uma única banda (ainda requer band_specs)"""
    topology_name = "nobel-eu"
    topo_file = os.path.join("..", "..", "examples", "topologies", f"{topology_name}.xml")
    mods = define_modulations()
    topology = get_topology(
        topo_file, topology_name, mods,
        max_span_length=80, default_attenuation=0.2,
        default_noise_figure=4.5, k_paths=5
    )
    
    # Configuração de banda única - ainda requer band_specs
    band_specs = [
        {"name": "C", "start_thz": 191.60, "num_slots": 20, "noise_figure_db": 5.0, "attenuation_db_km": 0.20}
    ]

    env_args = dict(
        topology=topology,
        band_specs=band_specs,  # Obrigatório mesmo para uma banda
        seed=10,
        allow_rejection=True,
        load=300,
        episode_length=5,
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
    return topology, env_args


# 3) Função de debug
def debug_run():
    topology, env_args = create_environment()
    env = QRMSAEnvWrapper(**env_args)

    # Prints para verificar multibanda
    print("\n=== CONFIGURAÇÃO MULTIBANDA ===")
    print(f"Número de bandas: {env.env.num_bands}")
    print(f"Total de slots: {env.env.total_slots}")
    print(f"Action space size: {env.env.action_space.n}")
    
    for i, band in enumerate(env.env.bands):
        print(f"Banda {i}: {band}")
    
    print(f"Formato do espaço de ação: k_paths({env.env.k_paths}) × num_bands({env.env.num_bands}) × modulations({env.env.modulations_to_consider}) × total_slots({env.env.total_slots}) + 1")
    expected_actions = env.env.k_paths * env.env.num_bands * env.env.modulations_to_consider * env.env.total_slots + 1
    print(f"Ações esperadas: {expected_actions}, Ações reais: {env.env.action_space.n}")

    obs, info = env.reset()
    svc = env.env.current_service

    print("\n=== ESTADO INICIAL DOS LINKS (MULTIBANDA) ===")
    for u, v in env.env.topology.edges():
        idx = env.env.topology[u][v]["index"]
        slots = env.env.topology.graph["available_slots"][idx]
        print(f"Link {(u,'->',v)} slots ({len(slots)} total): {''.join(map(str,slots.tolist()))}")

    print("\n--- Serviço atual ---")
    print(f"ID={svc.service_id}, src={svc.source}->{svc.destination}, bit_rate={svc.bit_rate}")
    print("Máscara válida de ações (1=válido):", info["mask"].astype(int)[:50], "... (primeiros 50)")

    # Encontra uma ação válida
    valid_actions = np.where(info["mask"] == 1)[0]
    if len(valid_actions) > 0:
        action = valid_actions[0]  # Pega primeira ação válida
        print(f"\n>> Selecionou ação válida {action}")

        # Decodifica em 4 dimensões: [path, band, modulation, slot]
        path_i, band_i, mod_i, slot_i = env.env.encoded_decimal_to_array(int(action))
        print(f">> Decoded action (4D): path={path_i}, band={band_i}, mod={mod_i}, slot={slot_i}")

        band = env.env.bands[band_i]
        path = env.env.k_shortest_paths[svc.source, svc.destination][path_i]
        modulation = env.env.modulations[mod_i]
        num_slots = env.env.get_number_slots(svc, modulation)
        
        print(f">> Banda selecionada: {band.name} (slots {band.slot_start}-{band.slot_end-1})")
        print(f">> Slot inicial global: {slot_i}, num_slots: {num_slots}")
        
        # Verificar se o slot está dentro da banda
        if band.contains_slot_range(slot_i, num_slots):
            print(f">> ✓ Slot range [{slot_i}, {slot_i+num_slots}) está dentro da banda {band.name}")
        else:
            print(f">> ✗ Slot range [{slot_i}, {slot_i+num_slots}) NÃO está dentro da banda {band.name}")

        available = env.env.get_available_slots(path)
        print(f">> Slots disponíveis na rota: {''.join(map(str,available.tolist()))}")

        # Candidatos específicos da banda
        band_candidates = env.env._get_candidates_in_band(available, num_slots, band)
        print(f">> Candidatos válidos na banda {band.name}: {band_candidates}")

        # Frequência central via banda
        if band.contains_slot_range(slot_i, num_slots):
            center_freq = band.center_frequency_hz_from_global(slot_i, num_slots)
            print(f">> Frequência central calculada pela banda: {center_freq/1e12:.3f} THz")

        # executa o step
        obs2, reward, done, trunc, info2 = env.step(action)
        print(f"\n>> Step(action={action}) => reward={reward:.2f}, done={done}")
        
        # Verifica se o serviço foi aceito e qual banda foi atribuída
        if svc.accepted:
            print(f">> ✓ Serviço aceito! Banda atribuída: {svc.current_band}")
            print(f">> Slot inicial: {svc.initial_slot}, número de slots: {svc.number_slots}")
            print(f">> Frequência central: {svc.center_frequency/1e12:.3f} THz")
        else:
            print(f">> ✗ Serviço rejeitado")

        # depois da alocação, mostra links envolvidos
        print("\n=== LINKS DEPOIS DA ALOCAÇÃO (MULTIBANDA) ===")
        for u, v in zip(path.node_list, path.node_list[1:]):
            idx = env.env.topology[u][v]["index"]
            slots = env.env.topology.graph["available_slots"][idx]
            running = env.env.topo_cache.edge_running_services[idx] if hasattr(env.env, 'topo_cache') else []
            print(f" {u}->{v} slots: {''.join(map(str,slots.tolist()))},  #services={[s.service_id for s in running]}")

    else:
        print("\n>> ✗ Nenhuma ação válida encontrada!")

    print("\n>> Nova máscara:", info2["mask"].astype(int)[:50] if 'info2' in locals() else "N/A")


if __name__ == "__main__":
    print("=== TESTE MULTIBANDA (2 bandas) ===")
    debug_run()
    
    print("\n" + "="*60)
    print("=== TESTE BANDA ÚNICA (ainda requer band_specs) ===")
    
    # Testar banda única
    topology_single, env_args_single = create_single_band_environment()
    env_single = QRMSAEnvWrapper(**env_args_single)
    
    print(f"Número de bandas: {env_single.env.num_bands}")
    print(f"Total de slots: {env_single.env.total_slots}")
    print(f"Action space size: {env_single.env.action_space.n}")
    print(f"Banda: {env_single.env.bands[0]}")
    
    obs, info = env_single.reset()
    print("✓ Ambiente de banda única funcionando!")
    print(f"num_spectrum_resources calculado: {env_single.env.num_spectrum_resources}")

