#!/usr/bin/env python3
# ring4_test.py - Script minimalista para testar multibanda com ring4

import os
import numpy as np
import random

from optical_networking_gym.topology import Modulation, get_topology
from optical_networking_gym.wrappers.qrmsa_gym import QRMSAEnvWrapper


def define_modulations():
    return (
        Modulation("QPSK",   200_000, 2, minimum_osnr=6.72, inband_xt=-17),
        Modulation("16QAM",    500, 4, minimum_osnr=13.24, inband_xt=-23),
    )


def create_ring4_environment():
    topology_name = "ring_4"
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
        k_paths=2,
        modulations_to_consider=2,  # Volta para 2 modulações
        defragmentation=False,
        n_defrag_services=0,
        gen_observation=True,
    )
    return env_args


def print_mask_breakdown(env, mask):
    """Printa a mask conforme formato solicitado: C + 16QAM + k0 : 0 - 10 [111111111]"""
    print("\n=== MASK BREAKDOWN ===")
    
    # Debug: verificar ordem das modulações
    print("Debug modulações:")
    for i, mod in enumerate(env.env.modulations):
        print(f"  mod_idx {i}: {mod.name}")
    
    try:
        # Para cada k-path
        for path_idx in range(env.env.k_paths):
            # Para cada banda
            for band_idx, band in enumerate(env.env.bands):
                # Para cada modulação disponível  
                for mod_idx in range(len(env.env.modulations)):
                    modulation = env.env.modulations[mod_idx]
                    
                    # Coletar ações e máscaras para esta combinação path+banda+modulação
                    actions = []
                    band_masks = []
                    
                    for slot_idx in range(band.num_slots):  # Usar slots relativos à banda (0-9)
                        try:
                            # Calcular ação para esta combinação
                            action = calculate_action(env, path_idx, band_idx, mod_idx, slot_idx)
                            if action is not None and action < len(mask):
                                actions.append(action)
                                band_masks.append(int(mask[action]))
                            else:
                                band_masks.append(0)
                        except Exception as e:
                            print(f"Erro calculando action para slot {slot_idx}: {e}")
                            band_masks.append(0)
                    
                    mask_str = ''.join(map(str, band_masks))
                    
                    # Range de ações para esta combinação
                    if actions:
                        action_range = f"{min(actions)} - {max(actions)}"
                    else:
                        action_range = "N/A"
                    
                    print(f"{band.name} + {modulation.name} + k{path_idx}: {action_range} [{mask_str}]")
                    
    except Exception as e:
        print(f"Erro no breakdown: {e}")
        # Fallback simples
        print("Máscara (primeiros 50):", ''.join(map(str, mask.astype(int)[:50])))


def calculate_action(env, path_idx, band_idx, mod_idx, slot_relative):
    """Calcula a ação baseado na lógica do encoding com slots relativos"""
    # Usar slots_per_band em vez de total_slots
    slots_per_band = 10  # Como definido no action space corrigido
    
    # A ação 0 é rejeição
    # As ações válidas começam de 1
    # slot 0 -> ação 1, slot 1 -> ação 2, etc.
    #
    # Mas se o usuário disse que "ações de 0 a 9 deveriam ser 16qam":
    # - Ações 0-9: 16QAM slots 0-9
    # - Ações 10-19: QPSK slots 0-9 (sendo ação 19 = 0 porque slot 9 não cabe)
    #
    # Isso significa que não há offset de +1, as ações mapeiam diretamente para slots
    
    # Calcular ação usando slots relativos SEM o +1
    action = (path_idx * (env.env.num_bands * len(env.env.modulations) * slots_per_band) + 
              band_idx * (len(env.env.modulations) * slots_per_band) + 
              mod_idx * slots_per_band + 
              slot_relative)
    
    return action


def main():
    env_args = create_ring4_environment()
    env = QRMSAEnvWrapper(**env_args)

    print("=== CONFIGURAÇÃO RING4 MULTIBANDA ===")
    print(f"Número de bandas: {env.env.num_bands}")
    print(f"Total de slots: {env.env.total_slots}")
    print(f"Action space size: {env.env.action_space.n}")
    
    for i, band in enumerate(env.env.bands):
        print(f"Banda {i}: {band}")

    # Reset do ambiente
    obs, info = env.reset()
    svc = env.env.current_service
    
    print(f"\n=== SERVIÇO ATUAL ===")
    print(f"ID={svc.service_id}, src={svc.source}->{svc.destination}, bit_rate={svc.bit_rate}")
    
    # Mostrar num_slots para cada modulação
    for mod in env.env.modulations:
        num_slots = env.env.get_number_slots(svc, mod)
        print(f"Modulação {mod.name}: {num_slots} slots")
    
    # Mostrar máscara
    mask = info["mask"]
    print(f"\nMáscara completa: {mask.astype(int)}")
    

    step_count = 0
    
    while True:
        step_count += 1  
        print_mask_breakdown(env, mask)
        action = 0
        print(f"\n=== EXECUTANDO AÇÃO {action} (Step {step_count}) ===")
        
        # Decodificar ação
        path_i, band_i, mod_i, slot_i = env.env.encoded_decimal_to_array(int(action))
        print(f"Decoded action: path={path_i}, band={band_i}, mod={mod_i}, slot={slot_i}")
        
        band = env.env.bands[band_i]
        modulation = env.env.modulations[mod_i]
        num_slots = env.env.get_number_slots(svc, modulation)

        print(f"Banda: {band.name}, Modulação: {modulation.name}, Slot inicial: {slot_i}, Num slots: {num_slots}")
        
        # Executar step
        obs2, reward, done, trunc, info2 = env.step(action)
        mask = info2["mask"]
        print(f"Step result: reward={reward:.2f}, done={done}")
        
        # Verificar se foi aceito
        if svc.accepted:
            print(f"✓ Serviço aceito! Banda: {svc.current_band}")
            print(f"Slot inicial: {svc.initial_slot}, Número de slots: {svc.number_slots}")
        else:
            print(f"✗ Serviço rejeitado")
            
        # Atualizar serviço atual para próximo step
        svc = env.env.current_service

        print("\n=== LINKS DEPOIS DA ALOCAÇÃO ===")
        for u, v in env.env.topology.edges():
            idx = env.env.topology[u][v]["index"]
            slots = env.env.topology.graph["available_slots"][idx]
            print(f"Link {u}->{v} slots: {''.join(map(str, slots.tolist()))}")
        if done :
            break


if __name__ == "__main__":
    main()
