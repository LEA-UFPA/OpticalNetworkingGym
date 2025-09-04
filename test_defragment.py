#!/usr/bin/env python3
"""
Teste para verificar problemas na função defragment
"""

import os
import numpy as np
from optical_networking_gym.topology import get_topology, Modulation
from optical_networking_gym.wrappers.qrmsa_gym import QRMSAEnvWrapper


def test_defragment_logic():
    """Testa problemas específicos da função defragment"""
    
    # Setup básico
    topology_name = "ring_4"
    topo_file = f"examples/topologies/{topology_name}.txt"
    mods = (
        Modulation("QPSK", 200_000, 2, minimum_osnr=6.72, inband_xt=-17),
        Modulation("16QAM", 500, 4, minimum_osnr=13.24, inband_xt=-23),
    )
    
    topology = get_topology(
        topo_file, topology_name, mods,
        max_span_length=80, default_attenuation=0.2,
        default_noise_figure=4.5, k_paths=5
    )
    
    env_args = dict(
        topology=topology,
        seed=42,
        allow_rejection=True,
        load=25,
        episode_length=100,
        num_spectrum_resources=20,  # Aumenta espectro para 20 slots
        launch_power_dbm=0,
        frequency_slot_bandwidth=12.5e9,
        frequency_start=3e8/1565e-9,
        bandwidth=20*12.5e9,  # Ajusta bandwidth para 20 slots
        bit_rate_selection="discrete",
        bit_rates=(10, 40),  # Serviços menores: 10 e 40 Gbps
        margin=0,
        measure_disruptions=False,
        file_name="",
        k_paths=2,
        modulations_to_consider=2,
        defragmentation=True,
        n_defrag_services=1,  # Defragmenta a cada 1 serviço para ver mais ação
        gen_observation=True,
    )
    
    env = QRMSAEnvWrapper(**env_args)
    obs, info = env.reset()
    
    print("=== TESTE DE DEFRAGMENTAÇÃO ===")
    
    # Aloca alguns serviços para criar fragmentação
    services_allocated = []
    total_rewards = 0
    accepted_services = 0
    rejected_services = 0
    done = False  # Inicializa a variável

    for i in range(20):  # Reduzindo para 20 para ver melhor o comportamento
        if done:
            print(f"\n!!! Episódio terminou no serviço {i}, reiniciando...")
            obs, info = env.reset()
            done = False
            continue
            
        valid_actions = [idx for idx, val in enumerate(info["mask"][:-1]) if val == 1]
        if valid_actions:
            action = valid_actions[0]
            obs, reward, done, trunc, info = env.step(action)
            total_rewards += reward
            
            if reward >= 0:
                accepted_services += 1
                print(f"✅ Serviço {i} ACEITO com reward {reward:.2f}")
            else:
                rejected_services += 1
                print(f"❌ Serviço {i} REJEITADO com reward {reward:.2f}")
                
            print(f"   Done: {done}, Truncated: {trunc}")
            
            # Mostra estado da rede quando há poucos slots livres
            if i % 5 == 0:  # A cada 5 serviços
                running_services = env.env.topology.graph["running_services"]
                print(f"   Estado: {len(running_services)} serviços ativos")
                sample_links = list(env.env.topology.edges())[:2]
                for u, v in sample_links:
                    idx = env.env.topology[u][v]["index"]
                    slots = env.env.topology.graph["available_slots"][idx]
                    free_slots = sum(slots)
                    print(f"     {u}->{v}: {free_slots} slots livres de {len(slots)}")
        else:
            print(f"⚠️ Serviço {i}: Nenhuma ação válida disponível")
            # Se não há ações válidas, verifica se podemos forçar defragmentação
            running_services = env.env.topology.graph["running_services"]
            if len(running_services) > 0:
                print(f"   Tentando defragmentação forçada com {len(running_services)} serviços...")
                env.env.defragment(num_services=2)  # Força defragmentação de 2 serviços
            
    print(f"\n=== RESUMO FINAL ===")
    print(f"Serviços aceitos: {accepted_services}")
    print(f"Serviços rejeitados: {rejected_services}")
    print(f"Reward total: {total_rewards:.2f}")
    
    # Estado final
    running_services = env.env.topology.graph["running_services"]
    print(f"Serviços ativos no final: {len(running_services)}")
    
    return True

if __name__ == "__main__":
    success = test_defragment_logic()
    print(f"\n=== TESTE CONCLUÍDO: {'✅ SUCESSO' if success else '❌ FALHA'} ===")
