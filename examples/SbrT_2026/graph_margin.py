import argparse
import logging
import os
import random
from typing import List, Tuple
from multiprocessing import Pool
import numpy as np
import time
from datetime import datetime

from optical_networking_gym.topology import Modulation, get_topology


def run_environment(    
    n_eval_episodes,
    heuristic,
    monitor_file_base,
    topology,
    seed,
    allow_rejection,
    load,
    episode_length,
    num_spectrum_resources,
    launch_power_dbm,
    bandwidth,
    frequency_start,
    frequency_slot_bandwidth,
    bit_rate_selection,
    bit_rates,
    margin, # Variável independente neste script
    file_name,
    measure_disruptions,
    defragmentation,
    n_defrag_services,
    gen_observation,
) -> None:
    from optical_networking_gym.wrappers.qrmsa_gym import QRMSAEnvWrapper
    from optical_networking_gym.heuristics.heuristics import (
        heuristic_shortest_available_path_first_fit_best_modulation,
        heuristic_highest_snr,
        heuristic_lowest_fragmentation,
        load_balancing_best_modulation,
    )

    # Configurações do ambiente baseadas no padrão QRMSA
    env_args = dict(
        topology=topology,
        seed=seed,
        allow_rejection=allow_rejection,
        load=load,
        episode_length=episode_length,
        num_spectrum_resources=num_spectrum_resources,
        launch_power_dbm=launch_power_dbm,
        bandwidth=bandwidth,
        frequency_start=frequency_start,
        frequency_slot_bandwidth=frequency_slot_bandwidth,
        bit_rate_selection=bit_rate_selection,
        bit_rates=bit_rates,
        margin=margin,
        file_name=file_name,
        measure_disruptions=measure_disruptions,
        k_paths=5, 
        modulations_to_consider=6,
        defragmentation=defragmentation,
        n_defrag_services=n_defrag_services,
        gen_observation=gen_observation,
    )

    # Mapeamento da heurística
    fn_map = {
        1: heuristic_shortest_available_path_first_fit_best_modulation,
        2: heuristic_highest_snr,
        3: heuristic_lowest_fragmentation,
        4: load_balancing_best_modulation
    }
    fn_heuristic = fn_map.get(heuristic, heuristic_shortest_available_path_first_fit_best_modulation)

    env = QRMSAEnvWrapper(**env_args)
    
    # Nome do ficheiro focado na margem testada
    monitor_final_name = f"{monitor_file_base}_m_{margin}_load_{load}.csv"

    os.makedirs(os.path.dirname(monitor_final_name), exist_ok=True)

    with open(monitor_final_name, "wt", encoding="UTF-8") as file_handler:
        file_handler.write(f"# Date: {datetime.now()}\n")
        header = "episode,service_blocking_rate,episode_service_blocking_rate,mean_gsnr\n"
        file_handler.write(header)

        for ep in range(n_eval_episodes):
            env.reset()
            done = False
            while not done:
                action, _, _ = fn_heuristic(env)
                _, _, done, _, info = env.step(action)
            
            # Cálculo de GSNR médio dos serviços ativos
            services = env.env.topology.graph.get("services", [])
            mean_gsnr = np.mean([s.OSNR for s in services]) if services else 0.0
            
            row = f"{ep},{info['service_blocking_rate']},{info['episode_service_blocking_rate']},{mean_gsnr}\n"
            file_handler.write(row)

    print(f"Finalizado: Margem {margin} dB | Resultados em: {monitor_final_name}")

def main():
    parser = argparse.ArgumentParser(description='Simulação: Variação de Margem de OSNR')
    parser.add_argument('-t', '--topology_file', type=str, default='nsfnet_chen.txt')
    parser.add_argument('-e', '--num_episodes', type=int, default=10)
    parser.add_argument('-l', '--load', type=float, default=400.0, help='Carga fixa (Erlang)')
    parser.add_argument('-th', '--threads', type=int, default=1)
    parser.add_argument('-hi', '--heuristic_index', type=int, default=1)
    parser.add_argument(
        '-m', '--margin',
        type=float,
        nargs='+',
        default=[0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
        help='Um ou mais valores de margem de OSNR em dB (ex: -m 0.0 1.0 2.0)',
    )
    args = parser.parse_args()

    # Definição dos formatos de modulação
    cur_modulations = (
        Modulation("BPSK", 7000, 1, 3.71, -14),
        Modulation("QPSK", 3300, 2, 6.72, -17),
        Modulation("8QAM", 3370, 3, 10.84, -20),
        Modulation("16QAM", 3489, 4, 13.24, -23),
        Modulation("32QAM", 2652, 5, 16.16, -26),
        Modulation("64QAM", 1387, 6, 19.01, -29),
    )

    # Carregamento da topologia
    topology_path = os.path.join("examples/topologies", args.topology_file)
    topology = get_topology(topology_path, None, cur_modulations, 80, 0.2, 4.5, 5)

    margins = args.margin
    env_args = []
    seed = 50

    for m in margins:
        # Gera argumentos para cada nível de margem
        sim_args = (
            args.num_episodes, 
            args.heuristic_index, 
            "examples/SbrT_2026/results/margin_study/m_episodes",
            topology,
            seed,
            True,
            args.load, # Carga fixa para todos os testes
            1000, 
            320, 
            1.0, # launch_power_dbm
            4e12, # bandwidth
            3e8/1565e-9, # frequency_start
            12.5e9, # frequency_slot_bandwidth
            "discrete", 
            (10, 40, 100, 400), 
            m, # Margem variável
            f"examples/SbrT_2026/results/margin_study/m_services_{m}", 
            False, # measure_disruptions
            False, # defragmentation
            0, # n_defrag_services
            False, # gen_observation
        )
        env_args.append(sim_args)

    print(f"Iniciando simulações para {len(margins)} níveis de margem (Carga: {args.load} Erlang)...")
    
    if args.threads > 1:
        with Pool(processes=args.threads) as pool:
            pool.starmap(run_environment, env_args)
    else:
        for arg in env_args:
            run_environment(*arg)

    print("Simulações concluídas com sucesso.")

if __name__ == "__main__":
    main()