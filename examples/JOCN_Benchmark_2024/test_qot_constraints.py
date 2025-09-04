#!/usr/bin/env python3
"""
Script para testar 3 QoT constraints (ASE+NLI, ASE, DIST) com 3 cargas diferentes.
Baseado no graph_load.py e melhorado com as funções do utils.py.
"""

import argparse
import logging
import os
import random
import time
from datetime import datetime
from typing import List, Tuple
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from optical_networking_gym.topology import Modulation, get_topology
from optical_networking_gym.wrappers.qrmsa_gym import QRMSAEnvWrapper


class QoTTestUtils:
    """Utilitários para testes de QoT constraints."""
    
    @staticmethod
    def define_modulations() -> Tuple[Modulation, ...]:
        """Define as modulações conforme utils.py OFC_2025."""
        return (
            Modulation("BPSK", 5_000, 1, minimum_osnr=3.71, inband_xt=-14),
            Modulation("QPSK", 3_300,   2, minimum_osnr=6.72, inband_xt=-17),
            Modulation("8QAM",  3_370,   3, minimum_osnr=10.84, inband_xt=-20),
            Modulation("16QAM",   3_489,   4, minimum_osnr=13.24, inband_xt=-23),
            Modulation("32QAM",   2_652,   5, minimum_osnr=16.16, inband_xt=-26),
            Modulation("64QAM",   1_387,   6, minimum_osnr=19.01, inband_xt=-29),
        )

    @staticmethod
    def get_topology(topology_name: str, qot_constraint: str = "ASE+NLI") -> object:
        """Cria a topologia com as modulações e QoT constraint especificada."""
        base_dir = Path(__file__).resolve().parents[2]
        topo_dir = base_dir / "examples" / "topologies"
        
        for ext in ("xml", "txt"):
            topo_path = topo_dir / f"{topology_name}.{ext}"
            if topo_path.exists():
                break
        else:
            raise FileNotFoundError(f"Topologia '{topology_name}.xml|.txt' não encontrada em {topo_dir}")

        mods = QoTTestUtils.define_modulations()
        
        return get_topology(
            str(topo_path),
            topology_name,
            mods,
            max_span_length=80,
            default_attenuation=0.2,
            default_noise_figure=4.5,
            k_paths=5,
        )

    @staticmethod
    def create_environment_args(
        topology_name: str,
        qot_constraint: str,
        seed: int = 0,
        bit_rates: Tuple[int, ...] = (10, 40, 100, 400, 1000),
        load: int = 250,
        num_spectrum_resources: int = 320,
        episode_length: int = 1000,
        defragmentation: bool = False,
        k_paths: int = 5,
        gen_observation: bool = False,
    ) -> dict:
        """Cria argumentos do ambiente baseado no utils.py OFC_2025."""
        
        topology = QoTTestUtils.get_topology(topology_name, qot_constraint)
        
        random.seed(seed)
        np.random.seed(seed)

        return dict(
            topology=topology,
            seed=seed,
            allow_rejection=True,
            load=load,
            episode_length=episode_length,
            num_spectrum_resources=num_spectrum_resources,
            launch_power_dbm=1.0,  # best_launch_power
            frequency_slot_bandwidth=12.5e9,
            frequency_start=3e8 / 1565e-9,
            bandwidth=num_spectrum_resources * 12.5e9,
            bit_rate_selection="discrete",
            bit_rates=bit_rates,
            margin=0,
            measure_disruptions=False,
            file_name="",
            k_paths=k_paths,
            modulations_to_consider=6,
            defragmentation=defragmentation,
            n_defrag_services=0,
            gen_observation=gen_observation,
            qot_constraint=qot_constraint
        )


def run_qot_test(
    n_eval_episodes: int,
    heuristic_idx: int,
    monitor_file_name: str,
    topology_name: str,
    qot_constraint: str,
    load: float,
    defragmentation: bool,
    seed: int = 50
) -> None:
    """Executa teste para uma configuração específica."""
    
    # Importações dentro da função para evitar problemas com multiprocessing
    from optical_networking_gym.heuristics.heuristics import (
        heuristic_shortest_available_path_first_fit_best_modulation,
        heuristic_highest_snr,
        heuristic_lowest_fragmentation,
    )
    
    # Criar argumentos do ambiente
    env_args = QoTTestUtils.create_environment_args(
        topology_name=topology_name,
        qot_constraint=qot_constraint,
        seed=seed,
        load=load,
        defragmentation=defragmentation,
        gen_observation=False
    )
    
    # Seleção da heurística
    if heuristic_idx == 1:
        fn_heuristic = heuristic_shortest_available_path_first_fit_best_modulation
    elif heuristic_idx == 2:
        fn_heuristic = heuristic_highest_snr
    elif heuristic_idx == 3:
        fn_heuristic = heuristic_lowest_fragmentation
    else:
        raise ValueError(f"Heuristic index `{heuristic_idx}` not supported!")

    # Criar ambiente
    env = QRMSAEnvWrapper(**env_args)
    
    # Definir nome do arquivo de saída conforme padrão solicitado
    output_file = (
        f"results/load_episodes_1_{qot_constraint.lower().replace('+', '')}_"
        f"def_{defragmentation}_0_{topology_name}_1.0_{load}.0_nw_cnr_nobel-eu.csv"
    )
    
    # Criar diretório se não existir
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    print(f"Iniciando simulação: {qot_constraint}, Load={load}, Defrag={defragmentation}")
    print(f"Arquivo de saída: {output_file}")
    
    # Executar simulação e salvar resultados
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        # Header
        f.write(f"# Date: {datetime.now()}\n")
        header = [
            "episode",
            "service_blocking_rate", "episode_service_blocking_rate",
            "bit_rate_blocking_rate", "episode_bit_rate_blocking_rate",
            "episode_service_realocations", "episode_defrag_cicles"
        ]
        # Adicionar modulações
        for mod in env.env.modulations:
            header.append(f"modulation_{mod.spectral_efficiency}")
        header.extend([
            "episode_disrupted_services",
            "fragmentation_shannon_entropy",
            "fragmentation_route_cuts", 
            "fragmentation_route_rss",
            "episode_time",
            "mean_gsnr"
        ])
        f.write(",".join(header) + "\n")

        # Executar episódios
        for ep in range(n_eval_episodes):
            start_time = time.time()
            obs, info = env.reset()
            done = False

            while not done:
                action, _, _ = fn_heuristic(env)
                _, _, done, _, info = env.step(action)

            ep_time = time.time() - start_time
            
            # Calcular GSNR médio
            mean_gsnr = 0.0
            services = env.env.topology.graph.get("services", [])
            if services:
                mean_gsnr = sum(service.OSNR for service in services) / len(services)

            # Montar linha de dados
            row = [
                ep,
                info.get("service_blocking_rate", 0.0),
                info.get("episode_service_blocking_rate", 0.0),
                info.get("bit_rate_blocking_rate", 0.0),
                info.get("episode_bit_rate_blocking_rate", 0.0),
                info.get("episode_service_realocations", 0),
                info.get("episode_defrag_cicles", 0),
            ]
            
            # Adicionar modulações
            for mod in env.env.modulations:
                row.append(info.get(f"modulation_{mod.spectral_efficiency}", 0))
            
            row.extend([
                info.get("episode_disrupted_services", 0),
                info.get("fragmentation_shannon_entropy", 0.0),
                info.get("fragmentation_route_cuts", 0),
                info.get("fragmentation_route_rss", 0.0),
                f"{ep_time:.2f}",
                f"{mean_gsnr:.4f}"
            ])
            
            f.write(",".join(map(str, row)) + "\n")
            print(f"  Episódio {ep+1}/{n_eval_episodes} concluído")

    print(f"Simulação concluída: {output_file}")


def main():
    """Função principal para executar todos os testes."""
    
    # Parse de argumentos da linha de comando
    parser = argparse.ArgumentParser(description='QoT Constraints Test')
    parser.add_argument(
        '-th', '--threads',
        type=int,
        default=16,
        help='Number of threads to be used for running simulations (default: 1)'
    )
    parser.add_argument(
        '-e', '--episodes',
        type=int,
        default=100,
        help='Number of episodes per simulation (default: 30)'
    )
    parser.add_argument(
        '-s', '--seed',
        type=int,
        default=5,
        help='Random seed (default: 50)'
    )
    args = parser.parse_args()
    
    # Configuração de logging
    logging.getLogger("rmsaenv").setLevel(logging.INFO)
    np.set_printoptions(linewidth=np.inf)
    
    # Parâmetros de teste
    topology_name = "nobel-eu"
    best_launch_power = 1.0
    loads = [100, 200, 300, 400]  # 4 cargas diferentes
    qot_constraints = ["ASE", "ASE+NLI", "DIST"]  # 3 QoT constraints
    defragmentation_options = [False, True]  # Com e sem defragmentação
    n_eval_episodes = args.episodes
    seed = args.seed
    threads = args.threads
    
    # Definir seed global
    random.seed(seed)
    np.random.seed(seed)
    
    print("=== Teste de QoT Constraints ===")
    print(f"Topologia: {topology_name}")
    print(f"Cargas: {loads}")
    print(f"QoT Constraints: {qot_constraints}")
    print(f"Defragmentação: {defragmentation_options}")
    print(f"Episódios por configuração: {n_eval_episodes}")
    print(f"Threads: {threads}")
    print("=" * 40)
    
    # Lista para armazenar todos os argumentos de simulação
    simulation_args = []
    
    # Gerar todas as combinações
    for load in loads:
        for qot_constraint in qot_constraints:
            for defragmentation in defragmentation_options:
                args = (
                    n_eval_episodes,
                    1,  # heuristic_idx (shortest path first fit)
                    f"load_episodes_1_{qot_constraint.lower().replace('+', '')}",
                    topology_name,
                    qot_constraint,
                    load,
                    defragmentation,
                    seed
                )
                simulation_args.append(args)
    
    print(f"Total de simulações: {len(simulation_args)}")
    
    # Executar simulações com ou sem multiprocessing baseado na contagem de threads
    if threads > 1:
        print(f"Executando simulações em paralelo com {threads} threads...")
        with Pool(processes=threads) as pool:
            # Usando starmap para mapear múltiplos argumentos
            pool.starmap(run_qot_test, simulation_args)
    else:
        print("Executando simulações sequencialmente...")
        # Executar simulações sequencialmente
        for i, args in enumerate(simulation_args, 1):
            print(f"\n[{i}/{len(simulation_args)}] Executando simulação...")
            run_qot_test(*args)
    
    print("\n=== Todas as simulações concluídas! ===")
    print("Resultados salvos em:")
    for load in loads:
        for qot_constraint in qot_constraints:
            for defragmentation in defragmentation_options:
                filename = (
                    f"results/load_episodes_1_{qot_constraint.lower().replace('+', '')}_"
                    f"def_{defragmentation}_0_{topology_name}_{best_launch_power}_{load}.0_nw_cnr_nobel-eu.csv"
                )
                print(f"  - {filename}")


if __name__ == "__main__":
    main()
