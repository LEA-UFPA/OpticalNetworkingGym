"""
Teste de Rede Óptica Multibanda - Versão Completa com Multiprocessing
Baseado na estrutura funcional do graph_load.py

Features:
- Suporte para múltiplas bandas (C, L, S) e combinações
- Heurísticas específicas para multiband (only heuristics that work with multiband)
- Progress bars com tqdm para melhor visualização
- Processamento paralelo com multiprocessing para simular múltiplas cargas simultaneamente
- Resultados salvos em CSV com metadados completos

Available Multiband Heuristics:
    5: Priority Band C then L (não funciona corretamente ainda)
    6: Shortest Path First-Fit Best Modulation Best Band (RECOMENDADO - funciona corretamente)

Usage examples:
    # Sequencial (1 worker) - usando heurística 6 (recomendada)
    python multiband_sim.py -e 5 -hi 6 -t nobel-eu.xml -b C_L_S
    
    # Paralelo com 4 workers
    python multiband_sim.py -e 5 -hi 6 -t nobel-eu.xml -b C_L_S -w 4
    
    # Paralelo usando todos os CPUs disponíveis
    python multiband_sim.py -e 5 -hi 6 -t nobel-eu.xml -b C_L_S -w -1
    
    # Usando apenas banda C
    python multiband_sim.py -e 5 -hi 6 -t nobel-eu.xml -b C -w 4
"""
import argparse
import logging
import os
import random
import time
from datetime import datetime
from typing import List, Tuple
from multiprocessing import Pool, cpu_count

import numpy as np
from tqdm import tqdm

from optical_networking_gym.topology import Modulation, get_topology
from optical_networking_gym.wrappers.qrmsa_gym import QRMSAEnvWrapper
from optical_networking_gym.heuristics.multiband_heuristics import (
    heuristic_priority_band_C_then_L,
    shortest_available_path_first_fit_best_modulation_best_band,
)

# ===================================================
# Constantes
# ===================================================
SEED = 50
DEFAULT_TOPOLOGY = "nobel-eu.xml"
K_PATHS = 2
MODULATIONS_TO_CONSIDER = 2
MAX_SPAN_LENGTH = 100.0
DEFAULT_ATTENUATION = 0.2
DEFAULT_NOISE_FIGURE = 4.5
FREQUENCY_SLOT_BANDWIDTH = 12.5e9
RESULTS_DIR = "results"

# Band specifications
BAND_C = {"name": "C", "start_thz": 191.60, "num_slots": 344, "noise_figure_db": 5.5, "attenuation_db_km": 0.191}
BAND_L = {"name": "L", "start_thz": 185.83, "num_slots": 406, "noise_figure_db": 6.0, "attenuation_db_km": 0.200}
BAND_S = {"name": "S", "start_thz": 197.22, "num_slots": 647, "noise_figure_db": 7.0, "attenuation_db_km": 0.220}

BAND_SPECS_C_L_S = [BAND_C, BAND_L, BAND_S]
BAND_SPECS_C_L = [BAND_C, BAND_L]
BAND_SPECS_L_C_S = [BAND_L, BAND_C, BAND_S]
BAND_SPECS_C = [BAND_C]
BAND_SPECS_L = [BAND_L]
BAND_SPECS_S = [BAND_S]

# Band specifications mapping
BAND_SPECS_MAP = {
    'C_L_S': BAND_SPECS_C_L_S,
    'C_L': BAND_SPECS_C_L,
    'L_C_S': BAND_SPECS_L_C_S,
    'C': BAND_SPECS_C,
    'L': BAND_SPECS_L,
    'S': BAND_SPECS_S,
}

# ===================================================
# Configuração
# ===================================================
logging.getLogger("rmsaenv").setLevel(logging.INFO)
np.set_printoptions(linewidth=np.inf)

random.seed(SEED)
np.random.seed(SEED)

# ===================================================
# Heuristic mapping (Multiband-specific only)
# ===================================================
HEURISTIC_FUNCTIONS = {
    5: heuristic_priority_band_C_then_L,  # NOTE: This heuristic does not work correctly yet
    6: shortest_available_path_first_fit_best_modulation_best_band,  # Recommended - Works correctly
}


def get_heuristic_function(heuristic_index: int):
    """Retorna a função heurística baseada no índice"""
    if heuristic_index not in HEURISTIC_FUNCTIONS:
        raise ValueError(f"Heuristic index `{heuristic_index}` is not found! Available: {list(HEURISTIC_FUNCTIONS.keys())}")
    return HEURISTIC_FUNCTIONS[heuristic_index]

# Topology load configurations
TOPOLOGY_LOADS = {
    "nobel-eu.xml": (400, 1401, 100),
    "germany50.xml": (100, 1501, 100),
    "janos-us.xml": (100, 601, 50),
    "nsfnet_chen.txt": (100, 601, 50),
    "ring_4.txt": (100, 601, 50),
}


def get_loads(topology_name: str) -> np.ndarray:
    """
    Retorna as cargas apropriadas para cada topologia.
    
    Args:
        topology_name: Nome do arquivo de topologia
        
    Returns:
        Array numpy com os valores de carga para a topologia
        
    Raises:
        ValueError: Se o nome da topologia não for reconhecido
    """
    if topology_name not in TOPOLOGY_LOADS:
        available = ", ".join(TOPOLOGY_LOADS.keys())
        raise ValueError(
            f"Unknown topology name: '{topology_name}'. "
            f"Available topologies: {available}"
        )
    
    start, stop, step = TOPOLOGY_LOADS[topology_name]
    return np.arange(start, stop, step)


def get_band_specs(band_specs_key: str) -> List[dict]:
    """Retorna as especificações de banda baseadas na chave"""
    if band_specs_key not in BAND_SPECS_MAP:
        raise ValueError(f"Unknown band_specs key: {band_specs_key}. Available: {list(BAND_SPECS_MAP.keys())}")
    return BAND_SPECS_MAP[band_specs_key]

def define_modulations() -> Tuple[Modulation, ...]:
    return (
        Modulation( 
            name="QPSK",
            maximum_length=10_000,
            spectral_efficiency=2,
            minimum_osnr=6.72,
            inband_xt=-17,
        ),
        Modulation(
            name="8QAM",
            maximum_length=1_000,
            spectral_efficiency=3,
            minimum_osnr=10.84,
            inband_xt=-20,
        ),
        Modulation(
            name="16QAM",
            maximum_length=500,
            spectral_efficiency=4,
            minimum_osnr=13.24,
            inband_xt=-23,
        ),
        Modulation(
            name="32QAM",
            maximum_length=250,
            spectral_efficiency=5,
            minimum_osnr=16.16,
            inband_xt=-26,
        ),
        Modulation(
            name="64QAM",
            maximum_length=125,
            spectral_efficiency=6,
            minimum_osnr=19.01,
            inband_xt=-29,
        ),
    )

def write_csv_header(f, topology, heuristic_index, n_eval_episodes, episode_length, 
                     bit_rates, band_specs, frequency_slot_bandwidth, max_span_length, modulations):
    """Escreve o cabeçalho do arquivo CSV com metadados da simulação"""
    f.write(f"# Date: {datetime.now()}\n")
    f.write(f"# Topology: {topology.name}\n")
    f.write(f"# Heuristic: {get_heuristic_function(heuristic_index).__name__}\n")
    f.write(f"# Episodes: {n_eval_episodes}\n")
    f.write(f"# Episode Length: {episode_length}\n")
    f.write(f"# Bit Rates: {bit_rates} Gbps\n")
    f.write(f"# Modulations: {[m.name for m in modulations]}\n")
    f.write(f"# Span Length: {max_span_length} km\n")
    f.write(f"\n# Band Specifications\n")
    
    band_names = [spec['name'] for spec in band_specs]
    f.write(f"# | | {" | ".join(band_names)} |\n")
    f.write(f"# |---|{"---|" * len(band_specs)}\n")
    
    start_thz_row = " | ".join([str(spec['start_thz']) for spec in band_specs])
    f.write(f"# | Frequência Inicial (Thz) | {start_thz_row} |\n")
    
    noise_figure_row = " | ".join([str(spec['noise_figure_db']) for spec in band_specs])
    f.write(f"# | Figura de Ruído (dB) | {noise_figure_row} |\n")
    
    attenuation_row = " | ".join([str(spec['attenuation_db_km']) for spec in band_specs])
    f.write(f"# | Coeficiente de Atenuação (dB/km) | {attenuation_row} |\n")
    
    slots_row = " | ".join([str(spec['num_slots']) for spec in band_specs])
    f.write(f"# | Número de Slots | {slots_row} |\n")
    
    calculated_bw = [spec['num_slots'] * frequency_slot_bandwidth for spec in band_specs]
    bw_row = " | ".join([f'{bw:.2e}' for bw in calculated_bw])
    f.write(f"# | Bandwidth (Hz) | {bw_row} |\n\n")


def write_csv_data_header(f, modulations):
    """Escreve o cabeçalho dos dados do CSV"""
    header = (
        "episode,service_blocking_rate,episode_service_blocking_rate,"
        "bit_rate_blocking_rate,episode_bit_rate_blocking_rate,"
        "episode_service_realocations,episode_defrag_cicles"
    )
    for mf in modulations:
        header += f",modulation_{mf.spectral_efficiency}"
    header += ",episode_disrupted_services,episode_time,mean_gsnr\n"
    f.write(header)


def calculate_mean_gsnr(env):
    """Calcula a média do GSNR dos serviços ativos"""
    services = env.env.topology.graph["services"]
    if len(services) == 0:
        return 0.0
    return sum(service.OSNR for service in services) / len(services)


def write_episode_results(f, episode, info, modulations, episode_time, mean_gsnr):
    """Escreve os resultados de um episódio no CSV"""
    row = (
        f"{episode},{info['service_blocking_rate']},"
        f"{info['episode_service_blocking_rate']},"
        f"{info['bit_rate_blocking_rate']},"
        f"{info['episode_bit_rate_blocking_rate']},"
        f"{info['episode_service_realocations']},"
        f"{info['episode_defrag_cicles']}"
    )
    for mf in modulations:
        row += f",{info.get(f'modulation_{mf.spectral_efficiency}', 0.0)}"
    row += f",{info.get('episode_disrupted_services', 0)},{episode_time:.2f},{mean_gsnr}\n"
    f.write(row)


def run_environment_with_monitoring(
    n_eval_episodes: int,
    heuristic_index: int,
    topology,
    load: float,
    episode_length: int,
    launch_power_dbm: float,
    bit_rates: tuple,
    band_specs: List[dict],
    max_span_length: float = MAX_SPAN_LENGTH,
    show_progress: bool = True
) -> None:
    """
    Executa o ambiente com a heurística especificada e salva os resultados em um arquivo CSV.
    """
    # Configurações do ambiente
    env_args = dict(
        topology=topology,
        band_specs=band_specs,
        seed=SEED,
        allow_rejection=True,
        load=load,
        episode_length=episode_length,
        launch_power_dbm=launch_power_dbm,
        frequency_slot_bandwidth=FREQUENCY_SLOT_BANDWIDTH,
        bit_rate_selection="discrete",
        bit_rates=bit_rates,
        margin=0,
        file_name="",
        measure_disruptions=False,
        k_paths=K_PATHS,
        modulations_to_consider=MODULATIONS_TO_CONSIDER,
        defragmentation=False,
        n_defrag_services=0,
        gen_observation=False,  # Forçar False para evitar problemas com high load
    )

    # Seleção da heurística baseada no índice
    fn_heuristic = get_heuristic_function(heuristic_index)

    # Criação do ambiente
    env = QRMSAEnvWrapper(**env_args)
    
    # Criar arquivo CSV
    monitor_final_name = f"load_results_{topology.name}_{load}.csv"
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    with open(f"{RESULTS_DIR}/{monitor_final_name}", "wt") as f:
        write_csv_header(
            f, topology, heuristic_index, n_eval_episodes, episode_length,
            bit_rates, band_specs, FREQUENCY_SLOT_BANDWIDTH, max_span_length, env.env.modulations
        )
        write_csv_data_header(f, env.env.modulations)

        # Create progress bar for episodes
        episode_iterator = range(n_eval_episodes)
        if show_progress:
            episode_iterator = tqdm(
                episode_iterator,
                desc=f"Load {load}",
                unit="ep",
                leave=False
            )
        
        for ep in episode_iterator:
            env.reset()
            done = False
            start_time = time.time()
            
            while not done:
                action, _, _ = fn_heuristic(env)
                _, _, done, _, info = env.step(action)
                
            end_time = time.time()
            episode_time = end_time - start_time

            # Print episode completion info
            if show_progress:
                tqdm.write(f"Episódio {ep} finalizado.")
                tqdm.write(str(info))
            else:
                print(f"Episódio {ep} finalizado.")
                print(info)

            mean_gsnr = calculate_mean_gsnr(env)
            write_episode_results(f, ep, info, env.env.modulations, episode_time, mean_gsnr)

    if show_progress:
        tqdm.write(f"Finalizado! Resultados salvos em: {RESULTS_DIR}/{monitor_final_name}")
    else:
        print(f"\nFinalizado! Resultados salvos em: {RESULTS_DIR}/{monitor_final_name}")



def run_single_load(args_tuple):
    """
    Wrapper function for multiprocessing. Runs a single load simulation.
    
    Args:
        args_tuple: Tuple containing all arguments for run_environment_with_monitoring
    """
    (n_eval_episodes, heuristic_index, topology, load, episode_length,
     launch_power_dbm, bit_rates, band_specs, max_span_length) = args_tuple
    
    run_environment_with_monitoring(
        n_eval_episodes=n_eval_episodes,
        heuristic_index=heuristic_index,
        topology=topology,
        load=load,
        episode_length=episode_length,
        launch_power_dbm=launch_power_dbm,
        bit_rates=bit_rates,
        band_specs=band_specs,
        max_span_length=max_span_length,
        show_progress=True
    )
    
    return load


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Teste de Rede Óptica Multibanda')
    parser.add_argument(
        '-e', '--num_episodes',
        type=int,
        default=5,
        help='Número de episódios a serem simulados (default: 5)'
    )
    parser.add_argument(
        '-s', '--episode_length',
        type=int,
        default=100000,
        help='Número de chegadas por episódio (default: 100000)'
    )
    parser.add_argument(
        '-hi', '--heuristic_index',
        type=int,
        default=6,
        choices=list(HEURISTIC_FUNCTIONS.keys()),
        help='Índice da heurística multiband (5: Prioridade Banda C então L [não funciona], 6: Multibanda Best Band [RECOMENDADO])'
    )
    parser.add_argument(
        '-p', '--power',
        type=float,
        default=0.0,
        help='Potência de lançamento em dBm (default: 0.0)'
    )
    parser.add_argument(
        '-b', '--band_specs',
        type=str,
        default='C_L_S',
        choices=['C_L_S', 'C_L', 'L_C_S', 'C', 'L', 'S'],
        help='Especificações de bandas. Opções: C_L_S (C, L, S), C_L (C, L), L_C_S (L, C, S), C, L, S (default: C_L_S)'
    )
    parser.add_argument(
        '-t', '--topology',
        type=str,
        default='nobel-eu.xml',
        choices=['nobel-eu.xml', 'germany50.xml', 'janos-us.xml', 'nsfnet_chen.txt', 'ring_4.txt'],
        help='Nome da topologia. Suportadas: nobel-eu.xml, germany50.xml, janos-us.xml, nsfnet_chen.txt, ring_4.txt (default: nobel-eu.xml)'
    )
    parser.add_argument(
        '-w', '--workers',
        type=int,
        default=1,
        help='Número de workers para processamento paralelo. Use 1 para desabilitar paralelização, -1 para usar todos os CPUs disponíveis (default: 1)'
    )
    
    return parser.parse_args()

def main():
    """Função principal com opções de linha de comando"""
    args = parse_arguments()

    # Configurar modulações
    modulations = define_modulations()

    # Determinar o caminho da topologia
    script_dir = os.path.dirname(os.path.abspath(__file__))
    topologies_dir = os.path.join(script_dir, "..", "topologies")
    topology_path = os.path.join(topologies_dir, args.topology)
        
    if not os.path.exists(topology_path):
        raise FileNotFoundError(f"Arquivo de topologia '{topology_path}' não encontrado.")

    # Obter as cargas para a topologia
    loads = get_loads(args.topology)
    
    # Obter as especificações de banda
    band_specs = get_band_specs(args.band_specs)
    
    # Criar topologia
    topology = get_topology(
        topology_path,
        None,
        modulations,
        MAX_SPAN_LENGTH,
        DEFAULT_ATTENUATION,
        DEFAULT_NOISE_FIGURE,
        K_PATHS
    )
    
    # Determine number of workers
    num_workers = args.workers
    if num_workers == -1:
        num_workers = cpu_count()
    
    # Execute simulations
    if num_workers > 1:
        # Parallel execution
        print(f"\nExecutando simulações em paralelo com {num_workers} workers...")
        print(f"Total de {len(loads)} cargas a serem processadas.\n")
        
        # Prepare arguments for each load
        load_args = [
            (
                args.num_episodes,
                args.heuristic_index,
                topology,
                current_load,
                args.episode_length,
                args.power,
                (48, 120),
                band_specs,
                MAX_SPAN_LENGTH
            )
            for current_load in loads
        ]
        
        with Pool(processes=num_workers) as pool:
            # Use imap_unordered with tqdm for progress tracking
            list(tqdm(
                pool.imap_unordered(run_single_load, load_args),
                total=len(load_args),
                desc="Total Progress",
                unit="load"
            ))
    else:
        # Sequential execution (original simple approach)
        print(f"\nExecutando simulações sequencialmente...")
        print(f"Total de {len(loads)} cargas a serem processadas.\n")
        
        # Iterar sobre todas as cargas da topologia
        for current_load in tqdm(loads, desc="Total Progress", unit="load"):
            run_environment_with_monitoring(
                n_eval_episodes=args.num_episodes,
                heuristic_index=args.heuristic_index,
                topology=topology,
                load=current_load,
                episode_length=args.episode_length,
                launch_power_dbm=args.power,
                bit_rates=(48, 120),
                band_specs=band_specs,
                max_span_length=MAX_SPAN_LENGTH
            )

    print("\n✅ Todas as simulações foram executadas com sucesso!")

if __name__ == "__main__":
    main()