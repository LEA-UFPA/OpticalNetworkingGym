"""
Teste de Rede Óptica Multibanda - Versão Corrigida
Baseado na estrutura funcional do graph_load.py
"""
import argparse
import logging
import os
import random
from typing import List, Tuple
from multiprocessing import Pool, Manager
import queue
import threading

import numpy as np
import time
from datetime import datetime
from tqdm import tqdm

from optical_networking_gym.topology import Modulation, get_topology
from optical_networking_gym.wrappers.qrmsa_gym import QRMSAEnvWrapper
from optical_networking_gym.heuristics.heuristics import (
    shortest_available_path_first_fit_best_modulation_best_band,
    heuristic_shortest_available_path_first_fit_best_modulation,
    get_best_band_path_pso
)

# ===================================================
# Função para definir as cargas com base no nome da topologia
# ===================================================
def get_loads(topology_name: str) -> np.ndarray:
    """Retorna as cargas apropriadas para cada topologia"""
    if topology_name == "nobel-eu.xml":
        return np.arange(700, 2401, 100)  # Usando valores seguros do graph_load.py
    elif topology_name == "germany50.xml":
        return np.arange(100, 1501, 100)
    elif topology_name == "janos-us.xml":
        return np.arange(100, 601, 50)
    elif topology_name == "nsfnet_chen.txt":
        return np.arange(100, 601, 50)
    elif topology_name == "ring_4.txt":
        return np.arange(100, 601, 50)
    else:
        raise ValueError(f"Unknown topology name: {topology_name}")

def get_heuristic_function(heuristic_index: int):
    """Retorna a função heurística baseada no índice"""
    if heuristic_index == 1:
        return shortest_available_path_first_fit_best_modulation_best_band
    elif heuristic_index == 2:
        return shortest_available_path_first_fit_best_modulation_best_band
    elif heuristic_index == 3:
        return get_best_band_path_pso
    else:
        raise ValueError(f"Heuristic index `{heuristic_index}` is not found!")

def print_link_slots(env, stage):
    """Função para debug visual dos slots nos links"""
    print(f"\n=== LINKS {stage} ===")
    for u, v in env.env.topology.edges():
        idx = env.env.topology[u][v]["index"]
        slots = env.env.topology.graph["available_slots"][idx]
        print(f"Link {u}->{v} slots: {''.join(map(str, slots.tolist()))}")

# Configuração de logging, semente e argumentos de entrada
logging.getLogger("rmsaenv").setLevel(logging.INFO)
np.set_printoptions(linewidth=np.inf)

seed = 50
random.seed(seed)
np.random.seed(seed)

def define_modulations() -> Tuple[Modulation, ...]:
    return (
        # BPSK removido — não será considerado nas simulações
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

def run_environment_with_monitoring(
    n_eval_episodes: int,
    heuristic_index: int,
    band_name: str,
    monitor_file_name: str,
    topology,
    seed: int,
    allow_rejection: bool,
    load: float,
    episode_length: int,
    launch_power_dbm: float,
    frequency_slot_bandwidth: float,
    bit_rate_selection: str,
    bit_rates: tuple,
    margin: float,
    file_name: str,
    measure_disruptions: bool,
    defragmentation: bool,
    n_defrag_services: int,
    gen_observation: bool,
    band_specs: List[dict] = None,
    debug: bool = False,
    progress_queue = None,
    worker_id: int = 0
) -> None:
    """
    Executa o ambiente com a heurística especificada e salva os resultados em um arquivo CSV.
    Baseado na função run_environment do graph_load.py
    """
    
    # Configurações do ambiente - Usando band_specs diretamente se fornecido
    if band_specs:
        env_args = dict(
            topology=topology,
            band_specs=band_specs,
            seed=seed,
            allow_rejection=allow_rejection,
            load=load,
            episode_length=episode_length,
            launch_power_dbm=launch_power_dbm,
            frequency_slot_bandwidth=frequency_slot_bandwidth,
            bit_rate_selection=bit_rate_selection,
            bit_rates=bit_rates,
            margin=margin,
            file_name=file_name,
            measure_disruptions=measure_disruptions,
            k_paths=2,
            # modulations_to_consider deixado ao padrão do ambiente
            defragmentation=defragmentation,
            n_defrag_services=n_defrag_services,
            gen_observation=False,  # Forçar False para evitar problemas com high load
        )
    else:
        # Configuração tradicional (backward compatibility)
        env_args = dict(
            topology=topology,
            seed=seed,
            allow_rejection=allow_rejection,
            load=load,
            episode_length=episode_length,
            launch_power_dbm=launch_power_dbm,
            frequency_slot_bandwidth=frequency_slot_bandwidth,
            bit_rate_selection=bit_rate_selection,
            bit_rates=bit_rates,
            margin=margin,
            file_name=file_name,
            measure_disruptions=measure_disruptions,
            k_paths=2,
            modulations_to_consider=5,  # Usar número atual de modulações
            defragmentation=defragmentation,
            n_defrag_services=n_defrag_services,
            gen_observation=False,  # Forçar False para evitar problemas com high load
        )

    # Seleção da heurística baseada no índice
    fn_heuristic = get_heuristic_function(heuristic_index)

    # Criação do ambiente
    env = QRMSAEnvWrapper(**env_args)
    
    # Criar arquivo CSV igual ao graph_load.py
    monitor_final_name = f"load_results_{topology.name}_{band_name}_{load}.csv"
    os.makedirs("results", exist_ok=True)
    
    with open(f"results/{monitor_final_name}", "wt") as f:
        f.write(f"# Date: {datetime.now()}\n")
        # Cabeçalho completo igual ao que é salvo nos dados
        header = (
            "episode,service_blocking_rate,episode_service_blocking_rate,"
            "bit_rate_blocking_rate,episode_bit_rate_blocking_rate,"
            "episode_service_realocations,episode_defrag_cicles"
        )
        for mf in env.env.modulations:
            header += f",modulation_{mf.spectral_efficiency}"
        header += ",episode_disrupted_services,episode_time,mean_gsnr\n"
        f.write(header)

        for ep in range(n_eval_episodes):
            if progress_queue:
                # First time init for this task (or reset)
                # We can send init message every episode, or just once per task.
                # However, since tasks are granular (per load), we can treat each task as a bar reset.
                # But here we loop episodes. Let's make the bar represent the episodes.
                progress_queue.put((worker_id, 'init', n_eval_episodes, f"Worker {worker_id}: {band_name} Load {load}"))
            
            obs, info = env.reset()
            done = False
            start_time = time.time()
            step_count = 0
            
            while not done:
                action, path_id, mod_id = fn_heuristic(env)
                obs, reward, done, truncated, info = env.step(action)

                step_count += 1
                if progress_queue:
                    progress_queue.put((worker_id, 'update', 1))

            if progress_queue:
                progress_queue.put((worker_id, 'done'))
                
            end_time = time.time()
            ep_time = end_time - start_time

            print(f"Episódio {ep} finalizado.")
            print(info)

            # Salvar dados no CSV exatamente como no graph_load.py
            row = (
                f"{ep},{info['service_blocking_rate']},"
                f"{info['episode_service_blocking_rate']},"
                f"{info['bit_rate_blocking_rate']},"
                f"{info['episode_bit_rate_blocking_rate']},"
                f"{info['episode_service_realocations']},"
                f"{info['episode_defrag_cicles']}"
            )
            for mf in env.env.modulations:
                row += f",{info.get(f'modulation_{mf.spectral_efficiency}', 0.0)}"
            row += f",{info.get('episode_disrupted_services', 0)},{ep_time:.2f}"
            mean_gsnr = 0.0
            if len(env.env.topology.graph["services"]) > 0:
                for service in env.env.topology.graph["services"]:
                    mean_gsnr += service.OSNR
                mean_gsnr /= len(env.env.topology.graph["services"])
            row += f",{mean_gsnr}\n"
            f.write(row)

    print(f"\nFinalizado! Resultados salvos em: results/{monitor_final_name}")

def create_environment(topology_name="nobel-eu.xml", episode_length=10, debug=True):
    """Cria o ambiente de teste baseado na configuração do graph_load.py"""
    
    # Determinar o arquivo de topologia
    if topology_name.endswith('.txt'):
        topo_file = f"examples/topologies/{topology_name}"
    else:
        topo_file = f"examples/topologies/{topology_name}.txt"
    
    # Verificar se o arquivo existe
    if not os.path.exists(topo_file):
        raise FileNotFoundError(f"Arquivo de topologia '{topo_file}' não encontrado.")
    
    # Usar modulações completas
    mods = define_modulations()
    
    # Criar topologia
    topology = get_topology(
        topo_file, None, mods,
        max_span_length=100, 
        default_attenuation=0.2,
        default_noise_figure=4.5, 
        k_paths=2  # Reduzido para compatibilidade com redes menores
    )

    # Configuração de banda multibanda baseada no graph_load.py
    band_specs = {
        "BandaC": [
            {"name": "C", "start_thz": 191.60, "num_slots": 344, "noise_figure_db": 5.5, "attenuation_db_km": 0.191}
        ],
        "BandaL": [
            {"name": "L", "start_thz": 185.83, "num_slots": 406, "noise_figure_db": 6.0, "attenuation_db_km": 0.200}
        ],
        "BandaS": [
            {"name": "S", "start_thz": 197.22, "num_slots": 647, "noise_figure_db": 7.0, "attenuation_db_km": 0.220}
        ],
        "BandaC+L": [
            {"name": "L", "start_thz": 185.83, "num_slots": 406, "noise_figure_db": 5.5, "attenuation_db_km": 0.200},
            {"name": "C", "start_thz": 191.60, "num_slots": 344, "noise_figure_db": 6.0, "attenuation_db_km": 0.191}
        ],
        "BandaC+L+S": [
            {"name": "L", "start_thz": 185.83, "num_slots": 406, "noise_figure_db": 5.5, "attenuation_db_km": 0.200},
            {"name": "C", "start_thz": 191.60, "num_slots": 344, "noise_figure_db": 6.0, "attenuation_db_km": 0.191},
            {"name": "S", "start_thz": 197.22, "num_slots": 647, "noise_figure_db": 7.0, "attenuation_db_km": 0.220}
        ]
    }
    
    # Configurações do ambiente usando band_specs (padrão do graph_load.py)
    env_args = dict(
        topology=topology,
        band_specs=band_specs,
        seed=10,
        allow_rejection=True,
        load=1000,
        episode_length=episode_length,
        launch_power_dbm=0,
        frequency_slot_bandwidth=12.5e9,
        bit_rate_selection="discrete",
        bit_rates=(48, 120),
        margin=0,
        measure_disruptions=False,
        file_name="",
        k_paths=2,  # Reduzido para compatibilidade com redes menores
        # modulations_to_consider removido - deixar o ambiente usar o padrão
        defragmentation=False,
        n_defrag_services=0,
        gen_observation=False,  # SOLUÇÃO: Desabilitar observação resolve o IndexError
    )
    
    return env_args, debug


def progress_listener(q, total_episodes_all_tasks):
    """
    Listener for tqdm progress bars.
    """
    bars = {}
    # Global bar
    global_bar = tqdm(total=total_episodes_all_tasks, desc="Total Simulation Progress", position=0)
    
    while True:
        msg = q.get()
        if msg == 'KILL':
            break
            
        worker_id = msg[0]
        action = msg[1]
        
        if action == 'init':
            total = msg[2]
            desc = msg[3]
            if worker_id not in bars:
                # Use position = worker_id + 1 to stack bars below global
                bars[worker_id] = tqdm(total=total, desc=desc, position=worker_id+1, leave=False)
            else:
                bars[worker_id].reset(total=total)
                bars[worker_id].set_description(desc)
                
        elif action == 'update':
            steps = msg[2]
            if worker_id in bars:
                bars[worker_id].update(steps)
            global_bar.update(steps)
            
        elif action == 'done':
            if worker_id in bars:
                bars[worker_id].set_description(f"Worker {worker_id} Idle")
                
    global_bar.close()
    for b in bars.values():
        b.close()


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Teste de Rede Óptica Multibanda')
    parser.add_argument(
        '-t', '--topology_file',
        type=str,
        default='nobel-eu.xml',
        help='Aa'
    )
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
        help='Número de chegadas por episódio (default: 100)'
    )
    parser.add_argument(
        '-hi', '--heuristic_index',
        type=int,
        default=1,
        choices=[1,2,3],
        help='Índice da heurística (1: First fit, 2: shortest_available_path_first_fit_best_modulation_best_band, 3: get_best_band_path_pso)'
    )

    parser.add_argument('-mb', '--bands', nargs='+', type=str, 
                        default=['BandaC+L+S'],
                        choices=['BandaC', 'BandaL', 'BandaS', 'BandaC+L', 'BandaC+L+S'],
                        help='Especifica uma ou mais bandas para simular')
    parser.add_argument(
        '-d', '--debug',
        action='store_true',
        help='Habilita debug visual dos slots'
    )
    parser.add_argument(
        '-p', '--power',
        type=float,
        default=0.0,
        help='Potência de lançamento em dBm (default: 0.0)'
    )
    parser.add_argument(
        '-th', '--threads',
        type=int,
        default=os.cpu_count(),
        help='Número de processos/threads para execução paralela (default: todos os cores)'
    )
    
    return parser.parse_args()

def main():
    """Função principal com opções de linha de comando baseada no graph_load.py"""
    args = parse_arguments()
    
    # Print Simulation Parameters
    print("\n" + "="*50)
    print(f"  Optical Networking Gym - Simulation Configuration")
    print("="*50)
    print(f"  Topology File    : {args.topology_file}")
    print(f"  Bands            : {', '.join(args.bands)}")
    print(f"  Episodes         : {args.num_episodes}")
    print(f"  Episode Length   : {args.episode_length}")
    print(f"  Heuristic Index  : {args.heuristic_index}")
    print(f"  Launch Power     : {args.power} dBm")
    print(f"  Threads          : {args.threads}")
    print(f"  Debug Mode       : {args.debug}")
    print("="*50 + "\n")

    # Usar modulações completas
    cur_modulations = define_modulations()

    # Determinar o caminho da topologia (caminho absoluto baseado no diretório do script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    topologies_dir = os.path.join(script_dir, "..", "topologies")
    
    if args.topology_file.endswith('.xml'):
        topology_path = os.path.join(topologies_dir, args.topology_file)
    else:
        topology_path = os.path.join(topologies_dir, args.topology_file)
        
    if not os.path.exists(topology_path):
        raise FileNotFoundError(f"Arquivo de topologia '{topology_path}' não encontrado.")

    # Configurações de banda com slots uniformes
    
    band_specs_all = {
        "BandaC": [{"name": "C", "start_thz": 191.60, "num_slots": 344, "noise_figure_db": 5.5, "attenuation_db_km": 0.191}],
        "BandaL": [{"name": "L", "start_thz": 185.83, "num_slots": 406, "noise_figure_db": 6.0, "attenuation_db_km": 0.200}],
        "BandaS": [{"name": "S", "start_thz": 197.22, "num_slots": 647, "noise_figure_db": 7.0, "attenuation_db_km": 0.220}],
        "BandaC+L": [   
            {"name": "L", "start_thz": 185.83, "num_slots": 406, "noise_figure_db": 5.5, "attenuation_db_km": 0.200},
            {"name": "C", "start_thz": 191.60, "num_slots": 344, "noise_figure_db": 6.0, "attenuation_db_km": 0.191},
        ],
        "BandaC+L+S": [
            {"name": "L", "start_thz": 185.83, "num_slots": 406, "noise_figure_db": 5.5, "attenuation_db_km": 0.200},
            {"name": "C", "start_thz": 191.60, "num_slots": 344, "noise_figure_db": 6.0, "attenuation_db_km": 0.191},
            {"name": "S", "start_thz": 197.22, "num_slots": 647, "noise_figure_db": 7.0, "attenuation_db_km": 0.220},
        ]
    }

    # Obter as cargas baseadas na topologia
    loads = get_loads(args.topology_file)

    # Create a list of tasks for multiprocessing
    tasks = []

    for band_name in args.bands:
        band_specs = band_specs_all[band_name]
        
        # Criar topologia
        topology = get_topology(
            topology_path,
            None,
            cur_modulations,
            100,  # max_span_length
            0.2,  # default_attenuation
            4.5,  # default_noise_figure
            2  # k_paths
        )
        
        # Iterar sobre todas as cargas da topologia
        for current_load in loads:
            # Append task arguments tuple
            tasks.append((
                args.num_episodes,
                args.heuristic_index,
                band_name,
                None, # monitor_file_name
                topology,
                seed,
                True, # allow_rejection
                current_load,
                args.episode_length,
                args.power,
                12.5e9, # frequency_slot_bandwidth
                "discrete", # bit_rate_selection
                (48, 120), # bit_rates
                0, # margin
                "", # file_name
                False, # measure_disruptions
                False, # defragmentation
                0, # n_defrag_services
                False, # gen_observation
                band_specs,
                args.debug
            ))

    # Determine number of processes to use 
    # Use user provided threads or fallback to safe minimum
    num_processes = args.threads
    if num_processes < 1:
        num_processes = 1
        
    # Setup Manager and Queue
    manager = Manager()
    queue = manager.Queue()
    
    # Calculate total episodes
    total_episodes_global = sum(t[0] for t in tasks)
    
    # Start listener
    monitor_thread = threading.Thread(target=progress_listener, args=(queue, total_episodes_global))
    monitor_thread.start()
    
    # Worker ID management
    worker_id_queue = manager.Queue()
    for i in range(num_processes):
        worker_id_queue.put(i)
        
    def worker_init(q):
        global process_worker_id
        process_worker_id = q.get()

    # Re-build tasks to include queue and placeholder
    updated_tasks = []
    for t in tasks:
        l = list(t)
        l.append(queue)
        l.append(None) # Placeholder for worker_id
        updated_tasks.append(tuple(l))

    print(f"Iniciando simulação com {num_processes} processos para {len(tasks)} tarefas...")
    
    # Helper wrapper to inject worker_id
    global run_wrapper
    def run_wrapper(*args):
        args_list = list(args)
        args_list[-1] = process_worker_id
        return run_environment_with_monitoring(*args_list)

    with Pool(processes=num_processes, initializer=worker_init, initargs=(worker_id_queue,)) as pool:
        pool.starmap(run_wrapper, updated_tasks)

    # Stop listener
    queue.put('KILL')
    monitor_thread.join()

    print("\nTodas as simulações foram executadas.")

# Global to store worker ID in each process
process_worker_id = 0

if __name__ == "__main__":
    main()