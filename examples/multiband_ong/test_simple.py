"""
Teste Simples de Rede Óptica Multibanda - Versão Corrigida
Baseado na estrutura funcional do graph_load.py
"""
import argparse
import logging
import os
import random
from typing import List, Tuple
from multiprocessing import Pool

import numpy as np
import time
from datetime import datetime
from tqdm import tqdm

from optical_networking_gym.topology import Modulation, get_topology
from optical_networking_gym.wrappers.qrmsa_gym import QRMSAEnvWrapper
from optical_networking_gym.heuristics.heuristics import (
    shortest_available_path_first_fit_best_modulation,
    shortest_available_path_lowest_spectrum_best_modulation,
    best_modulation_load_balancing,
    load_balancing_best_modulation,
    rnd,
    heuristic_shortest_available_path_first_fit_best_modulation,
    heuristic_highest_snr,
    heuristic_lowest_fragmentation,
    heuristic_priority_band_C_then_L,
    shortest_available_path_first_fit_best_modulation_best_band,
)

# ===================================================
# Função para definir as cargas com base no nome da topologia
# ===================================================
def get_loads(topology_name: str) -> np.ndarray:
    """Retorna as cargas apropriadas para cada topologia"""
    if topology_name == "nobel-eu.xml":
        return np.arange(300, 901, 50)
    elif topology_name == "germany50.xml":
        return np.arange(300, 801, 50)
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
        return heuristic_shortest_available_path_first_fit_best_modulation
    elif heuristic_index == 2:
        return heuristic_highest_snr
    elif heuristic_index == 3:
        return heuristic_lowest_fragmentation
    elif heuristic_index == 4:
        return load_balancing_best_modulation
    elif heuristic_index == 5:
        return heuristic_priority_band_C_then_L
    elif heuristic_index == 6:
        return shortest_available_path_first_fit_best_modulation_best_band
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
        Modulation(
            name="BPSK",
            maximum_length=100_000,
            spectral_efficiency=1,
            minimum_osnr=3.71,
            inband_xt=-14,
        ),
        Modulation(
            name="QPSK",
            maximum_length=2_000,
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

def define_modulations_simplified() -> Tuple[Modulation, ...]:
    """Modulações simplificadas para teste
    
    NOTA: Se você quiser usar todas as 6 modulações COM observações habilitadas (gen_observation=True),
    use apenas as duas primeiras modulações para evitar o IndexError em decimal_to_array.
    Com 6 modulações, use gen_observation=False.
    """
    return (
        Modulation("BPSK", 100_000, 1, minimum_osnr=3.71, inband_xt=-14),
        Modulation("QPSK", 2_000, 2, minimum_osnr=6.72, inband_xt=-17),
        Modulation("8QAM", 1_000, 3, minimum_osnr=10.84, inband_xt=-20),
        Modulation("16QAM", 500, 4, minimum_osnr=13.24, inband_xt=-23),
        Modulation("32QAM", 250, 5, minimum_osnr=16.16, inband_xt=-26),
        Modulation("64QAM", 125, 6, minimum_osnr=19.01, inband_xt=-29),
    )

def print_link_slots(env, stage):
    """Função para debug visual dos slots nos links"""
    print(f"\n=== LINKS {stage} ===")
    for u, v in env.env.topology.edges():
        idx = env.env.topology[u][v]["index"]
        slots = env.env.topology.graph["available_slots"][idx]
        print(f"Link {u}->{v} slots: {''.join(map(str, slots.tolist()))}")

def run_environment_with_monitoring(
    n_eval_episodes: int,
    heuristic_index: int,
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
            k_paths=5 if topology.number_of_nodes() > 4 else 2,
            modulations_to_consider=6,
            defragmentation=defragmentation,
            n_defrag_services=n_defrag_services,
            gen_observation=gen_observation,
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
            k_paths=5 if topology.number_of_nodes() > 4 else 2,
            modulations_to_consider=6,
            defragmentation=defragmentation,
            n_defrag_services=n_defrag_services,
            gen_observation=gen_observation,
        )

    # Seleção da heurística baseada no índice
    fn_heuristic = get_heuristic_function(heuristic_index)

    # Criação do ambiente
    env = QRMSAEnvWrapper(**env_args)
    env.reset()

    if monitor_file_name:
        # Definição do nome final do arquivo CSV de monitoramento
        monitor_final_name = "_".join([
            monitor_file_name, 
            topology.name, 
            str(env.env.launch_power_dbm), 
            str(env.env.load) + "_test_simple.csv"
        ])

        # Crie o diretório se não existir
        os.makedirs(os.path.dirname(monitor_final_name), exist_ok=True)

        # Preparação do arquivo CSV
        with open(monitor_final_name, "wt", encoding="UTF-8") as file_handler:
            file_handler.write(f"# Date: {datetime.now()}\n")
            header = (
                "episode,service_blocking_rate,episode_service_blocking_rate,"
                "bit_rate_blocking_rate,episode_bit_rate_blocking_rate, episode_service_realocations, episode_defrag_cicles"
            )
            for mf in env.env.modulations:
                header += f",modulation_{mf.spectral_efficiency}"
            header += ",episode_disrupted_services,episode_time,"
            header += "mean_gsnr\n"
            file_handler.write(header)

            # Execução dos episódios
            for ep in range(n_eval_episodes):
                obs, info = env.reset()
                done = False
                start_time = time.time()
                step_count = 0
                
                print(f"\n=== EPISÓDIO {ep + 1}/{n_eval_episodes} ===")
                
                while not done:
                    # Debug visual - mostrar slots antes da alocação
                    if debug and ep == 0 and step_count < 5:  # Apenas primeiro episódio e primeiros 5 steps para não poluir
                        print_link_slots(env, "ANTES DA ALOCAÇÃO")
                        print(f"\nCurrent service: {env.env.current_service}")
                    
                    action, path_id, mod_id = fn_heuristic(env)
                    obs, reward, done, truncated, info = env.step(action)
                    
                    # Debug visual - mostrar slots depois da alocação
                    if debug and ep == 0 and step_count < 5:
                        print_link_slots(env, "DEPOIS DA ALOCAÇÃO")
                        print(f"Action taken: {action}")
                        print("-" * 50)
                    
                    step_count += 1
                    
                end_time = time.time()
                ep_time = end_time - start_time

                print(f"Episódio {ep} finalizado.")
                print(info)

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
                file_handler.write(row)

        print(f"\nFinalizado! Resultados salvos em: {monitor_final_name}")
    else:
        # Executar sem monitoramento (modo simples)
        for ep in range(n_eval_episodes):
            obs, info = env.reset()
            done = False
            step_count = 0
            
            while not done:
                # Debug visual - mostrar slots antes da alocação
                if debug and ep == 0 and step_count < 5:  # Apenas primeiro episódio e primeiros 5 steps para não poluir
                    print_link_slots(env, "ANTES DA ALOCAÇÃO")
                    print(f"\nCurrent service: {env.env.current_service}")
                
                action, path_id, mod_id = fn_heuristic(env)
                obs, reward, done, truncated, info = env.step(action)
                
                # Debug visual - mostrar slots depois da alocação
                if debug and ep == 0 and step_count < 5:
                    print_link_slots(env, "DEPOIS DA ALOCAÇÃO")
                    print(f"Action taken: {action}")
                    print("-" * 50)
                
                step_count += 1

def create_environment(topology_name="nobel-eu.xml", episode_length=10, simple_modulations=True, debug=True):
    """Cria o ambiente de teste baseado na configuração do graph_load.py"""
    
    # Determinar o arquivo de topologia
    if topology_name.endswith('.txt'):
        topo_file = f"examples/topologies/{topology_name}"
    else:
        topo_file = f"examples/topologies/{topology_name}.txt"
    
    # Verificar se o arquivo existe
    if not os.path.exists(topo_file):
        raise FileNotFoundError(f"Arquivo de topologia '{topo_file}' não encontrado.")
    
    # Escolher modulações
    if simple_modulations:
        mods = define_modulations_simplified()
        print("Usando modulações simplificadas (QPSK e 16QAM)")
    else:
        mods = define_modulations()
        print("Usando modulações completas (6 modulações)")
    
    # Criar topologia
    topology = get_topology(
        topo_file, None, mods,
        max_span_length=100, 
        default_attenuation=0.2,
        default_noise_figure=4.5, 
        k_paths=2  # Reduzido para compatibilidade com redes menores
    )
    
    # Configuração de banda multibanda baseada no graph_load.py
    band_specs = [
        {"name": "L", "start_thz": 186.00, "num_slots": 406, "noise_figure_db": 6.0, "attenuation_db_km": 0.20},
        {"name": "C", "start_thz": 191.60, "num_slots": 344, "noise_figure_db": 5.5, "attenuation_db_km": 0.191},
        {"name": "S", "start_thz": 197.22, "num_slots": 647, "noise_figure_db": 7.0, "attenuation_db_km": 0.22},
    ]
    
    # Configurações do ambiente usando band_specs (padrão do graph_load.py)
    env_args = dict(
        topology=topology,
        band_specs=band_specs,
        seed=10,
        allow_rejection=True,
        load=300,
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
        # O erro ocorria em decimal_to_array quando tentava acessar modulation IDs
        # que não existiam no array allowed_mods. Com gen_observation=False,
        # o ambiente não tenta gerar observações e evita esse problema.
    )
    
    return env_args, debug

def run_test_simple():
    """Função principal para executar o teste simples usando a nova estrutura"""
    # Configurações fixas para o teste
    topology_name = "nobel-eu.xml"  # Nome correto do arquivo
    episode_length = 100000
    num_episodes = 5
    simple_modulations = True
    debug = True
    heuristic_index = 1  # Heurística multibanda
    
    # Escolher modulações
    if simple_modulations:
        mods = define_modulations_simplified()
        print("Usando modulações simplificadas (QPSK e 16QAM)")
    else:
        mods = define_modulations()
        print("Usando modulações completas (6 modulações)")
    
    # Determinar o arquivo de topologia (caminho correto)
    if topology_name.endswith('.xml'):
        topo_file = f"examples/topologies/{topology_name}"
    else:
        topo_file = f"examples/topologies/{topology_name}.txt"
    
    # Verificar se o arquivo existe
    if not os.path.exists(topo_file):
        raise FileNotFoundError(f"Arquivo de topologia '{topo_file}' não encontrado.")
    
    # Criar topologia
    topology = get_topology(
        topo_file, None, mods,
        max_span_length=100, 
        default_attenuation=0.2,
        default_noise_figure=4.5, 
        k_paths=2
    )
    
    # Configuração de banda multibanda baseada no graph_load.py
    band_specs = [
        {"name": "L", "start_thz": 186.00, "num_slots": 406, "noise_figure_db": 6.0, "attenuation_db_km": 0.20},
        {"name": "C", "start_thz": 191.60, "num_slots": 344, "noise_figure_db": 5.5, "attenuation_db_km": 0.191},
        {"name": "S", "start_thz": 197.22, "num_slots": 647, "noise_figure_db": 7.0, "attenuation_db_km": 0.22},
    ]
    
    # Executar usando a nova função
    run_environment_with_monitoring(
        n_eval_episodes=num_episodes,
        heuristic_index=heuristic_index,
        monitor_file_name=None,  # Sem arquivo de monitoramento para teste simples
        topology=topology,
        seed=10,
        allow_rejection=True,
        load=300,
        episode_length=episode_length,
        launch_power_dbm=0,
        frequency_slot_bandwidth=12.5e9,
        bit_rate_selection="discrete",
        bit_rates=(48, 120),
        margin=0,
        file_name="",
        measure_disruptions=False,
        defragmentation=False,
        n_defrag_services=0,
        gen_observation=False,
        band_specs=band_specs,
        debug=debug
    )
    
    print("Todas as simulações foram executadas.")

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Teste Simples de Rede Óptica Multibanda')
    parser.add_argument(
        '-t', '--topology_file',
        type=str,
        default='nobel-eu.xml',
        help='Arquivo de topologia a ser utilizado (default: ring_4.txt)'
    )
    parser.add_argument(
        '-e', '--num_episodes',
        type=int,
        default=5,
        help='Número de episódios a serem simulados (default: 1)'
    )
    parser.add_argument(
        '-s', '--episode_length',
        type=int,
        default=100000,
        help='Número de chegadas por episódio (default: 10)'
    )
    parser.add_argument(
        '-hi', '--heuristic_index',
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5, 6],
        help='Índice da heurística (1: First Fit, 2: Highest SNR, 3: Lowest Fragmentation, 4: Load Balancing, 5: Prioridade Banda C então L, 6: Multibanda Best Band)'
    )
    parser.add_argument(
        '-mf', '--monitor_file_name',
        type=str,
        default=None,
        help='Nome base para o arquivo CSV de monitoramento (deixe vazio para não salvar)'
    )
    parser.add_argument('-mb', '--bands', nargs='+', type=str, 
                        default=['BandaC'],
                        choices=['BandaC', 'BandaL', 'BandaS', 'BandaC+L'],
                        help='Especifica uma ou mais bandas para simular')
    parser.add_argument(
        '-d', '--debug',
        action='store_true',
        help='Habilita debug visual dos slots'
    )
    parser.add_argument(
        '-sm', '--simple_modulations',
        action='store_true',
        help='Usa apenas 6 modulações (BPSK até 64QAM) ao invés das simplificadas'
    )
    parser.add_argument(
        '-st', '--simple_test',
        action='store_true',
        help='Executa no modo de teste simples: ring_4, 1 episódio, debug habilitado'
    )
    parser.add_argument(
        '-l', '--load',
        type=float,
        default=300.0,
        help='Carga da simulação (default: 300.0)'
    )
    
    return parser.parse_args()

def main():
    """Função principal com opções de linha de comando baseada no graph_load.py"""
    args = parse_arguments()
    
    # Modo de teste simples - sobrescreve configurações
    if args.simple_test:
        args.topology_file = "nobel-eu.xml"  # Nome correto
        args.num_episodes = 5
        args.episode_length = 100000
        args.debug = True
        args.heuristic_index = 1
        args.bands = ['BandaC']
        print("=== MODO TESTE SIMPLES ATIVADO ===")
        print("Configurações: nobel-eu, 5 episódios, 100000 steps, debug ON, heurística First Fit")
        print("="*50)

    # Escolha das modulações baseada no argumento
    if args.simple_modulations:
        cur_modulations = define_modulations()
        print("Usando modulações completas (6 modulações)")
    else:
        cur_modulations = define_modulations_simplified()
        print("Usando modulações simplificadas (QPSK e 16QAM)")

    # Determinar o caminho da topologia
    if args.topology_file.endswith('.xml'):
        topology_path = f"examples/topologies/{args.topology_file}"
    else:
        topology_path = f"examples/topologies/{args.topology_file}"
        
    if not os.path.exists(topology_path):
        raise FileNotFoundError(f"Arquivo de topologia '{topology_path}' não encontrado.")

    # Configurações de banda baseadas no graph_load.py
    if args.topology_file == "nobel-eu.xml":
        band_specs_options = {
            "BandaC": [{"name": "C", "start_thz": 191.60, "num_slots": 344, "noise_figure_db": 5.5, "attenuation_db_km": 0.191}],
            "BandaL": [{"name": "L", "start_thz": 185.83, "num_slots": 406, "noise_figure_db": 6.0, "attenuation_db_km": 0.200}],
            "BandaS": [{"name": "S", "start_thz": 197.22, "num_slots": 647, "noise_figure_db": 7.0, "attenuation_db_km": 0.220}],
            "BandaC+L": [
                {"name": "L", "start_thz": 185.83, "num_slots": 406, "noise_figure_db": 5.5, "attenuation_db_km": 0.200},
                {"name": "C", "start_thz": 191.60, "num_slots": 344, "noise_figure_db": 6.0, "attenuation_db_km": 0.191},
                {"name": "S", "start_thz": 197.22, "num_slots": 647, "noise_figure_db": 7.0, "attenuation_db_km": 0.220},
            ]
        }
        print("Usando configuração especial de band_specs para ring_4")
    else:
        band_specs_options = {
            "BandaC": [{"name": "C", "start_thz": 191.60, "num_slots": 344, "noise_figure_db": 5.5, "attenuation_db_km": 0.191}],
            "BandaL": [{"name": "L", "start_thz": 185.83, "num_slots": 406, "noise_figure_db": 6.0, "attenuation_db_km": 0.200}],
            "BandaS": [{"name": "S", "start_thz": 197.22, "num_slots": 647, "noise_figure_db": 7.0, "attenuation_db_km": 0.220}],
            "BandaC+L": [
                {"name": "C", "start_thz": 191.60, "num_slots": 344, "noise_figure_db": 5.5, "attenuation_db_km": 0.191},
                {"name": "L", "start_thz": 185.83, "num_slots": 406, "noise_figure_db": 6.0, "attenuation_db_km": 0.200},
            ]
        }

    print(f"Bandas selecionadas para simulação: {args.bands}")

    for band_name in args.bands:
        print(f"\n--- Simulando banda: {band_name} ---")
        band_specs = band_specs_options[band_name]
        
        # Criar topologia
        topology = get_topology(
            topology_path,
            None,
            cur_modulations,
            100,  # max_span_length
            0.2,  # default_attenuation
            4.5,  # default_noise_figure
            5 if topology_path.find("ring_4") == -1 else 2  # k_paths
        )
        
        monitor_file = None
        if args.monitor_file_name:
            monitor_file = f"{args.monitor_file_name}_{band_name}"

        # Executar simulação
        run_environment_with_monitoring(
            n_eval_episodes=args.num_episodes,
            heuristic_index=args.heuristic_index,
            monitor_file_name=monitor_file,
            topology=topology,
            seed=seed,
            allow_rejection=True,
            load=args.load,
            episode_length=args.episode_length,
            launch_power_dbm=0.0,
            frequency_slot_bandwidth=12.5e9,
            bit_rate_selection="discrete",
            bit_rates=(48, 120),
            margin=0,
            file_name=f"service_logs/load_services_{band_name}_{args.load}.csv" if args.monitor_file_name else "",
            measure_disruptions=False,
            defragmentation=False,
            n_defrag_services=0,
            gen_observation=False,
            band_specs=band_specs,
            debug=args.debug
        )

    print("\nTodas as simulações foram executadas.")

if __name__ == "__main__":
    if len(os.sys.argv) == 1:
        # Se nenhum argumento foi passado, executar o teste simples padrão
        run_test_simple()
    else:
        # Se argumentos foram passados, usar o parser
        main()