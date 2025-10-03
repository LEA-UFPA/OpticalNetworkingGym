import os
import time
from datetime import datetime
from optical_networking_gym.topology import Modulation, get_topology

# ===================================================
# Função para definir as cargas com base no nome da topologia
# ===================================================
def get_loads(topology_name: str) -> np.ndarray:
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

# ===================================================
# Função que executa o ambiente de simulação
# (adaptada do código do launch power)
# ===================================================
def run_environment(    
    n_eval_episodes,
    heuristic,
    monitor_file_name,
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
    margin,
    file_name,
    measure_disruptions,
    defragmentation,
    n_defrag_services,
    gen_observation,
    band_specs: List[dict] = None,
    debug: bool = False,
) -> None:
    """
    Executa o ambiente com a heurística especificada e salva os resultados em um arquivo CSV.

    :param n_eval_episodes: Número de episódios a serem executados.
    :param heuristic: Índice da heurística a ser utilizada.
    :param monitor_file_name: Nome base para o arquivo CSV de monitoramento.
    :param topology: Objeto de topologia.
    :param seed: Semente para geração de números aleatórios.
    :param allow_rejection: Permitir rejeição de solicitações.
    :param load: Carga a ser utilizada na simulação.
    :param episode_length: Número de chegadas por episódio.
    :param num_spectrum_resources: Número de recursos de espectro.
    :param launch_power_dbm: Potência de lançamento em dBm.
    :param bandwidth: Largura de banda.
    :param frequency_start: Frequência inicial.
    :param frequency_slot_bandwidth: Largura de banda do slot de frequência.
    :param bit_rate_selection: Seleção de taxa de bits.
    :param bit_rates: Taxas de bits disponíveis.
    :param margin: Margem.
    :param file_name: Nome do arquivo para salvar serviços.
    :param measure_disruptions: Medir interrupções.
    :param defragmentation: Habilita defragmentação.
    :param n_defrag_services: Número de serviços defragmentados por ciclo.
    :param gen_observation: Gera observação para agentes RL.
    :param band_specs: Especificações das bandas para simulação multibanda.
    :param debug: Habilita debug visual dos slots (apenas primeiro episódio).
    """
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
            k_paths=5, 
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

    # Seleção da heurística baseada no índice
    if heuristic == 1:
        fn_heuristic = heuristic_shortest_available_path_first_fit_best_modulation
    elif heuristic == 2:
        fn_heuristic = heuristic_highest_snr
    elif heuristic == 3:
        fn_heuristic = heuristic_lowest_fragmentation
    elif heuristic == 4:
        fn_heuristic = load_balancing_best_modulation
    elif heuristic == 5:  # Escolha um índice para sua heurística de prioridade
        fn_heuristic = heuristic_priority_band_C_then_L
    elif heuristic == 6:  # Nova heurística multibanda
        fn_heuristic = shortest_available_path_first_fit_best_modulation_best_band
    else:
        raise ValueError(f"Heuristic index `{heuristic}` is not found!")

    # Criação do ambiente
    env = QRMSAEnvWrapper(**env_args)
    monitor_file_name = f"load_results_{topology.name}_{load}.csv"
    os.makedirs("results", exist_ok=True)
    
    with open(f"results/{monitor_file_name}", "wt") as f:
        f.write(f"# Date: {datetime.now()}\n")
        f.write("episode,service_blocking_rate,bit_rate_blocking_rate\n")

        for ep in range(5):
            obs, info = env.reset()
            done = False
            start_time = time.time()
            step_count = 0
            
            while not done:
                # Debug visual - mostrar slots antes da alocação
                if debug and ep == 0 and step_count < 5:  # Apenas primeiro episódio e primeiros 5 steps para não poluir
                    print_link_slots(env, "ANTES DA ALOCAÇÃO")
                    print(f"\nCurrent service: {env.env.current_service}")
                
                action,_,_ = fn_heuristic(env)
                _, _, done, _, info = env.step(action)
                
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
            for service in env.env.topology.graph["services"]:
                mean_gsnr += service.OSNR
            mean_gsnr /= len(env.env.topology.graph["services"])
            row += f",{mean_gsnr}\n"
            file_handler.write(row)
                

    print(f"\nFinalizado! Resultados salvos em: {monitor_final_name}")

def starmap_helper(args):
    """Helper function for multiprocessing starmap compatibility"""
    return run_environment(*args)

def print_link_slots(env, stage):
    """Função para debug visual dos slots nos links"""
    print(f"\n=== LINKS {stage} ===")
    for u, v in env.env.topology.edges():
        idx = env.env.topology[u][v]["index"]
        slots = env.env.topology.graph["available_slots"][idx]
        print(f"Link {u}->{v} slots: {''.join(map(str, slots.tolist()))}")

def define_modulations_simplified():
    """Modulações simplificadas do test_simple"""
    return (
        Modulation("QPSK", 200_000, 2, minimum_osnr=6.72, inband_xt=-17),
        Modulation("16QAM", 500, 4, minimum_osnr=13.24, inband_xt=-23),
    )

# ===================================================
# Configuração de logging, semente e argumentos de entrada
# ===================================================
logging.getLogger("rmsaenv").setLevel(logging.INFO)
np.set_printoptions(linewidth=np.inf)

seed = 50
random.seed(seed)
np.random.seed(seed)

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Simulação de Rede Óptica - Variação de Carga')
    parser.add_argument(
        '-t', '--topology_file',
        type=str,
        default='nobel-eu.xml',#'nobel-eu.xml',
        help='Arquivo de topologia a ser utilizado (default: nsfnet_chen.txt)'
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
        help='Número de chegadas por episódio (default: 1000)'
    )
    parser.add_argument(
        '-th', '--threads',
        type=int,
        default=25,
        help='Número de threads para execução das simulações (default: 2)'
    )
    # Argumento para a heurística a ser utilizada
    parser.add_argument(
        '-hi', '--heuristic_index',
        type=int,
        default=1,
        choices=[1, 2, 3, 4, 5, 6],
        help='Índice da heurística (1: First Fit, 2: Lowest Spectrum, 3: Load Balancing Modulation, 4: Load Balancing Best Modulation, 5: Prioridade Banda C então L, 6: Multibanda Best Band)'
    )
    # Nome base para o arquivo CSV de monitoramento
    parser.add_argument(
        '-mf', '--monitor_file_name',
        type=str,
        default='examples/jocn_benchmark_2024/results/load_episodes',
        help='Nome base para o arquivo CSV de monitoramento'
    )
    parser.add_argument('-mb', '--bands', nargs='+', type=str, 
                        default=['BandaC'],
                        choices=['BandaC', 'BandaL', 'BandaS', 'BandaC+L'],
                        help='Especifica uma ou mais bandas para simular (ex: --bands BandaC BandaL BandaC+L).')
    
    # Argumento para habilitar debug visual
    parser.add_argument(
        '-d', '--debug',
        action='store_true',
        help='Habilita debug visual dos slots (apenas para o primeiro episódio)'
    )
    
    # Argumento para usar modulações simplificadas
    parser.add_argument(
        '-sm', '--simple_modulations',
        action='store_true',
        help='Usa apenas 2 modulações (QPSK e 16QAM) como no test_simple'
    )
    
    # Argumento para modo de teste simples (como test_simple.py)
    parser.add_argument(
        '-st', '--simple_test',
        action='store_true',
        help='Executa no modo de teste simples: ring_4, 5 episódios curtos, debug habilitado'
    )
    
    return parser.parse_args()

# ===================================================
# Função principal
# ===================================================
def main():
    args = parse_arguments()
    
    # Modo de teste simples - sobrescreve configurações para replicar test_simple.py
    if args.simple_test:
        args.topology_file = "ring_4.txt"
        args.num_episodes = 1  # Apenas 1 episódio para debug
        args.episode_length = 5  # Episódios curtos
        args.debug = True
        args.simple_modulations = True
        args.heuristic_index = 6  # Usa a heurística multibanda
        args.bands = ['BandaC+L']  # Multibanda
        print("=== MODO TESTE SIMPLES ATIVADO ===")
        print("Configurações: ring_4, 1 episódio, 5 steps, debug ON, modulações simples, heurística multibanda")
        print("="*50)

    # Multi-band functionality: Add band specifications with physical parameters
    band_specifications = {
        "BandaC": {"frequency_start": 191.60e12, "num_spectrum_resources": 344,
                   "attenuation": 0.191, "noise_figure": 5.5},
        "BandaL": {"frequency_start": 185.83e12, "num_spectrum_resources": 320,
                   "attenuation": 0.200, "noise_figure": 6},
        "BandaS": {"frequency_start": 197.22e12, "num_spectrum_resources": 320, 
                   "attenuation": 0.220, "noise_figure": 7},
        "BandaC+L": [
            {"name": "C", "start_thz": 191.60, "num_slots": 320, "noise_figure_db": 5.5, "attenuation_db_km": 0.191},
            {"name": "L", "start_thz": 185.83, "num_slots": 320, "noise_figure_db": 6.0, "attenuation_db_km": 0.200},
        ]
    }

    # Escolha das modulações baseada no argumento
    if args.simple_modulations:
        cur_modulations = define_modulations_simplified()
        print("Usando modulações simplificadas (QPSK e 16QAM)")
    else:
        # Definição das modulações (valores atualizados conforme exemplo do launch power)
        cur_modulations: Tuple[Modulation, ...] = (
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
        print("Usando modulações completas (6 modulações)")

    attenuation_db_km = 0.2
    default_noise_figure_db = 4.5

    
    # Determinar o caminho da topologia baseado na extensão
    if args.topology_file.endswith('.xml'):
        topology_path = f"/home/lucas/Documentos/ONG/OpticalNetworkingGym/examples/topologies/{args.topology_file}"
    else:
        topology_path = f"/home/lucas/Documentos/ONG/OpticalNetworkingGym/examples/topologies/{args.topology_file}"
        
    if not os.path.exists(topology_path):
        raise FileNotFoundError(f"Arquivo de topologia '{topology_path}' não encontrado.")
        
    # Configuração especial para ring_4 (inspirada no test_simple)
    if args.topology_file == "ring_4.txt":
        special_band_specs = [
            {"name": "L", "start_thz": 186.00, "num_slots": 10, "noise_figure_db": 4.5, "attenuation_db_km": 0.22},
            {"name": "C", "start_thz": 191.60, "num_slots": 10, "noise_figure_db": 5.0, "attenuation_db_km": 0.20},
            {"name": "S", "start_thz": 197.22, "num_slots": 10, "noise_figure_db": 6.0, "attenuation_db_km": 0.24},
        ]
        print("Usando configuração especial de band_specs para ring_4")
    else:
        special_band_specs = [
            {"name": "C", "start_thz": 191.60, "num_slots": 344, "noise_figure_db": 5.5, "attenuation_db_km": 0.191},
            {"name": "S", "start_thz": 197.22, "num_slots": 320, "noise_figure_db": 7.0, "attenuation_db_km": 0.220},
            {"name": "L", "start_thz": 185.83, "num_slots": 320, "noise_figure_db": 6.0, "attenuation_db_km": 0.200},
        ]
        
    band_specs = special_band_specs
    # Parâmetros de simulação
    threads = args.threads
    bandwidth = 4e12
    frequency_start = 3e8 / 1565e-9
    frequency_slot_bandwidth = 12.5e9
    bit_rates = (48,120)
    margin = 0

    launch_power = 0.0

    loads = get_loads(args.topology_file)

    strategies = list(range(1, 5))

    
    print(f"Bandas selecionadas para simulação: {args.bands}")
    
    env_args = []
    for band_name in args.bands:
        if band_name == "BandaC+L":
            band_specs = [
                {"name": "C", "start_thz": 191.60, "num_slots": 344, "noise_figure_db": 5.5, "attenuation_db_km": 0.191},
                {"name": "L", "start_thz": 185.83, "num_slots": 406, "noise_figure_db": 6.0, "attenuation_db_km": 0.200},
            ]
            topology = get_topology(
                topology_path,
                None,
                cur_modulations,
                100,
                0.2,  # Atenuação default
                4.5,  # NF default
                5
            )
            for current_load in get_loads(args.topology_file):
                run_environment(
                    args.num_episodes,
                    args.heuristic_index,
                    f"{args.monitor_file_name}_{band_name}",
                    topology,
                    seed,
                    True,
                    current_load,
                    args.episode_length,
                    sum([spec["num_slots"] for spec in band_specs]),  # total slots
                    launch_power,
                    sum([spec["num_slots"] for spec in band_specs]) * frequency_slot_bandwidth,  # total bandwidth
                    min([spec["start_thz"] for spec in band_specs]) * 1e12,  # menor freq de início
                    frequency_slot_bandwidth,
                    "discrete",
                    bit_rates,
                    margin,
                    f"service_logs/load_services_{band_name}_{current_load}.csv",
                    False,
                    False,
                    0,
                    False,
                    band_specs,
                    args.debug
                )
        else:
            # Simulação para banda única 
            specs = band_specifications[band_name]
            topology = get_topology(
                topology_path,
                None,
                cur_modulations,
                100,
                specs["attenuation"],
                specs["noise_figure"],
                5
            )
            for current_load in get_loads(args.topology_file):
                run_environment(
                    args.num_episodes,
                    args.heuristic_index,
                    f"{args.monitor_file_name}_{band_name}",
                    topology,
                    seed,
                    True,
                    current_load,
                    args.episode_length,
                    specs["num_spectrum_resources"],
                    launch_power,
                    specs["num_spectrum_resources"] * frequency_slot_bandwidth,
                    specs["frequency_start"],
                    frequency_slot_bandwidth,
                    "discrete",
                    bit_rates,
                    margin,
                    f"service_logs/load_services_{band_name}_{current_load}.csv",
                    False,
                    False,
                    0,
                    False,
                    [specs],
                    args.debug
                )

    # Execução das simulações utilizando multiprocessing se houver mais de uma thread
    if not env_args:
        print("Nenhuma simulação para executar. Verifique os argumentos.")
        return
        
    print(f"Iniciando {len(env_args)} simulações em {threads} threads...")
    
    if threads > 1:
        with Pool(processes=threads) as pool:
            list(tqdm(pool.imap_unordered(starmap_helper, env_args), 
                     total=len(env_args), desc="Simulando Cargas/Bandas"))
    else:
        for arg in tqdm(env_args, desc="Simulando Cargas/Bandas"):
            run_environment(*arg)

    print("Todas as simulações foram executadas.")

if __name__ == "__main__":
    # Exemplos de uso:
    # 
    # Modo de teste simples (equivalente ao test_simple.py):
    # python graph_load.py --simple_test
    #
    # Modo normal com debug visual:
    # python graph_load.py -t ring_4.txt -d -sm -hi 6
    #
    # Simulação completa com multibanda:
    # python graph_load.py -t nobel-eu.xml -mb BandaC+L -hi 6 -e 10
    #
    main()