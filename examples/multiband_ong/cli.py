"""
CLI Interface for OpticalNetworkingGym
Allows running simulations with configurable parameters via command line arguments.
"""
import argparse
import logging
import os
import random
import sys
import time
from datetime import datetime
from typing import List, Tuple, Dict, Any

import numpy as np
from tqdm import tqdm

# Add the parent directory to sys.path to ensure imports work if run from examples/multiband_ong
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
# Also add examples/plots to path to import plot script
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'plots'))

from optical_networking_gym.topology import Modulation, get_topology
from optical_networking_gym.wrappers.qrmsa_gym import QRMSAEnvWrapper
from optical_networking_gym.heuristics.heuristics import (
    shortest_available_path_first_fit_best_modulation,
    shortest_available_path_lowest_spectrum_best_modulation,
    best_modulation_load_balancing,
    load_balancing_best_modulation,
    heuristic_shortest_available_path_first_fit_best_modulation,
    heuristic_highest_snr,
    heuristic_lowest_fragmentation,
    heuristic_priority_band_C_then_L,
    shortest_available_path_first_fit_best_modulation_best_band,
)

try:
    from plot import generate_plot
except ImportError as e:
    print(f"Warning: Could not import plotting module. Plotting will be disabled.")
    print(f"  Error: {e}")
    print(f"  To enable plotting, please install pandas and matplotlib: pip install pandas matplotlib")
    generate_plot = None

# ===================================================
# Helper Functions (Adapted from test_simple.py)
# ===================================================

def get_loads(topology_name: str) -> np.ndarray:
    """Returns appropriate loads for each topology"""
    if topology_name == "nobel-eu.xml":
        return np.arange(400, 1401, 100)  
    elif topology_name == "germany50.xml":
        return np.arange(100, 1501, 100)
    elif topology_name == "janos-us.xml":
        return np.arange(100, 601, 50)
    elif topology_name == "nsfnet_chen.txt":
        return np.arange(100, 601, 50)
    elif topology_name == "ring_4.txt":
        return np.arange(100, 601, 50)
    else:
        # Default fallback if unknown
        return np.arange(100, 1001, 100)

def get_heuristic_function(heuristic_index: int):
    """Returns heuristic function based on index"""
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

def define_modulations() -> Tuple[Modulation, ...]:
    return (
        Modulation(name="QPSK", maximum_length=10_000, spectral_efficiency=2, minimum_osnr=6.72, inband_xt=-17),
        Modulation(name="8QAM", maximum_length=1_000, spectral_efficiency=3, minimum_osnr=10.84, inband_xt=-20),
        Modulation(name="16QAM", maximum_length=500, spectral_efficiency=4, minimum_osnr=13.24, inband_xt=-23),
        Modulation(name="32QAM", maximum_length=250, spectral_efficiency=5, minimum_osnr=16.16, inband_xt=-26),
        Modulation(name="64QAM", maximum_length=125, spectral_efficiency=6, minimum_osnr=19.01, inband_xt=-29),
    )

def run_environment_with_monitoring(
    n_eval_episodes: int,
    heuristic_index: int,
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
    measure_disruptions: bool,
    defragmentation: bool,
    n_defrag_services: int,
    band_specs: List[dict] = None,
    debug: bool = False,
    max_span_length: float = 100.
) -> None:
    """
    Executes the environment with specified parameters and saves results to CSV.
    """
    
    # Environment Configuration
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
        file_name="", # Not used internally for file saving in this wrapper version usually
        measure_disruptions=measure_disruptions,
        k_paths=2,
        modulations_to_consider=2,  
        defragmentation=defragmentation,
        n_defrag_services=n_defrag_services,
        gen_observation=False,
    )

    # Select Heuristic
    fn_heuristic = get_heuristic_function(heuristic_index)

    # Create Environment
    env = QRMSAEnvWrapper(**env_args)
    
    # Create Results Directory
    # Updated filename format to include heuristic and band for uniqueness/comparison
    heuristic_name = fn_heuristic.__name__
    # Simplify heuristic name for filename
    if heuristic_name.startswith("heuristic_"):
        heuristic_name = heuristic_name.replace("heuristic_", "")
    
    # Get band name from specs if possible, or pass it as arg. 
    # Since band_specs is a list of dicts, we can construct a name or use the one passed to CLI.
    # But here we only have band_specs. Let's try to infer or just use a generic name if not passed.
    # Actually, it's better to pass the band name string to this function.
    # For now, let's just use a hash or join names if not passed.
    # BUT, I can update the function signature to accept `band_name`.
    
    # Let's assume band_name is passed or we construct it.
    # To avoid changing signature too much, let's try to find it in band_specs
    band_str = "+".join([b['name'] for b in band_specs]) if band_specs else "Default"
    
    monitor_final_name = f"load_results_{topology.name}_{heuristic_name}_Banda{band_str}_{load}.csv"
    os.makedirs("results", exist_ok=True)
    
    print(f"Starting simulation for Load: {load}, Heuristic: {heuristic_index}, Episodes: {n_eval_episodes}")
    
    with open(f"results/{monitor_final_name}", "wt") as f:
        # Write Header Metadata
        f.write(f"# Date: {datetime.now()}\n")
        f.write(f"# Topology: {topology.name}\n")
        f.write(f"# Heuristic: {fn_heuristic.__name__}\n")
        f.write(f"# Episodes: {n_eval_episodes}\n")
        f.write(f"# Episode Length: {episode_length}\n")
        f.write(f"# Bit Rates: {bit_rates} Gbps\n")
        f.write(f"# Modulations: {[m.name for m in env.env.modulations]}\n")
        f.write(f"# Span Length: {max_span_length} km\n")
        f.write(f"\n# Band Specifications\n")
        
        if band_specs:
            band_names = [spec['name'] for spec in band_specs]
            f.write(f"# | | {' | '.join(band_names)} |\n")
            f.write(f"# |---|{'---|' * len(band_specs)}\n")
            
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

        # CSV Header
        header = (
            "episode,service_blocking_rate,episode_service_blocking_rate,"
            "bit_rate_blocking_rate,episode_bit_rate_blocking_rate,"
            "episode_service_realocations,episode_defrag_cicles"
        )
        for mf in env.env.modulations:
            header += f",modulation_{mf.spectral_efficiency}"
        header += ",episode_disrupted_services,episode_time,mean_gsnr\n"
        f.write(header)

        # Run Episodes
        for ep in range(n_eval_episodes):
            obs, info = env.reset()
            done = False
            start_time = time.time()
            
            # Progress bar for each episode
            pbar = tqdm(total=episode_length, desc=f"Ep {ep+1}/{n_eval_episodes}", unit="req", leave=False)
            
            while not done:
                action, path_id, mod_id = fn_heuristic(env)
                obs, reward, done, truncated, info = env.step(action)
                pbar.update(1)
                
            pbar.close()
            
            end_time = time.time()
            ep_time = end_time - start_time

            # Write Row
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
            f.flush()

    print(f"Finished! Results saved to: results/{monitor_final_name}")

# ===================================================
# Main CLI Logic
# ===================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='CLI for OpticalNetworkingGym Simulator',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Simulation Control
    sim_group = parser.add_argument_group('Simulation Control')
    sim_group.add_argument('-e', '--episodes', type=int, default=5, help='Number of episodes')
    sim_group.add_argument('-l', '--episode-length', type=int, default=100000, help='Arrivals per episode')
    sim_group.add_argument('-s', '--seed', type=int, default=50, help='Random seed')
    sim_group.add_argument('-d', '--debug', action='store_true', help='Enable debug mode')

    # Topology
    topo_group = parser.add_argument_group('Topology')
    topo_group.add_argument('-t', '--topology', type=str, default='nobel-eu.xml', help='Topology file name')
    topo_group.add_argument('--max-span-length', type=float, default=100.0, help='Max span length (km)')
    topo_group.add_argument('--slot-bandwidth', type=float, default=12.5e9, help='Frequency slot bandwidth (Hz)')

    # Traffic & Load
    traffic_group = parser.add_argument_group('Traffic & Load')
    traffic_group.add_argument('--load', type=str, default=None, 
                               help='Load value(s). Can be a single number or start:stop:step (e.g. 100:1000:100). If not set, uses topology defaults.')
    traffic_group.add_argument('--bit-rates', nargs='+', type=int, default=[48, 120], help='Allowed bit rates (Gbps)')

    # Physical Layer
    phy_group = parser.add_argument_group('Physical Layer')
    phy_group.add_argument('-p', '--power', type=float, default=0.0, help='Launch power (dBm)')
    phy_group.add_argument('-b', '--bands', type=str, default='BandaC+L+S', 
                           choices=['BandaC', 'BandaL', 'BandaS', 'BandaC+L', 'BandaC+L+S'],
                           help='Bands configuration to use')

    # Heuristics
    heur_group = parser.add_argument_group('Heuristics')
    heur_group.add_argument('--heuristic', type=str, default='best_band',
                            choices=['first_fit', 'highest_snr', 'lowest_fragmentation', 'load_balancing', 'priority_c_l', 'best_band'],
                            help='Heuristic algorithm')

    # Features
    feat_group = parser.add_argument_group('Features')
    feat_group.add_argument('--defrag', action='store_true', help='Enable defragmentation')
    feat_group.add_argument('--defrag-services', type=int, default=0, help='Number of services to defragment')
    feat_group.add_argument('--measure-disruptions', action='store_true', help='Measure disruptions')
    feat_group.add_argument('--no-rejection', action='store_false', dest='allow_rejection', help='Disable service rejection (default: allowed)')
    feat_group.add_argument('-y', '--yes', action='store_true', help='Skip confirmation prompt')
    parser.add_argument('-i', '--interactive', action='store_true', help='Run in interactive menu mode')
    parser.set_defaults(allow_rejection=True)

    return parser.parse_args()

def get_user_choice(options: List[str], prompt: str) -> str:
    print(f"\n{prompt}:")
    for idx, opt in enumerate(options, 1):
        print(f"  {idx}. {opt}")
    
    while True:
        try:
            choice = input(f"Select an option (1-{len(options)}): ").strip()
            idx = int(choice) - 1
            if 0 <= idx < len(options):
                return options[idx]
            print("Invalid selection. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def get_user_input(prompt: str, default: str = None, value_type: type = str) -> Any:
    while True:
        user_input = input(f"{prompt} [{default}]: ").strip()
        if not user_input:
            if default is not None:
                return default
            print("Value is required.")
            continue
        
        try:
            if value_type == bool:
                return user_input.lower() in ('y', 'yes', 'true', '1')
            return value_type(user_input)
        except ValueError:
            print(f"Invalid input. Expected type {value_type.__name__}.")

def interactive_mode(args: argparse.Namespace) -> argparse.Namespace:
    print("\n" + "="*40)
    print("   OpticalNetworkingGym Interactive Mode   ")
    print("="*40)

    # Comparison / Plot Only Option
    print("\nAction:")
    print("  1. Run Simulation")
    print("  2. Compare/Plot Existing Results")
    action = get_user_choice(["Run Simulation", "Compare/Plot Existing Results"], "Select Action")
    
    if action == "Compare/Plot Existing Results":
        args.skip_sim = True
    else:
        args.skip_sim = False

    # Topology
    script_dir = os.path.dirname(os.path.abspath(__file__))
    topologies_dir = os.path.join(script_dir, "..", "topologies")
    topologies = [f for f in os.listdir(topologies_dir) if f.endswith(('.xml', '.txt'))]
    if topologies:
        args.topology = get_user_choice(sorted(topologies), "Select Topology")
    else:
        print("No topology files found. Using default.")

    if args.skip_sim:
        return args

    # Heuristic
    heuristics = ['first_fit', 'highest_snr', 'lowest_fragmentation', 'load_balancing', 'priority_c_l', 'best_band']
    args.heuristic = get_user_choice(heuristics, "Select Heuristic")

    # Bands
    bands = ['BandaC', 'BandaL', 'BandaS', 'BandaC+L', 'BandaC+L+S']
    args.bands = get_user_choice(bands, "Select Bands")

    # Load
    print("\nLoad Configuration:")
    print("  Format: single value (e.g., 100) or range start:stop:step (e.g., 100:1000:100)")
    args.load = get_user_input("Enter Load", default="100:1000:100")

    # Episodes
    args.episodes = get_user_input("Number of Episodes", default=5, value_type=int)

    # Advanced Options
    if get_user_input("\nConfigure advanced options? (y/N)", default="n", value_type=bool):
        args.episode_length = get_user_input("Episode Length", default=100000, value_type=int)
        args.seed = get_user_input("Random Seed", default=50, value_type=int)
        args.power = get_user_input("Launch Power (dBm)", default=0.0, value_type=float)
        args.defrag = get_user_input("Enable Defragmentation? (y/N)", default="n", value_type=bool)
        if args.defrag:
            args.defrag_services = get_user_input("Defrag Services Count", default=10, value_type=int)
        args.debug = get_user_input("Enable Debug Mode? (y/N)", default="n", value_type=bool)

    return args

def parse_load_argument(load_arg: str, topology_name: str) -> np.ndarray:
    if load_arg is None:
        return get_loads(topology_name)
    
    if ':' in load_arg:
        try:
            parts = [int(x) for x in load_arg.split(':')]
            if len(parts) == 2:
                return np.arange(parts[0], parts[1])
            elif len(parts) == 3:
                return np.arange(parts[0], parts[1], parts[2])
            else:
                raise ValueError
        except ValueError:
            print(f"Error: Invalid load range format '{load_arg}'. Use start:stop:step")
            sys.exit(1)
    else:
        try:
            return np.array([float(load_arg)])
        except ValueError:
            print(f"Error: Invalid load value '{load_arg}'")
            sys.exit(1)

def get_heuristic_index(name: str) -> int:
    mapping = {
        'first_fit': 1,
        'highest_snr': 2,
        'lowest_fragmentation': 3,
        'load_balancing': 4,
        'priority_c_l': 5,
        'best_band': 6
    }
    return mapping.get(name, 6)

def main():
    while True:
        args = parse_arguments()
        
        # Reset skip_sim default (it might be set by interactive mode)
        args.skip_sim = False

        if args.interactive:
            args = interactive_mode(args)
        
        # Setup Logging
        logging.getLogger("rmsaenv").setLevel(logging.INFO if args.debug else logging.WARNING)
        np.set_printoptions(linewidth=np.inf)
        
        # Seeds
        random.seed(args.seed)
        np.random.seed(args.seed)
        
        # Topology Path
        script_dir = os.path.dirname(os.path.abspath(__file__))
        topologies_dir = os.path.join(script_dir, "..", "topologies")
        topology_path = os.path.join(topologies_dir, args.topology)
        
        if not os.path.exists(topology_path):
             if os.path.exists(args.topology):
                topology_path = args.topology
             else:
                print(f"Error: Topology file '{topology_path}' not found.")
                sys.exit(1)

        # Run Simulation Block
        if not args.skip_sim:
            # Band Specifications
            band_specs_options = {
                "BandaC": [{"name": "C", "start_thz": 191.60, "num_slots": 344, "noise_figure_db": 5.5, "attenuation_db_km": 0.191}],
                "BandaL": [{"name": "L", "start_thz": 185.83, "num_slots": 406, "noise_figure_db": 6.0, "attenuation_db_km": 0.200}],
                "BandaS": [{"name": "S", "start_thz": 197.22, "num_slots": 647, "noise_figure_db": 7.0, "attenuation_db_km": 0.220}],
                "BandaC+L": [
                    {"name": "L", "start_thz": 185.83, "num_slots": 406, "noise_figure_db": 5.5, "attenuation_db_km": 0.200},
                    {"name": "C", "start_thz": 191.60, "num_slots": 344, "noise_figure_db": 6.0, "attenuation_db_km": 0.191},
                ],
                "BandaC+L+S": [
                    {"name": "C", "start_thz": 191.60, "num_slots": 344, "noise_figure_db": 5.5, "attenuation_db_km": 0.191},
                    {"name": "L", "start_thz": 185.83, "num_slots": 344, "noise_figure_db": 6.0, "attenuation_db_km": 0.200},
                    {"name": "S", "start_thz": 197.22, "num_slots": 344, "noise_figure_db": 7.0, "attenuation_db_km": 0.220},
                ]
            }
            
            if args.topology == "nobel-eu.xml":
                band_specs_options["BandaC+L+S"] = [
                    {"name": "C", "start_thz": 191.60, "num_slots": 344, "noise_figure_db": 5.5, "attenuation_db_km": 0.191},
                    {"name": "L", "start_thz": 185.83, "num_slots": 344, "noise_figure_db": 6.0, "attenuation_db_km": 0.200},
                    {"name": "S", "start_thz": 197.22, "num_slots": 344, "noise_figure_db": 7.0, "attenuation_db_km": 0.220},
                ]

            band_specs = band_specs_options[args.bands]

            # Modulations
            cur_modulations = define_modulations()

            # Create Topology
            topology = get_topology(
                topology_path,
                None,
                cur_modulations,
                args.max_span_length,
                0.2,  # default_attenuation
                4.5,  # default_noise_figure
                2     # k_paths
            )

            # Loads
            loads = parse_load_argument(args.load, args.topology)
            
            # Heuristic Index
            heuristic_idx = get_heuristic_index(args.heuristic)

            # Summary and Confirmation
            print("\n" + "="*40)
            print("       SIMULATION CONFIGURATION       ")
            print("="*40)
            print(f"{'Topology':<20}: {args.topology}")
            print(f"{'Bands':<20}: {args.bands}")
            print(f"{'Heuristic':<20}: {args.heuristic} (Index: {heuristic_idx})")
            print(f"{'Episodes':<20}: {args.episodes}")
            print(f"{'Episode Length':<20}: {args.episode_length}")
            print(f"{'Seed':<20}: {args.seed}")
            print(f"{'Power':<20}: {args.power} dBm")
            print(f"{'Defragmentation':<20}: {args.defrag}")
            print(f"{'Loads':<20}: {loads}")
            print("="*40 + "\n")

            if not args.yes:
                try:
                    response = input("Proceed with simulation? [Y/n]: ").strip().lower()
                    if response not in ('', 'y', 'yes'):
                        print("Simulation aborted by user.")
                        # Instead of exit, just break loop or continue to plot?
                        # Usually abort means stop everything. But in loop mode maybe just go to next?
                        # Let's keep it as exit for safety, or just continue to prompt.
                        # Let's continue to prompt to allow user to change mind.
                        pass 
                except KeyboardInterrupt:
                    print("\nSimulation aborted.")
                    sys.exit(0)

            # Only run if confirmed (or yes flag)
            if args.yes or response in ('', 'y', 'yes'):
                print(f"Running simulation for topology: {args.topology}")
                print(f"Bands: {args.bands}")
                print(f"Loads to simulate: {loads}")
                
                for current_load in loads:
                    run_environment_with_monitoring(
                        n_eval_episodes=args.episodes,
                        heuristic_index=heuristic_idx,
                        topology=topology,
                        seed=args.seed,
                        allow_rejection=args.allow_rejection,
                        load=current_load,
                        episode_length=args.episode_length,
                        launch_power_dbm=args.power,
                        frequency_slot_bandwidth=args.slot_bandwidth,
                        bit_rate_selection="discrete",
                        bit_rates=tuple(args.bit_rates),
                        margin=0,
                        measure_disruptions=args.measure_disruptions,
                        defragmentation=args.defrag,
                        n_defrag_services=args.defrag_services,
                        band_specs=band_specs,
                        debug=args.debug,
                        max_span_length=args.max_span_length
                    )
                print("\nAll simulations completed.")

        # Plotting Prompt
        if generate_plot and (args.interactive or not args.yes):
            print("\n" + "="*40)
            print("           RESULTS PLOTTING           ")
            print("="*40)
            
            should_plot = False
            if args.interactive:
                 should_plot = get_user_input("Generate plot from results? (y/N)", default="y", value_type=bool)
            else:
                 try:
                    resp = input("Generate plot from results? [Y/n]: ").strip().lower()
                    should_plot = resp in ('', 'y', 'yes')
                 except KeyboardInterrupt:
                    should_plot = False

            if should_plot:
                metrics_map = {
                    "1": "episode_service_blocking_rate",
                    "2": "episode_bit_rate_blocking_rate",
                    "3": "mean_gsnr",
                    "4": "episode_service_realocations"
                }
                
                selected_metrics = []
                
                if args.interactive:
                    print("\nSelect Metric to Plot:")
                    print("  1. Service Blocking Rate")
                    print("  2. Bit Rate Blocking Rate")
                    print("  3. Mean GSNR")
                    print("  4. Service Reallocations")
                    print("  5. All Metrics")
                    
                    choice = input("Select an option (1-5): ").strip()
                    if choice == "5":
                        selected_metrics = list(metrics_map.values())
                    elif choice in metrics_map:
                        selected_metrics = [metrics_map[choice]]
                    else:
                        print("Invalid selection. Defaulting to Service Blocking Rate.")
                        selected_metrics = ["episode_service_blocking_rate"]
                else:
                    # Non-interactive default
                    selected_metrics = ["episode_service_blocking_rate"]

                for metric in selected_metrics:
                    generate_plot(
                        results_dir="results",
                        topology_name=args.topology.replace('.xml', '').replace('.txt', ''),
                        y_metric=metric,
                        output_dir="figures",
                        episode_length=args.episode_length
                    )

        # Loop Prompt
        if args.interactive:
            if not get_user_input("\nRun another simulation/action? (y/N)", default="n", value_type=bool):
                print("Exiting...")
                break
        else:
            # In CLI mode, we usually run once and exit.
            break

if __name__ == "__main__":
    main()
