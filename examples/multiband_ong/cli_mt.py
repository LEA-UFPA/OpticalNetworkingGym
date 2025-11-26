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
import shutil
from datetime import datetime
from typing import List, Tuple, Dict, Any
from multiprocessing import Pool, cpu_count
import multiprocessing

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
    from plotAN import gerar_grafico_probabilidade_bloqueio as generate_plot
except ImportError as e:
    print(f"Warning: Could not import plotting module. Plotting will be disabled.")
    print(f"  Error: {e}")
    print(f"  To enable plotting, please install pandas and matplotlib: pip install pandas matplotlib")
    generate_plot = None

# ===================================================
# Constants
# ===================================================
RESULTS_DIR = "results"
FREQUENCY_SLOT_BANDWIDTH = 12.5e9

# ===================================================
# Helper Functions (Adapted from test_simple.py)
# ===================================================

def clear_screen():
    """Clears the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

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

def write_csv_header(f, topology, heuristic_index, n_eval_episodes, episode_length, bit_rates, band_specs, frequency_slot_bandwidth, max_span_length, modulations):
    f.write(f"# Date: {datetime.now()}\n")
    f.write(f"# Topology: {topology.name}\n")
    f.write(f"# Heuristic: {get_heuristic_function(heuristic_index).__name__}\n")
    f.write(f"# Episodes: {n_eval_episodes}\n")
    f.write(f"# Episode Length: {episode_length}\n")
    f.write(f"# Bit Rates: {bit_rates} Gbps\n")
    f.write(f"# Modulations: {[m.name for m in modulations]}\n")
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

def write_csv_data_header(f, modulations):
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
    mean_gsnr = 0.0
    if len(env.env.topology.graph["services"]) > 0:
        gsnr_sum = 0
        for service in env.env.topology.graph["services"]:
            # Use OSNR attribute as gsnr is not available
            gsnr_sum += np.mean(service.OSNR)
        mean_gsnr = gsnr_sum / len(env.env.topology.graph["services"])
    return mean_gsnr

def write_episode_results(f, ep, info, modulations, ep_time, mean_gsnr):
    row = (
        f"{ep},{info['service_blocking_rate']},"
        f"{info['episode_service_blocking_rate']},"
        f"{info['bit_rate_blocking_rate']},"
        f"{info['episode_bit_rate_blocking_rate']},"
        f"{info['episode_service_realocations']},"
        f"{info['episode_defrag_cicles']}"
    )
    for mf in modulations:
        row += f",{info.get(f'modulation_{mf.spectral_efficiency}', 0.0)}"
    row += f",{info.get('episode_disrupted_services', 0)},{ep_time:.2f},{mean_gsnr}\n"
    f.write(row)

    f.write(row)



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
    max_span_length: float = 100.,
    show_progress: bool = True,
    tqdm_position: int = 0
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
    
    band_name = "+".join([b['name'] for b in band_specs]) if band_specs else "Default"
    
    monitor_final_name = f"load_results_{topology.name}_{heuristic_name}_Banda{band_name}_{load}.csv"
    
    # Create output directory based on band name
    # Create output directory based on band name
    # Prefix with "Banda" to match user request (e.g., results/BandaC)
    # If band_name already contains "Banda" (unlikely with current logic), avoid double prefix?
    # band_name comes from specs names "C", "L", "S". So "BandaC" is correct.
    output_dir = os.path.join(RESULTS_DIR, f"Banda{band_name}")
    os.makedirs(output_dir, exist_ok=True)
    
    if show_progress:
        # tqdm.write(f"Starting simulation for Load: {load}, Heuristic: {heuristic_index}, Episodes: {n_eval_episodes}")
        pass
    else:
        # print(f"Starting simulation for Load: {load}, Heuristic: {heuristic_index}, Episodes: {n_eval_episodes}")
        pass
    
    with open(f"{output_dir}/{monitor_final_name}", "wt") as f:
        write_csv_header(
            f, topology, heuristic_index, n_eval_episodes, episode_length,
            bit_rates, band_specs, frequency_slot_bandwidth, max_span_length, env.env.modulations
        )
        write_csv_data_header(f, env.env.modulations)

        # Flattened progress bar for all episodes
        # This ensures continuous updates even for long episodes
        total_steps = n_eval_episodes * episode_length
        
        if show_progress:
            pbar = tqdm(
                total=total_steps,
                desc=f"Load {load}",
                unit="req",
                leave=False,
                position=tqdm_position
            )
        
        for ep in range(n_eval_episodes):
            obs, info = env.reset()
            done = False
            start_time = time.time()
            
            # Update description to show current episode
            if show_progress:
                pbar.set_description(f"Load {load} (Ep {ep+1}/{n_eval_episodes})")
            
            while not done:
                action, path_id, mod_id = fn_heuristic(env)
                obs, reward, done, truncated, info = env.step(action)
                if show_progress:
                    pbar.update(1)
                
            end_time = time.time()
            episode_time = end_time - start_time
            
            mean_gsnr = calculate_mean_gsnr(env)
            write_episode_results(f, ep, info, env.env.modulations, episode_time, mean_gsnr)
            f.flush()

        if show_progress:
            pbar.close()
            
    if show_progress:
        # tqdm.write(f"Finished! Results saved to: {output_dir}/{monitor_final_name}")
        pass
    else:
        # print(f"Finished! Results saved to: {output_dir}/{monitor_final_name}")
        pass

def get_band_specs(args: argparse.Namespace) -> List[dict]:
    """
    Returns band specifications based on arguments, allowing custom slot counts.
    """
    # Default values or overrides from args
    slots_c = args.slots_c
    slots_l = args.slots_l
    slots_s = args.slots_s

    # Define bands with custom physical parameters
    band_c = {
        "name": "C", 
        "start_thz": args.start_thz_c, 
        "num_slots": slots_c, 
        "noise_figure_db": args.noise_figure_c, 
        "attenuation_db_km": args.attenuation_c
    }
    band_l = {
        "name": "L", 
        "start_thz": args.start_thz_l, 
        "num_slots": slots_l, 
        "noise_figure_db": args.noise_figure_l, 
        "attenuation_db_km": args.attenuation_l
    }
    band_s = {
        "name": "S", 
        "start_thz": args.start_thz_s, 
        "num_slots": slots_s, 
        "noise_figure_db": args.noise_figure_s, 
        "attenuation_db_km": args.attenuation_s
    }

    band_specs_options = {
        "BandaC": [band_c],
        "BandaL": [band_l],
        "BandaS": [band_s],
        "BandaC+L": [band_l, band_c], # Note: Order matters for some heuristics? Usually L then C or C then L. Keeping original order.
        "BandaC+L+S": [band_c, band_l, band_s],
    }
    
    # Specific override for nobel-eu if using default slots (simulating legacy behavior)
    # But if user explicitly changed slots, we should probably respect that?
    # The original code hardcoded 344 for all bands in nobel-eu C+L+S.
    # Let's check if args match defaults. If they do, and it's nobel-eu, apply the specific logic?
    # Or better: Just set the defaults for nobel-eu to be 344 if that's what's needed.
    # Actually, the user might WANT to change it.
    # So let's just use the values provided in args.
    # If the user wants the "nobel-eu default", they should use the default args or we handle it in main defaults.
    # However, `args.slots_c` defaults to 344.
    # Wait, original code had:
    # BandaC: 344, BandaL: 406, BandaS: 647.
    # BUT for nobel-eu C+L+S: all 344.
    # This implies the default depends on topology AND band config.
    # To keep it simple and flexible: We use the args.
    # If the user wants to replicate the specific nobel-eu case, they can set slots-l=344 and slots-s=344.
    # OR we can detect if it's nobel-eu and the user hasn't manually changed the slots (how to track that?).
    # Let's stick to the args. The user has control.
    
    return band_specs_options[args.bands]

def run_single_load(args_tuple):
    """
    Wrapper function for multiprocessing. Runs a single load simulation.
    """
    (n_eval_episodes, heuristic_index, topology, seed, allow_rejection, load, episode_length,
     launch_power_dbm, frequency_slot_bandwidth, bit_rates, measure_disruptions,
     defragmentation, n_defrag_services, band_specs, debug, max_span_length, tqdm_pos) = args_tuple
    
    # tqdm_pos = get_tqdm_position()
    
    run_environment_with_monitoring(
        n_eval_episodes=n_eval_episodes,
        heuristic_index=heuristic_index,
        topology=topology,
        seed=seed,
        allow_rejection=allow_rejection,
        load=load,
        episode_length=episode_length,
        launch_power_dbm=launch_power_dbm,
        frequency_slot_bandwidth=frequency_slot_bandwidth,
        bit_rate_selection="discrete",
        bit_rates=bit_rates,
        margin=0,
        measure_disruptions=measure_disruptions,
        defragmentation=defragmentation,
        n_defrag_services=n_defrag_services,
        band_specs=band_specs,
        debug=debug,
        max_span_length=max_span_length,
        tqdm_position=tqdm_pos
    )
    return load

# ===================================================
# Main CLI Logic
# ===================================================

def clean_results_directory():
    """Removes the results directory and recreates it."""
    if os.path.exists(RESULTS_DIR):
        print(f"Cleaning results directory: {RESULTS_DIR}...")
        try:
            shutil.rmtree(RESULTS_DIR)
            print("Results directory cleaned.")
        except Exception as e:
            print(f"Error cleaning results directory: {e}")
    
    # Recreate it immediately to avoid errors
    os.makedirs(RESULTS_DIR, exist_ok=True)


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
    sim_group.add_argument('--clean', action='store_true', help='Clean results directory before running')

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
    phy_group.add_argument('--slots-c', type=int, default=344, help='Number of slots for C Band')
    phy_group.add_argument('--slots-l', type=int, default=406, help='Number of slots for L Band')
    phy_group.add_argument('--slots-s', type=int, default=647, help='Number of slots for S Band')
    
    # Physical Layer Parameters (Per Band)
    phy_group.add_argument('--start-thz-c', type=float, default=191.60, help='Start Frequency (THz) for C Band')
    phy_group.add_argument('--start-thz-l', type=float, default=185.83, help='Start Frequency (THz) for L Band')
    phy_group.add_argument('--start-thz-s', type=float, default=197.22, help='Start Frequency (THz) for S Band')

    phy_group.add_argument('--noise-figure-c', type=float, default=5.5, help='Noise Figure (dB) for C Band')
    phy_group.add_argument('--noise-figure-l', type=float, default=6.0, help='Noise Figure (dB) for L Band')
    phy_group.add_argument('--noise-figure-s', type=float, default=7.0, help='Noise Figure (dB) for S Band')

    phy_group.add_argument('--attenuation-c', type=float, default=0.191, help='Attenuation (dB/km) for C Band')
    phy_group.add_argument('--attenuation-l', type=float, default=0.200, help='Attenuation (dB/km) for L Band')
    phy_group.add_argument('--attenuation-s', type=float, default=0.220, help='Attenuation (dB/km) for S Band')

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
    parser.add_argument('-w', '--workers', type=int, default=1, help='Number of workers for parallel processing (1=sequential, -1=all CPUs)')
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
    clear_screen()
    print("\n" + "="*40)
    print("   OpticalNetworkingGym Interactive Mode   ")
    print("="*40)

    # Comparison / Plot Only Option
    print("\nAction:")
    print("  1. Run Simulation")
    print("  2. Compare/Plot Existing Results")
    print("  3. Clean Results Directory")
    action = get_user_choice(["Run Simulation", "Compare/Plot Existing Results", "Clean Results Directory"], "Select Action")
    
    if action == "Clean Results Directory":
        clean_results_directory()
        # Ask what to do next
        return interactive_mode(args)
    elif action == "Compare/Plot Existing Results":
        args.skip_sim = True
    else:
        args.skip_sim = False

    # Topology
    script_dir = os.path.dirname(os.path.abspath(__file__))
    topologies_dir = os.path.join(script_dir, "..", "topologies")
    topologies = [f for f in os.listdir(topologies_dir) if f.endswith(('.xml', '.txt'))]
    if topologies:
        clear_screen()
        args.topology = get_user_choice(sorted(topologies), "Select Topology")
    else:
        print("No topology files found. Using default.")

    if args.skip_sim:
        return args

    # Heuristic
    heuristics = ['first_fit', 'highest_snr', 'lowest_fragmentation', 'load_balancing', 'priority_c_l', 'best_band']
    clear_screen()
    args.heuristic = get_user_choice(heuristics, "Select Heuristic")

    # Bands
    bands = ['BandaC', 'BandaL', 'BandaS', 'BandaC+L', 'BandaC+L+S']
    # Show default slots in help?
    # It's hard to show dynamic slots here without clutter.
    clear_screen()
    args.bands = get_user_choice(bands, "Select Bands")

    # Load
    clear_screen()
    print("\nLoad Configuration:")
    print("  Format: single value (e.g., 100) or range start:stop:step (e.g., 100:1000:100)")
    args.load = get_user_input("Enter Load", default="100:1000:100")

    # Episodes
    clear_screen()
    args.episodes = get_user_input("Number of Episodes", default=5, value_type=int)

    # Advanced Options
    clear_screen()
    if get_user_input("\nConfigure advanced options? (y/N)", default="n", value_type=bool):
        args.episode_length = get_user_input("Episode Length", default=100000, value_type=int)
        args.seed = get_user_input("Random Seed", default=50, value_type=int)
        args.power = get_user_input("Launch Power (dBm)", default=0.0, value_type=float)
        args.defrag = get_user_input("Enable Defragmentation? (y/N)", default="n", value_type=bool)
        if args.defrag:
            args.defrag_services = get_user_input("Defrag Services Count", default=10, value_type=int)
        args.debug = get_user_input("Enable Debug Mode? (y/N)", default="n", value_type=bool)
        args.workers = get_user_input("Number of Workers (1=seq, -1=all)", default=1, value_type=int)
        
        if get_user_input("Configure Band Slots? (y/N)", default="n", value_type=bool):
            print(f"Current Defaults: C={args.slots_c}, L={args.slots_l}, S={args.slots_s}")
            args.slots_c = get_user_input("Slots for C Band", default=args.slots_c, value_type=int)
            args.slots_l = get_user_input("Slots for L Band", default=args.slots_l, value_type=int)
            args.slots_s = get_user_input("Slots for S Band", default=args.slots_s, value_type=int)

        if get_user_input("Configure Physical Layer? (y/N)", default="n", value_type=bool):
            # Global
            print("\n--- Global Physical Parameters ---")
            bit_rates_str = get_user_input("Bit Rates (comma separated)", default="48,120")
            args.bit_rates = [int(x.strip()) for x in bit_rates_str.split(',')]
            args.max_span_length = get_user_input("Max Span Length (km)", default=args.max_span_length, value_type=float)
            args.slot_bandwidth = get_user_input("Slot Bandwidth (Hz)", default=args.slot_bandwidth, value_type=float)
            
            # Per Band
            print("\n--- Band C Parameters ---")
            args.start_thz_c = get_user_input("Start Freq (THz)", default=args.start_thz_c, value_type=float)
            args.noise_figure_c = get_user_input("Noise Figure (dB)", default=args.noise_figure_c, value_type=float)
            args.attenuation_c = get_user_input("Attenuation (dB/km)", default=args.attenuation_c, value_type=float)
            
            print("\n--- Band L Parameters ---")
            args.start_thz_l = get_user_input("Start Freq (THz)", default=args.start_thz_l, value_type=float)
            args.noise_figure_l = get_user_input("Noise Figure (dB)", default=args.noise_figure_l, value_type=float)
            args.attenuation_l = get_user_input("Attenuation (dB/km)", default=args.attenuation_l, value_type=float)
            
            print("\n--- Band S Parameters ---")
            args.start_thz_s = get_user_input("Start Freq (THz)", default=args.start_thz_s, value_type=float)
            args.noise_figure_s = get_user_input("Noise Figure (dB)", default=args.noise_figure_s, value_type=float)
            args.attenuation_s = get_user_input("Attenuation (dB/km)", default=args.attenuation_s, value_type=float)

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
        
        # Handle Clean Flag
        if args.clean:
            clean_results_directory()
            # Reset flag so it doesn't clean again in a loop if we were to loop (though main loop re-parses args)
            # But args is local.
            pass
        
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
             # Try checking if it's just the filename provided and it exists in current dir or relative
             if os.path.exists(args.topology):
                topology_path = args.topology
             else:
                print(f"Error: Topology file '{topology_path}' not found.")
                sys.exit(1)

        # Run Simulation Block
        if not args.skip_sim:
            # Band Specifications
            # Use the helper function to get specs based on args (including custom slots)
            band_specs = get_band_specs(args)

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
            clear_screen()
            print("\n" + "="*40)
            print("       SIMULATION CONFIGURATION       ")
            print("="*40)
            print(f"{'Topology':<20}: {args.topology}")
            print(f"{'Bands':<20}: {args.bands}")
            # Show slots for selected bands
            slots_summary = ", ".join([f"{b['name']}:{b['num_slots']}" for b in band_specs])
            print(f"{'Slots':<20}: {slots_summary}")
            
            # Show Physical Params
            print(f"{'Bit Rates':<20}: {args.bit_rates}")
            print(f"{'Span Length':<20}: {args.max_span_length} km")
            print(f"{'Slot Bandwidth':<20}: {args.slot_bandwidth} Hz")
            print("\n--- Band Details ---")
            for b in band_specs:
                print(f"  Band {b['name']}: Start={b['start_thz']}THz, Noise={b['noise_figure_db']}dB, Atten={b['attenuation_db_km']}dB/km")
            print("-" * 20)
            
            print(f"{'Heuristic':<20}: {args.heuristic} (Index: {heuristic_idx})")
            print(f"{'Episodes':<20}: {args.episodes}")
            print(f"{'Episode Length':<20}: {args.episode_length}")
            print(f"{'Seed':<20}: {args.seed}")
            print(f"{'Power':<20}: {args.power} dBm")
            print(f"{'Defragmentation':<20}: {args.defrag}")
            print(f"{'Workers':<20}: {args.workers}")
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
                clear_screen()
                print("="*40)
                print("       SIMULATION RUNNING...       ")
                print("="*40)
                print(f"{'Topology':<20}: {args.topology}")
                print(f"{'Bands':<20}: {args.bands}")
                print(f"{'Heuristic':<20}: {args.heuristic}")
                print(f"{'Loads':<20}: {loads}")
                print("-" * 40)
                # print(f"Running simulation for topology: {args.topology}")
                # print(f"Bands: {args.bands}")
                # print(f"Loads to simulate: {loads}")
                
                # Determine number of workers
                num_workers = args.workers
                if num_workers == -1:
                    num_workers = cpu_count()

                if num_workers > 1 and len(loads) > 1:
                    # print(f"\nRunning simulations in parallel with {num_workers} workers...")
                    
                    # Prepare arguments for each load
                    load_args = []
                    for idx, current_load in enumerate(loads):
                        # Calculate position: 1-based index (0 is for total progress)
                        # We want them to stack nicely.
                        # If we have N workers, we can use positions 1 to N.
                        # But since we use imap_unordered, tasks start as soon as a worker is free.
                        # Ideally, we want a unique position for each *active* task.
                        # However, mapping task index to position is simpler and works if N_tasks ~ N_workers.
                        # If N_tasks >> N_workers, bars might overwrite or jump if we reuse positions blindly.
                        # But reusing positions 1..N_workers is standard for pool.
                        # Let's try assigning position based on idx % num_workers + 1
                        # This way, if we have 12 workers, we use lines 1-12.
                        # When worker 1 finishes task 1 and starts task 13, it reuses line 1.
                        # This is cleaner than unique line for every single task (which would scroll off screen).
                        
                        pos = (idx % num_workers) + 1
                        
                        load_args.append((
                            args.episodes,
                            heuristic_idx,
                            topology,
                            args.seed,
                            args.allow_rejection,
                            current_load,
                            args.episode_length,
                            args.power,
                            args.slot_bandwidth,
                            tuple(args.bit_rates),
                            args.measure_disruptions,
                            args.defrag,
                            args.defrag_services,
                            band_specs,
                            args.debug,
                            args.max_span_length,
                            pos # Pass explicit position
                        ))
                    
                    with Pool(processes=num_workers) as pool:
                        # Use imap_unordered with tqdm for progress tracking
                        # Position 0 is for this total progress bar
                        list(tqdm(
                            pool.imap_unordered(run_single_load, load_args),
                            total=len(load_args),
                            desc="Total Progress",
                            unit="load",
                            position=0
                        ))
                else:
                    # print(f"\nRunning simulations sequentially...")
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
            clear_screen()
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
                # metrics_map = {
                #     "1": "episode_service_blocking_rate",
                #     "2": "episode_bit_rate_blocking_rate",
                #     "3": "mean_gsnr",
                #     "4": "episode_service_realocations"
                # }
                
                # selected_metrics = []
                
                # if args.interactive:
                #     print("\nSelect Metric to Plot:")
                #     print("  1. Blocking Probability")
                #     print("  2. Bit Rate Blocking Rate")
                #     print("  3. Mean GSNR")
                #     print("  4. Service Reallocations")
                #     print("  5. All Metrics")
                    
                #     choice = input("Select an option (1-5): ").strip()
                #     if choice == "5":
                #         selected_metrics = list(metrics_map.values())
                #     elif choice in metrics_map:
                #         selected_metrics = [metrics_map[choice]]
                #     else:
                #         print("Invalid selection. Defaulting to Blocking Probability.")
                #         selected_metrics = ["episode_service_blocking_rate"]
                # else:
                #     # Non-interactive default
                #     selected_metrics = ["episode_service_blocking_rate"]
                
                # Default to Blocking Probability only as requested
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
