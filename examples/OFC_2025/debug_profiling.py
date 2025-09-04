import os
import cProfile
import pstats
from datetime import datetime
import random
import numpy as np

from utils import SimulationUtils
from optical_networking_gym.heuristics.heuristics import shortest_available_path_first_fit_best_modulation

prof_dir = os.path.join(os.getcwd(), "perf")

os.makedirs(prof_dir, exist_ok=True)
print("running .... ")
episode_length = 500
ts = datetime.now().strftime("%Y%m%d_%H%M%S")
env_args = SimulationUtils.create_environment(
    topology_name="nobel-eu",
    modulation_names="BPSK, QPSK, 8QAM, 16QAM, 32QAM, 64QAM",
    seed=10,
    bit_rates=(10, 40, 100, 400),
    load=300,
    num_spectrum_resources=320,
    episode_length=episode_length,
    modulations_to_consider=4,
    defragmentation=True,
    k_paths=3,
    gen_observation=True,
)

csv_output = f"first_fit_{episode_length}_{ts}.csv"
prof_file = os.path.join(prof_dir, f"profile_{ts}.prof")

profiler = cProfile.Profile()
profiler.enable()

SimulationUtils.run_heuristic(
    n_eval_episodes=1,
    env_args=env_args,
    csv_output=csv_output,
    heuristic_fn=shortest_available_path_first_fit_best_modulation,
)

profiler.disable()

with open(prof_file, "w") as f:
    ps = pstats.Stats(profiler, stream=f)
    ps.strip_dirs().sort_stats("cumulative").print_stats(30)

print(f"CSV results saved to: {csv_output}")
print(f"Profile data saved to: {prof_file}")

