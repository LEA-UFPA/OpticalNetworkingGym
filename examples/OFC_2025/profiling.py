import cProfile
import pstats
import io
from datetime import datetime

import random
import numpy as np

from utils import SimulationUtils
from optical_networking_gym.heuristics.heuristics import shortest_available_path_first_fit_best_modulation
from optical_networking_gym.wrappers.qrmsa_gym import QRMSAEnvWrapper

def profile_step():
    # 1) Prepare ambiente igual ao seu main()
    env_args = SimulationUtils.create_environment(
        topology_name="nobel-eu",
        modulation_names="BPSK, QPSK, 8QAM, 16QAM, 32QAM, 64QAM",
        seed=10,
        bit_rates=(10, 40, 100, 400),
        load=300,
        num_spectrum_resources=320,
        episode_length=500,
        modulations_to_consider=4,
        defragmentation=True,
        k_paths=3,
        gen_observation=True,
    )
    env = QRMSAEnvWrapper(**env_args)

    # 2) Reset e obtenha obs+mask iniciais
    obs, info = env.reset()

    # 3) Escolha a ação inicial
    action = shortest_available_path_first_fit_best_modulation(info['mask'])

    # 4) Profile exatamente a chamada a step()
    pr = cProfile.Profile()
    pr.enable()

    obs2, reward, done, truncated, info2 = env.step(action)

    pr.disable()

    # 5) Formate e imprima relatório
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')  # ou 'tottime', 'ncalls'
    ps.print_stats(50)  # top 50 linhas

    print("=== Profiling da chamada env.step() ===")
    print(s.getvalue())

if __name__ == "__main__":
    profile_step()
