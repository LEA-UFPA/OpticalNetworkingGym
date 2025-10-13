# examples/OFC_2025/utils.py

import csv
import time
import random
from pathlib import Path
from typing import Tuple

import numpy as np

from optical_networking_gym.topology import Modulation, get_topology
from optical_networking_gym.wrappers.qrmsa_gym import QRMSAEnvWrapper


class SimulationUtils:
    @staticmethod
    def define_modulations() -> Tuple[Modulation, ...]:
        return (
            Modulation("BPSK", 100_000, 1, minimum_osnr=3.71, inband_xt=-14),
            Modulation("QPSK", 2_000,   2, minimum_osnr=6.72, inband_xt=-17),
            Modulation("8QAM",  1_000,   3, minimum_osnr=10.84, inband_xt=-20),
            Modulation("16QAM",   500,   4, minimum_osnr=13.24, inband_xt=-23),
            Modulation("32QAM",   250,   5, minimum_osnr=16.16, inband_xt=-26),
            Modulation("64QAM",   125,   6, minimum_osnr=19.01, inband_xt=-29),
        )

    @staticmethod
    def get_modulations(names_str: str) -> Tuple[Modulation, ...]:
        mods = {m.name.upper(): m for m in SimulationUtils.define_modulations()}
        return tuple(mods[name.strip().upper()] for name in names_str.split(","))

    @staticmethod
    def create_environment(
        topology_name: str,
        modulation_names: str,
        seed: int = 0,
        bit_rates: Tuple[int, ...] = (10, 40, 100, 400),
        load: int = 250,
        num_spectrum_resources: int = 320,
        episode_length: int = 1000,
        modulations_to_consider: int = None,
        defragmentation: bool = False,
        k_paths: int = 5,
        gen_observation: bool = True,
    ) -> dict:

        base_dir    = Path(__file__).resolve().parents[2]
        topo_dir    = base_dir / "examples" / "topologies"
        for ext in ("xml", "txt"):
            topo_path = topo_dir / f"{topology_name}.{ext}"
            if topo_path.exists():
                break
        else:
            raise FileNotFoundError(f"Topologia '{topology_name}.xml|.txt' não encontrada em {topo_dir}")

        mods = SimulationUtils.get_modulations(modulation_names)

        mc = modulations_to_consider or len(mods)

        random.seed(seed)
        np.random.seed(seed)

        return dict(
            topology= get_topology(
                str(topo_path),
                topology_name,
                mods,
                max_span_length=80,
                default_attenuation=0.2,
                default_noise_figure=4.5,
                k_paths=k_paths,
            ),
            seed=seed,
            allow_rejection=True,
            load=load,
            episode_length=episode_length,
            num_spectrum_resources=num_spectrum_resources,
            launch_power_dbm=0,
            frequency_slot_bandwidth=12.5e9,
            frequency_start=3e8 / 1565e-9,
            bandwidth= num_spectrum_resources * 12.5e9,
            bit_rate_selection="discrete",
            bit_rates=bit_rates,
            margin=0,
            measure_disruptions=False,
            file_name="",
            k_paths=k_paths,
            modulations_to_consider=mc,
            defragmentation=defragmentation,
            n_defrag_services=0,
            gen_observation= gen_observation,
            qot_constraint="ASE+NLI"  # Usar OSNR+NLI ao invés de DIST
        )

    @staticmethod
    def run_heuristic(
        n_eval_episodes: int,
        env_args: dict,
        csv_output: str,
        heuristic_fn,
    ):
        """
        Roda uma heurística em QRMSAEnvWrapper e salva métricas em CSV.
        """
        env = QRMSAEnvWrapper(**env_args)

        header = [
            "episode",
            "service_blocking_rate", "episode_service_blocking_rate",
            "bit_rate_blocking_rate", "episode_bit_rate_blocking_rate",
        ]
        header += [f"modulation_{m.spectral_efficiency}" for m in env.env.modulations]
        header += [
            "episode_disrupted_services", 
            "episode_defrag_cicles",
            "episode_service_realocations",
            "fragmentation_shannon_entropy",
            "fragmentation_route_cuts", 
            "fragmentation_route_rss",
            "episode_time"
        ]

        with open(csv_output, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)

            for ep in range(n_eval_episodes):
                start = time.time()
                obs, info = env.reset()
                done = False

                while not done:
                    action = heuristic_fn(info['mask'])
                    obs, _, done, _, info = env.step(action)

                ep_time = time.time() - start
                row = [
                    ep,
                    info.get("service_blocking_rate", 0.0),
                    info.get("episode_service_blocking_rate", 0.0),
                    info.get("bit_rate_blocking_rate", 0.0),
                    info.get("episode_bit_rate_blocking_rate", 0.0),
                ]
                row += [info.get(f"modulation_{mf.spectral_efficiency}", 0.0) for mf in env.env.modulations]
                row += [
                    info.get("episode_disrupted_services", 0), 
                    info.get("episode_defrag_cicles", 0),
                    info.get("episode_service_realocations", 0),
                    info.get("fragmentation_shannon_entropy", 0.0),
                    info.get("fragmentation_route_cuts", 0),
                    info.get("fragmentation_route_rss", 0.0),
                    f"{ep_time:.2f}"
                ]
                writer.writerow(row)

        print(f"Resultados salvos em: {csv_output}")
