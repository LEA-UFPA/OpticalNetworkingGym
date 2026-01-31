#!/usr/bin/env python3
"""
Teste completo da nova funcionalidade de captura detalhada de bloqueios
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from optical_networking_gym.topology import Modulation, get_topology
from optical_networking_gym.wrappers.qrmsa_gym import QRMSAEnvWrapper
from optical_networking_gym.heuristics.heuristics import shortest_available_path_first_fit_best_modulation_best_band
import numpy as np

# Configuração
topology_path = "examples/topologies/nobel-eu.xml"
modulations = (
    Modulation(name="QPSK", maximum_length=10_000, spectral_efficiency=2, minimum_osnr=6.72, inband_xt=-17),
    Modulation(name="8QAM", maximum_length=1_000, spectral_efficiency=3, minimum_osnr=10.84, inband_xt=-20),
    Modulation(name="16QAM", maximum_length=500, spectral_efficiency=4, minimum_osnr=13.24, inband_xt=-23),
)

topology = get_topology(topology_path, None, modulations, max_span_length=100, 
                       default_attenuation=0.2, default_noise_figure=4.5, k_paths=2)

band_specs = [
    {"name": "C", "start_thz": 191.60, "num_slots": 344, "noise_figure_db": 5.5, "attenuation_db_km": 0.191},
    {"name": "L", "start_thz": 185.83, "num_slots": 406, "noise_figure_db": 6.0, "attenuation_db_km": 0.200},
]

env_args = dict(
    topology=topology,
    band_specs=band_specs,
    seed=50,
    allow_rejection=True,
    load=800,  # Carga moderada
    episode_length=1000,
    launch_power_dbm=-10,  # POTÊNCIA BAIXA para forçar bloqueios por OSNR
    frequency_slot_bandwidth=12.5e9,
    bit_rate_selection="discrete",
    bit_rates=(48, 120),
    margin=3,  # MARGEM ALTA para forçar bloqueios por OSNR
    measure_disruptions=False,
    file_name="",
    k_paths=2,
    defragmentation=False,
    n_defrag_services=0,
    gen_observation=False,
)

env = QRMSAEnvWrapper(**env_args)

print("=" * 70)
print("TESTE COMPLETO - DECOMPOSIÇÃO ASE/NLI NOS BLOQUEIOS")
print("=" * 70)
print(f"Topologia: {topology.name}")
print(f"Carga: {env_args['load']} Erlangs")
print(f"Potência de lançamento: {env_args['launch_power_dbm']} dBm")
print(f"Margem OSNR: {env_args['margin']} dB")
print(f"Episódio: {env_args['episode_length']} serviços")
print("=" * 70)

obs, info = env.reset()
done = False
step_count = 0
rejects_osnr = 0
rejects_resource = 0
debug_limit = 3

while not done:
    action, blocked_resources, blocked_osnr, blocking_info = shortest_available_path_first_fit_best_modulation_best_band(env)
    
    is_rejection = (action == (env.action_space.n - 1))
    
    if is_rejection:
        # Setar flags de bloqueio
        env.env.current_service.blocked_due_to_resources = blocked_resources
        env.env.current_service.blocked_due_to_osnr = blocked_osnr
        
        # IMPORTANTE: Setar ASE e NLI quando disponíveis
        if blocking_info['osnr'] > 0:
            env.env.current_service.OSNR = blocking_info['osnr']
        if blocking_info['ase'] > 0:
            env.env.current_service.ASE = blocking_info['ase']
        if blocking_info['nli'] > 0:
            env.env.current_service.NLI = blocking_info['nli']
        
        if step_count < debug_limit:
            print(f"\n[DEBUG Step {step_count}] Bloqueio detectado:")
            print(f"  Resources: {blocked_resources}, OSNR: {blocked_osnr}")
            print(f"  Banda: {blocking_info['band']}")
            print(f"  OSNR: {blocking_info['osnr']:.2f} dB (req: {blocking_info['osnr_req']:.2f} dB)")
            print(f"  ASE: {blocking_info['ase']:.2f} dB, NLI: {blocking_info['nli']:.2f} dB")
            if blocking_info['ase'] > 0 and blocking_info['nli'] > 0:
                if blocking_info['ase'] > blocking_info['nli']:
                    print(f"  → Dominado por ASE (ruído)")
                else:
                    print(f"  → Dominado por NLI (interferência)")
        
        if blocked_osnr:
            rejects_osnr += 1
        if blocked_resources:
            rejects_resource += 1
    
    obs, reward, done, truncated, info = env.step(action)
    step_count += 1
    
    if step_count % 200 == 0:
        print(f"Step {step_count}/{env_args['episode_length']} - OSNR={rejects_osnr}, Resources={rejects_resource}")

print("\n" + "=" * 70)
print("RESULTADOS FINAIS")
print("=" * 70)
print(f"Taxa de bloqueio: {info.get('episode_service_blocking_rate', 0):.4f}")
print()
print("CONTADORES DO AMBIENTE:")
print(f"  Bloqueios por falta de recursos: {info.get('blocked_due_to_resources', 0)}")
print(f"  Bloqueios por OSNR insuficiente: {info.get('blocked_due_to_osnr', 0)}")
print(f"    - ASE dominante (ruído): {info.get('blocked_due_to_ase_dominant', 0)}")
print(f"    - NLI dominante (interferência): {info.get('blocked_due_to_nli_dominant', 0)}")
print("=" * 70)

# Análise
total_osnr_blocks = info.get('blocked_due_to_osnr', 0)
ase_blocks = info.get('blocked_due_to_ase_dominant', 0)
nli_blocks = info.get('blocked_due_to_nli_dominant', 0)

print("\nANÁLISE:")
if ase_blocks > 0:
    print(f"✅ Bloqueios dominados por ASE detectados: {ase_blocks} ({100*ase_blocks/total_osnr_blocks:.1f}% dos bloqueios OSNR)")
if nli_blocks > 0:
    print(f"✅ Bloqueios dominados por NLI detectados: {nli_blocks} ({100*nli_blocks/total_osnr_blocks:.1f}% dos bloqueios OSNR)")
if ase_blocks == 0 and nli_blocks == 0 and total_osnr_blocks > 0:
    print("⚠️  Nenhuma decomposição ASE/NLI foi registrada, apesar de haver bloqueios por OSNR.")
    print("   Isso pode indicar um problema na lógica de decomposição.")

print("=" * 70)
