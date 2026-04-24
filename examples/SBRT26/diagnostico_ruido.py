"""Diagnóstico numérico dos componentes de ruído da camada física.

Calcula ASE, NLI, GSNR e OSNR para diferentes cenários na topologia
configurada, mostrando o impacto da margem na decisão de aceitação.
"""
from __future__ import annotations

import math
import numpy as np

from optical_networking_gym_v2.defaults import (
    DEFAULT_K_PATHS,
    DEFAULT_LAUNCH_POWER_DBM,
    DEFAULT_NUM_SPECTRUM_RESOURCES,
    MODULATION_CATALOG,
    get_modulations,
    resolve_topology,
)
from optical_networking_gym_v2.network.topology import TopologyModel

# ── Constantes da camada física (mesmas do qot_kernel.py) ──
ABS_BETA_2 = abs(-21.3e-27)       # |β₂| em s²/m
GAMMA      = 1.3e-3               # γ em 1/(W·m)
H_PLANCK   = 6.626e-34            # h em J·s
FREQ_SLOT_BW = 12.5e9             # largura do slot (Hz)
FREQ_START = 3e8 / 1565e-9        # início da banda C (Hz)

TOPOLOGY_ID = "nobel-eu"
NUM_SLOTS   = DEFAULT_NUM_SPECTRUM_RESOURCES  # 320
LAUNCH_POWER_DBM = DEFAULT_LAUNCH_POWER_DBM   # 1.0 dBm
LAUNCH_POWER_W = 10 ** ((LAUNCH_POWER_DBM - 30.0) / 10.0)

topology = TopologyModel.from_file(
    resolve_topology(TOPOLOGY_ID),
    topology_id=TOPOLOGY_ID,
    k_paths=DEFAULT_K_PATHS,
)

# ── Escolher o caminho mais longo (pior caso) ──
longest_path = max(topology.paths, key=lambda p: p.length_km)
print(f"Topologia: {TOPOLOGY_ID}")
print(f"Caminho mais longo: {' → '.join(longest_path.node_names)}")
print(f"Comprimento: {longest_path.length_km:.1f} km")
print(f"Hops: {longest_path.hops}")
print(f"Potência lançamento: {LAUNCH_POWER_DBM} dBm ({LAUNCH_POWER_W*1e3:.2f} mW)")
print()

# ── Calcular spans no caminho ──
total_spans = 0
for link_id in longest_path.link_ids:
    link = topology.links[link_id]
    total_spans += len(link.spans)
print(f"Total de spans no caminho: {total_spans}")
print()

# ── Cenário: canal isolado (sem NLI cruzada, apenas SPM) ──
modulations = get_modulations("QPSK,8QAM,16QAM,32QAM,64QAM")

# Serviço com 4 slots (50 GHz de largura, típico para 100 Gbps QPSK)
service_num_slots = 4
bandwidth = FREQ_SLOT_BW * service_num_slots
center_freq = FREQ_START + FREQ_SLOT_BW * (NUM_SLOTS // 2)  # centro do espectro

print(f"Largura do serviço: {bandwidth/1e9:.1f} GHz ({service_num_slots} slots)")
print(f"Frequência central: {center_freq/1e12:.4f} THz")
print()

nli_prefactor = ((LAUNCH_POWER_W / bandwidth) ** 3) * (8.0 / (27.0 * math.pi * ABS_BETA_2)) * (GAMMA**2) * bandwidth

acc_ase = 0.0
acc_nli = 0.0

for link_id in longest_path.link_ids:
    link = topology.links[link_id]
    for span in link.spans:
        alpha = span.attenuation_normalized    # Np/m
        nf    = span.noise_figure_normalized   # linear
        L_m   = span.length_km * 1e3           # metros

        # ASE deste span
        p_ase = bandwidth * H_PLANCK * center_freq * (math.exp(2 * alpha * L_m) - 1) * nf

        # NLI deste span (apenas SPM, sem serviços interferentes)
        l_eff = (1.0 - math.exp(-2.0 * alpha * L_m)) / (2.0 * alpha)
        sum_phi = math.asinh(math.pi**2 * ABS_BETA_2 * bandwidth**2 / (4.0 * alpha))
        p_nli = nli_prefactor * l_eff * sum_phi

        acc_ase += p_ase / LAUNCH_POWER_W
        acc_nli += p_nli / LAUNCH_POWER_W

acc_gsnr = acc_ase + acc_nli

osnr_db = 10 * math.log10(1.0 / acc_gsnr)
ase_only_db = 10 * math.log10(1.0 / acc_ase)
nli_only_db = 10 * math.log10(1.0 / acc_nli) if acc_nli > 0 else float('inf')
nli_share = acc_nli / (acc_ase + acc_nli) * 100 if (acc_ase + acc_nli) > 0 else 0

print("=" * 70)
print("RESULTADO — CANAL ISOLADO (apenas SPM, sem XPM)")
print("=" * 70)
print(f"  Ruído ASE acumulado (linear normalizado): {acc_ase:.6e}")
print(f"  Ruído NLI acumulado (linear normalizado): {acc_nli:.6e}")
print(f"  GSNR acumulado (linear normalizado):      {acc_gsnr:.6e}")
print()
print(f"  ASE-only (se desconsiderasse NLI):  {ase_only_db:.2f} dB")
print(f"  NLI-only (se desconsiderasse ASE):  {nli_only_db:.2f} dB")
print(f"  OSNR (GSNR = ASE + NLI):            {osnr_db:.2f} dB")
print(f"  Degradação por NLI:                  {ase_only_db - osnr_db:.2f} dB")
print(f"  Proporção NLI/total:                 {nli_share:.2f}%")
print()

print("=" * 70)
print("IMPACTO DA MARGEM NA DECISÃO DE ACEITAÇÃO")
print("=" * 70)
print(f"{'Modulação':<10} {'OSNR mín':<12} {'OSNR calc':<12} {'Folga':<10} ", end="")
for m in [0.0, 0.5, 1.0, 2.0, 3.0]:
    print(f"{'m='+str(m)+'dB':<10}", end="")
print()
print("-" * 92)

for mod in modulations:
    folga = osnr_db - mod.minimum_osnr
    line = f"{mod.name:<10} {mod.minimum_osnr:<12.1f} {osnr_db:<12.2f} {folga:<10.2f} "
    for margin in [0.0, 0.5, 1.0, 2.0, 3.0]:
        threshold = mod.minimum_osnr + margin
        aceita = osnr_db >= threshold
        symbol = "✓ Aceita" if aceita else "✗ Bloqueia"
        line += f"{symbol:<10}"
    print(line)

print()
print("=" * 70)
print("CENÁRIO COM CARGA (XPM de vizinhos)")
print("=" * 70)
# Simular efeito de 5, 10, 20, 40 vizinhos no pior caso
for n_neighbors in [0, 5, 10, 20, 40]:
    acc_ase_loaded = 0.0
    acc_nli_loaded = 0.0

    for link_id in longest_path.link_ids:
        link = topology.links[link_id]
        for span in link.spans:
            alpha = span.attenuation_normalized
            nf = span.noise_figure_normalized
            L_m = span.length_km * 1e3

            p_ase = bandwidth * H_PLANCK * center_freq * (math.exp(2 * alpha * L_m) - 1) * nf

            l_eff = (1.0 - math.exp(-2.0 * alpha * L_m)) / (2.0 * alpha)
            l_eff_a = 1.0 / (2.0 * alpha)
            sum_phi = math.asinh(math.pi**2 * ABS_BETA_2 * bandwidth**2 / (4.0 * alpha))

            # Adicionar contribuição XPM de vizinhos
            neighbor_bw = bandwidth  # mesmo tamanho
            for i in range(1, n_neighbors + 1):
                delta_f = i * bandwidth * 1.5  # separação entre canais
                phi_xpm = (
                    math.asinh(math.pi**2 * ABS_BETA_2 * l_eff_a * neighbor_bw * (delta_f + neighbor_bw / 2))
                    - math.asinh(math.pi**2 * ABS_BETA_2 * l_eff_a * neighbor_bw * (delta_f - neighbor_bw / 2))
                ) - (1.0 * (neighbor_bw / abs(delta_f)) * (5.0 / 3.0) * (l_eff / L_m))
                sum_phi += phi_xpm

            p_nli = nli_prefactor * l_eff * sum_phi

            acc_ase_loaded += p_ase / LAUNCH_POWER_W
            acc_nli_loaded += p_nli / LAUNCH_POWER_W

    acc_gsnr_loaded = acc_ase_loaded + acc_nli_loaded
    osnr_loaded = 10 * math.log10(1.0 / acc_gsnr_loaded)
    nli_pct = acc_nli_loaded / acc_gsnr_loaded * 100

    print(f"  {n_neighbors:2d} vizinhos: OSNR = {osnr_loaded:6.2f} dB  |  NLI/total = {nli_pct:5.1f}%  |  ", end="")
    for mod in modulations[:3]:  # QPSK, 8QAM, 16QAM
        for margin in [0.0, 1.0, 2.0, 3.0]:
            if osnr_loaded >= mod.minimum_osnr + margin:
                status = "✓"
            else:
                status = "✗"
            print(f"{mod.name}+{margin}dB:{status} ", end="")
    print()

print()
print("=" * 70)
print("CONCLUSÃO")
print("=" * 70)
print("""
A soma dos ruídos é:  GSNR⁻¹ = ASE⁻¹ + NLI⁻¹  (em escala linear)
    OSNR(dB) = 10·log₁₀(P_launch / (P_ASE + P_NLI))

SIM, diminuir a margem AFETA DIRETAMENTE a probabilidade de bloqueio:
  - Margem MENOR → threshold MENOR → MAIS serviços aceitos → MENOR bloqueio QoT
  - Margem MAIOR → threshold MAIOR → MENOS serviços aceitos → MAIOR bloqueio QoT
  - MAS com margem menor, serviços aceitos ficam mais vulneráveis a
    degradação NLI quando novos vizinhos são adicionados (DISRUPTIONS)

O trade-off é:  bloqueio QoT ↔ disrupções por degradação de OSNR.
""")
