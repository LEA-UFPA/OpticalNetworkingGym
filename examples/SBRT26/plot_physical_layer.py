"""
Visualização da Camada Física — Optical Networking Gym
======================================================
Gera gráficos de qualidade para publicação mostrando:
  1. OSNR vs comprimento do caminho (todos os k-caminhos)
  2. Decomposição ASE vs NLI por comprimento
  3. Zonas de aceitação por modulação com linhas de margem
  4. Degradação por XPM (carga) no caminho mais longo
  5. Mapa de calor: quais pares origem-destino são viáveis por margem
  6. Perfil de ruído span-a-span no pior caminho

Saída: arquivos PNG em alta resolução no diretório de resultados.

Uso:
    PYTHONPATH=src python examples/SBRT26/plot_physical_layer.py
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from optical_networking_gym_v2.defaults import (
    DEFAULT_K_PATHS,
    DEFAULT_LAUNCH_POWER_DBM,
    DEFAULT_NUM_SPECTRUM_RESOURCES,
    MODULATION_CATALOG,
    get_modulations,
    resolve_topology,
)
from optical_networking_gym_v2.network.topology import TopologyModel

# ── Configuração global ──────────────────────────────────────────────
TOPOLOGY_ID = "nobel-eu"
NUM_SLOTS = DEFAULT_NUM_SPECTRUM_RESOURCES
LAUNCH_POWER_DBM = DEFAULT_LAUNCH_POWER_DBM
LAUNCH_POWER_W = 10 ** ((LAUNCH_POWER_DBM - 30.0) / 10.0)
FREQ_SLOT_BW = 12.5e9
FREQ_START = 3e8 / 1565e-9
SERVICE_NUM_SLOTS = 4
BANDWIDTH = FREQ_SLOT_BW * SERVICE_NUM_SLOTS
CENTER_FREQ = FREQ_START + FREQ_SLOT_BW * (NUM_SLOTS // 2)

# Constantes físicas
ABS_BETA_2 = abs(-21.3e-27)
GAMMA = 1.3e-3
H_PLANCK = 6.626e-34

OUTPUT_DIR = Path(__file__).resolve().parent / "resultados" / "physical_layer"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Estilo dos gráficos ─────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Paleta de cores
COLORS = {
    "ase": "#3498db",
    "nli": "#e74c3c",
    "osnr": "#2c3e50",
    "gsnr": "#8e44ad",
    "qpsk": "#27ae60",
    "8qam": "#f39c12",
    "16qam": "#e74c3c",
    "32qam": "#9b59b6",
    "64qam": "#1abc9c",
    "accept": "#2ecc71",
    "block": "#e74c3c",
    "margin": "#95a5a6",
}

MOD_COLORS = {
    "QPSK": "#27ae60",
    "8QAM": "#f39c12",
    "16QAM": "#e74c3c",
    "32QAM": "#9b59b6",
    "64QAM": "#1abc9c",
}


# ── Funções de cálculo ───────────────────────────────────────────────
def calculate_path_noise(topology: TopologyModel, path, n_neighbors: int = 0):
    """Calcula ASE, NLI e OSNR para um caminho, com n_neighbors vizinhos XPM."""
    nli_prefactor = (
        ((LAUNCH_POWER_W / BANDWIDTH) ** 3)
        * (8.0 / (27.0 * math.pi * ABS_BETA_2))
        * (GAMMA ** 2)
        * BANDWIDTH
    )
    acc_ase = 0.0
    acc_nli = 0.0
    ase_per_span = []
    nli_per_span = []

    for link_id in path.link_ids:
        link = topology.links[link_id]
        for span in link.spans:
            alpha = span.attenuation_normalized
            nf = span.noise_figure_normalized
            L_m = span.length_km * 1e3

            # ASE
            p_ase = BANDWIDTH * H_PLANCK * CENTER_FREQ * (math.exp(2 * alpha * L_m) - 1) * nf

            # NLI (SPM + XPM)
            l_eff = (1.0 - math.exp(-2.0 * alpha * L_m)) / (2.0 * alpha)
            l_eff_a = 1.0 / (2.0 * alpha)
            sum_phi = math.asinh(math.pi ** 2 * ABS_BETA_2 * BANDWIDTH ** 2 / (4.0 * alpha))

            for i in range(1, n_neighbors + 1):
                delta_f = i * BANDWIDTH * 1.5
                phi_xpm = (
                    math.asinh(math.pi ** 2 * ABS_BETA_2 * l_eff_a * BANDWIDTH * (delta_f + BANDWIDTH / 2))
                    - math.asinh(math.pi ** 2 * ABS_BETA_2 * l_eff_a * BANDWIDTH * (delta_f - BANDWIDTH / 2))
                ) - (1.0 * (BANDWIDTH / abs(delta_f)) * (5.0 / 3.0) * (l_eff / L_m))
                sum_phi += phi_xpm

            p_nli = nli_prefactor * l_eff * sum_phi

            ase_norm = p_ase / LAUNCH_POWER_W
            nli_norm = p_nli / LAUNCH_POWER_W
            acc_ase += ase_norm
            acc_nli += nli_norm
            ase_per_span.append(ase_norm)
            nli_per_span.append(nli_norm)

    acc_gsnr = acc_ase + acc_nli
    osnr_db = 10 * math.log10(1.0 / acc_gsnr) if acc_gsnr > 0 else 100.0
    ase_db = 10 * math.log10(1.0 / acc_ase) if acc_ase > 0 else 100.0
    nli_db = 10 * math.log10(1.0 / acc_nli) if acc_nli > 0 else 100.0
    nli_share = acc_nli / acc_gsnr * 100 if acc_gsnr > 0 else 0.0

    return {
        "osnr_db": osnr_db,
        "ase_db": ase_db,
        "nli_db": nli_db,
        "nli_share": nli_share,
        "acc_ase": acc_ase,
        "acc_nli": acc_nli,
        "ase_per_span": ase_per_span,
        "nli_per_span": nli_per_span,
    }


def get_unique_paths(topology: TopologyModel):
    """Retorna todos os caminhos únicos com os dados de ruído."""
    seen = set()
    results = []
    for path in topology.paths:
        key = (path.node_names, path.length_km)
        if key in seen:
            continue
        seen.add(key)
        noise = calculate_path_noise(topology, path)
        results.append({
            "path": path,
            "length_km": path.length_km,
            "hops": path.hops,
            **noise,
        })
    return results


# ── Gráfico 1: OSNR vs Comprimento ──────────────────────────────────
def plot_osnr_vs_length(path_data, modulations):
    fig, ax = plt.subplots(figsize=(10, 6))

    lengths = [d["length_km"] for d in path_data]
    osnrs = [d["osnr_db"] for d in path_data]
    ase_only = [d["ase_db"] for d in path_data]

    # Ordenar por comprimento
    sort_idx = np.argsort(lengths)
    lengths_s = np.array(lengths)[sort_idx]
    osnrs_s = np.array(osnrs)[sort_idx]
    ase_s = np.array(ase_only)[sort_idx]

    ax.scatter(lengths, osnrs, c=COLORS["osnr"], s=12, alpha=0.5, zorder=5, label="OSNR (ASE+NLI)")
    ax.scatter(lengths, ase_only, c=COLORS["ase"], s=12, alpha=0.3, zorder=4, label="Apenas ASE")

    # Tendência
    z_osnr = np.polyfit(np.log10(lengths_s), osnrs_s, 3)
    z_ase = np.polyfit(np.log10(lengths_s), ase_s, 3)
    x_smooth = np.logspace(np.log10(min(lengths)), np.log10(max(lengths)), 200)
    ax.plot(x_smooth, np.polyval(z_osnr, np.log10(x_smooth)), color=COLORS["osnr"], linewidth=2, zorder=6)
    ax.plot(x_smooth, np.polyval(z_ase, np.log10(x_smooth)), color=COLORS["ase"], linewidth=1.5,
            linestyle="--", zorder=5)

    # Degradação NLI (área entre curvas)
    ax.fill_between(
        x_smooth,
        np.polyval(z_ase, np.log10(x_smooth)),
        np.polyval(z_osnr, np.log10(x_smooth)),
        alpha=0.15, color=COLORS["nli"], label="Penalidade NLI",
    )

    # Linhas de threshold das modulações
    for mod_name, color in MOD_COLORS.items():
        mod = MODULATION_CATALOG[mod_name]
        ax.axhline(y=mod.minimum_osnr, color=color, linestyle=":", linewidth=1.2,
                    alpha=0.8, label=f"{mod_name} mín. ({mod.minimum_osnr} dB)")

    ax.set_xscale("log")
    ax.set_xlabel("Comprimento do caminho (km)")
    ax.set_ylabel("OSNR (dB)")
    ax.set_title(f"OSNR vs Comprimento do Caminho — {TOPOLOGY_ID}")
    ax.legend(loc="upper right", ncol=2, framealpha=0.9)
    ax.set_ylim(bottom=8)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    fig.savefig(OUTPUT_DIR / "01_osnr_vs_comprimento.png")
    plt.close(fig)
    print(f"  ✓ 01_osnr_vs_comprimento.png")


# ── Gráfico 2: Decomposição ASE vs NLI (barras empilhadas) ──────────
def plot_noise_decomposition(path_data):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Agrupar por faixas de comprimento
    bins = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000]
    bin_labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)]
    bin_ase = [[] for _ in range(len(bins)-1)]
    bin_nli = [[] for _ in range(len(bins)-1)]
    bin_share = [[] for _ in range(len(bins)-1)]

    for d in path_data:
        for i in range(len(bins)-1):
            if bins[i] <= d["length_km"] < bins[i+1]:
                bin_ase[i].append(d["acc_ase"])
                bin_nli[i].append(d["acc_nli"])
                bin_share[i].append(d["nli_share"])
                break

    mean_ase = [np.mean(b) if b else 0 for b in bin_ase]
    mean_nli = [np.mean(b) if b else 0 for b in bin_nli]
    mean_share = [np.mean(b) if b else 0 for b in bin_share]
    counts = [len(b) for b in bin_ase]

    # Filtrar bins vazios
    valid = [i for i, c in enumerate(counts) if c > 0]
    labels_v = [bin_labels[i] for i in valid]
    ase_v = [mean_ase[i] for i in valid]
    nli_v = [mean_nli[i] for i in valid]
    share_v = [mean_share[i] for i in valid]
    counts_v = [counts[i] for i in valid]

    x = np.arange(len(labels_v))
    width = 0.6

    # Barras empilhadas (escala log)
    ax1.bar(x, ase_v, width, label="ASE", color=COLORS["ase"], alpha=0.85)
    ax1.bar(x, nli_v, width, bottom=ase_v, label="NLI", color=COLORS["nli"], alpha=0.85)
    ax1.set_yscale("log")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels_v, rotation=45, ha="right")
    ax1.set_xlabel("Faixa de comprimento (km)")
    ax1.set_ylabel("Ruído normalizado (escala log)")
    ax1.set_title("Decomposição do Ruído por Faixa de Comprimento")
    ax1.legend()

    # Contagem de caminhos em cada bin
    for i, c in enumerate(counts_v):
        total = ase_v[i] + nli_v[i]
        ax1.text(i, total * 1.1, f"n={c}", ha="center", fontsize=8, color="#555")

    # Proporção NLI
    bars = ax2.bar(x, share_v, width, color=COLORS["nli"], alpha=0.75)
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels_v, rotation=45, ha="right")
    ax2.set_xlabel("Faixa de comprimento (km)")
    ax2.set_ylabel("Proporção NLI / Ruído Total (%)")
    ax2.set_title("Contribuição Relativa da NLI por Faixa")
    ax2.set_ylim(0, 35)
    ax2.axhline(y=20, color=COLORS["margin"], linestyle="--", alpha=0.5, label="Ref. 20%")
    ax2.legend()

    for bar, val in zip(bars, share_v):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{val:.1f}%", ha="center", fontsize=9, fontweight="bold")

    fig.suptitle(f"Análise de Ruído — {TOPOLOGY_ID}", fontsize=14, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "02_decomposicao_ruido.png")
    plt.close(fig)
    print(f"  ✓ 02_decomposicao_ruido.png")


# ── Gráfico 3: Zonas de aceitação com margens ───────────────────────
def plot_acceptance_zones(path_data, modulations):
    fig, ax = plt.subplots(figsize=(11, 6.5))

    lengths = np.array([d["length_km"] for d in path_data])
    osnrs = np.array([d["osnr_db"] for d in path_data])

    sort_idx = np.argsort(lengths)
    lengths_s = lengths[sort_idx]
    osnrs_s = osnrs[sort_idx]

    z = np.polyfit(np.log10(lengths_s), osnrs_s, 3)
    x_smooth = np.logspace(np.log10(min(lengths)), np.log10(max(lengths)), 300)
    osnr_trend = np.polyval(z, np.log10(x_smooth))

    # Curva OSNR
    ax.plot(x_smooth, osnr_trend, color=COLORS["osnr"], linewidth=2.5, label="OSNR (canal isolado)", zorder=10)
    ax.scatter(lengths, osnrs, c=COLORS["osnr"], s=8, alpha=0.3, zorder=5)

    # Threshold de cada modulação com margens
    margins = [0.0, 0.5, 1.0, 2.0, 3.0]
    for mod_name, color in list(MOD_COLORS.items())[:3]:  # QPSK, 8QAM, 16QAM
        mod = MODULATION_CATALOG[mod_name]
        for margin in margins:
            threshold = mod.minimum_osnr + margin
            alpha = 1.0 if margin == 0 else 0.4
            lw = 1.8 if margin == 0 else 0.8
            ls = "-" if margin == 0 else "--"
            lbl = f"{mod_name} ({mod.minimum_osnr} dB)" if margin == 0 else (
                f"{mod_name} + {margin} dB" if margin == margins[-1] else None
            )
            ax.axhline(y=threshold, color=color, linestyle=ls, linewidth=lw,
                       alpha=alpha, label=lbl)

    # Sombreamento da zona de aceitação QPSK
    qpsk_min = MODULATION_CATALOG["QPSK"].minimum_osnr
    ax.fill_between(x_smooth, qpsk_min, osnr_trend,
                     where=osnr_trend >= qpsk_min,
                     alpha=0.1, color=COLORS["accept"], label="Zona QPSK aceita (m=0)")

    # Encontrar comprimento máximo para QPSK com m=0
    cross_idx = np.where(osnr_trend < qpsk_min)[0]
    if len(cross_idx) > 0:
        max_length_qpsk = x_smooth[cross_idx[0]]
        ax.axvline(x=max_length_qpsk, color=COLORS["qpsk"], linestyle="-.", alpha=0.6, linewidth=1)
        ax.annotate(
            f"Limite QPSK\n≈{max_length_qpsk:.0f} km",
            xy=(max_length_qpsk, qpsk_min), xytext=(max_length_qpsk * 0.5, qpsk_min + 3),
            arrowprops=dict(arrowstyle="->", color=COLORS["qpsk"], lw=1.5),
            fontsize=10, color=COLORS["qpsk"], fontweight="bold",
        )

    ax.set_xscale("log")
    ax.set_xlabel("Comprimento do caminho (km)")
    ax.set_ylabel("OSNR / Threshold (dB)")
    ax.set_title(f"Zonas de Aceitação por Modulação e Margem — {TOPOLOGY_ID}")
    ax.legend(loc="upper right", ncol=2, framealpha=0.9, fontsize=8)
    ax.set_ylim(8, 28)
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())

    fig.savefig(OUTPUT_DIR / "03_zonas_aceitacao.png")
    plt.close(fig)
    print(f"  ✓ 03_zonas_aceitacao.png")


# ── Gráfico 4: Degradação por XPM (carga) ───────────────────────────
def plot_xpm_degradation(topology, longest_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

    neighbor_counts = list(range(0, 61, 2))
    osnrs = []
    nli_shares = []
    ase_dbs = []
    nli_dbs = []

    for n in neighbor_counts:
        result = calculate_path_noise(topology, longest_path, n_neighbors=n)
        osnrs.append(result["osnr_db"])
        nli_shares.append(result["nli_share"])
        ase_dbs.append(result["ase_db"])
        nli_dbs.append(result["nli_db"])

    # Subplot 1: OSNR vs vizinhos
    # Usar apenas contagens > 0 para log scale
    nc_plot = [n if n > 0 else 0.5 for n in neighbor_counts]
    ax1.plot(nc_plot, osnrs, color=COLORS["osnr"], linewidth=2.5, marker="o",
             markersize=4, label="OSNR (GSNR)")
    ax1.axhline(y=MODULATION_CATALOG["QPSK"].minimum_osnr, color=COLORS["qpsk"],
                linestyle="--", linewidth=1.5, label=f"QPSK mín. (12.6 dB)")

    # Margens
    for margin, alpha in [(0.5, 0.5), (1.0, 0.4), (2.0, 0.3)]:
        threshold = MODULATION_CATALOG["QPSK"].minimum_osnr + margin
        ax1.axhline(y=threshold, color=COLORS["qpsk"], linestyle=":",
                     linewidth=1, alpha=alpha, label=f"QPSK + {margin} dB")

    # Ponto de cruzamento com QPSK
    qpsk_min = MODULATION_CATALOG["QPSK"].minimum_osnr
    for i in range(1, len(osnrs)):
        if osnrs[i] < qpsk_min and osnrs[i-1] >= qpsk_min:
            cross_n = neighbor_counts[i-1] + (neighbor_counts[i] - neighbor_counts[i-1]) * (
                (osnrs[i-1] - qpsk_min) / (osnrs[i-1] - osnrs[i])
            )
            ax1.axvline(x=cross_n, color=COLORS["block"], linestyle="-.", alpha=0.5)
            ax1.annotate(
                f"Bloqueio QPSK\n≈{cross_n:.0f} vizinhos",
                xy=(cross_n, qpsk_min), xytext=(cross_n + 10, qpsk_min + 0.5),
                arrowprops=dict(arrowstyle="->", color=COLORS["block"], lw=1.5),
                fontsize=9, color=COLORS["block"], fontweight="bold",
            )
            break

    ax1.set_xscale("log")
    ax1.set_xlabel("Número de canais vizinhos no enlace")
    ax1.set_ylabel("OSNR (dB)")
    ax1.set_title("Degradação do OSNR por XPM")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.set_ylim(bottom=11)

    # Subplot 2: Proporção NLI
    ax2.fill_between(nc_plot, 0, nli_shares, color=COLORS["nli"], alpha=0.3)
    ax2.plot(nc_plot, nli_shares, color=COLORS["nli"], linewidth=2.5, marker="s",
             markersize=3, label="NLI / Ruído Total")
    ax2.set_xscale("log")
    ax2.set_xlabel("Número de canais vizinhos no enlace")
    ax2.set_ylabel("Proporção NLI (%)")
    ax2.set_title("Crescimento da Proporção de NLI")
    ax2.set_ylim(0, 50)
    ax2.legend()

    path_label = " → ".join(longest_path.node_names)
    fig.suptitle(
        f"Efeito da Carga (XPM) — Caminho: {path_label}\n"
        f"({longest_path.length_km:.0f} km, {longest_path.hops} hops)",
        fontsize=12, fontweight="bold", y=1.04,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "04_degradacao_xpm.png")
    plt.close(fig)
    print(f"  ✓ 04_degradacao_xpm.png")


# ── Gráfico 5: Mapa de viabilidade por par OD ───────────────────────
def plot_viability_heatmap(topology, path_data):
    margins = [0.0, 0.5, 1.0, 2.0, 3.0]
    n_margins = len(margins)
    fig, axes = plt.subplots(1, n_margins, figsize=(4 * n_margins, 4.5),
                              sharey=True)

    node_names = list(topology.node_names)
    n_nodes = len(node_names)
    qpsk_min = MODULATION_CATALOG["QPSK"].minimum_osnr

    cmap = LinearSegmentedColormap.from_list("viability", [COLORS["block"], "#ffeaa7", COLORS["accept"]])

    for ax_idx, margin in enumerate(margins):
        ax = axes[ax_idx]
        threshold = qpsk_min + margin
        matrix = np.full((n_nodes, n_nodes), np.nan)

        for d in path_data:
            path = d["path"]
            src_name = path.node_names[0]
            dst_name = path.node_names[-1]
            src_idx = node_names.index(src_name)
            dst_idx = node_names.index(dst_name)

            osnr_margin_val = d["osnr_db"] - threshold

            # Guardar o melhor k-caminho (maior OSNR margin)
            if np.isnan(matrix[src_idx, dst_idx]) or osnr_margin_val > matrix[src_idx, dst_idx]:
                matrix[src_idx, dst_idx] = osnr_margin_val
                matrix[dst_idx, src_idx] = osnr_margin_val

        im = ax.imshow(matrix, cmap=cmap, vmin=-10, vmax=10, aspect="equal")
        ax.set_title(f"Margem = {margin} dB", fontsize=11, fontweight="bold")
        ax.set_xticks(range(n_nodes))
        ax.set_xticklabels([n[:3] for n in node_names], rotation=90, fontsize=7)
        if ax_idx == 0:
            ax.set_yticks(range(n_nodes))
            ax.set_yticklabels([n[:3] for n in node_names], fontsize=7)
        else:
            ax.set_yticks([])

        # Contagem de pares viáveis
        viable = np.nansum(matrix > 0) / 2
        total = n_nodes * (n_nodes - 1) / 2
        ax.set_xlabel(f"{int(viable)}/{int(total)} pares viáveis", fontsize=9)

    fig.suptitle(
        f"Viabilidade QoT por Par Origem-Destino (QPSK) — {TOPOLOGY_ID}\n"
        "Verde = folga positiva • Vermelho = OSNR insuficiente",
        fontsize=12, fontweight="bold",
    )
    cbar = fig.colorbar(im, ax=axes, shrink=0.8, label="Margem de OSNR (dB)")
    fig.tight_layout(rect=[0, 0, 0.92, 0.92])
    fig.savefig(OUTPUT_DIR / "05_mapa_viabilidade.png")
    plt.close(fig)
    print(f"  ✓ 05_mapa_viabilidade.png")


# ── Gráfico 6: Perfil span-a-span do pior caminho ───────────────────
def plot_span_profile(topology, longest_path):
    noise = calculate_path_noise(topology, longest_path)
    ase_spans = np.array(noise["ase_per_span"])
    nli_spans = np.array(noise["nli_per_span"])
    n_spans = len(ase_spans)

    # ASE e NLI acumulados
    cum_ase = np.cumsum(ase_spans)
    cum_nli = np.cumsum(nli_spans)
    cum_gsnr = cum_ase + cum_nli
    cum_osnr = 10 * np.log10(1.0 / cum_gsnr)
    cum_ase_only = 10 * np.log10(1.0 / cum_ase)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    span_ids = np.arange(1, n_spans + 1)

    # Top: Ruído por span (escala log)
    ax1.bar(span_ids - 0.15, ase_spans * 1e3, 0.3, color=COLORS["ase"], alpha=0.8, label="ASE")
    ax1.bar(span_ids + 0.15, nli_spans * 1e3, 0.3, color=COLORS["nli"], alpha=0.8, label="NLI")
    ax1.set_yscale("log")
    ax1.set_ylabel("Ruído por Span (×10⁻³, escala log)")
    ax1.set_title("Contribuição Individual de Cada Span")
    ax1.legend()

    # Marcar separação de links
    span_cursor = 0
    for link_id in longest_path.link_ids:
        link = topology.links[link_id]
        span_cursor += len(link.spans)
        if span_cursor < n_spans:
            ax1.axvline(x=span_cursor + 0.5, color="#bdc3c7", linestyle="-", linewidth=0.8, alpha=0.5)
            ax2.axvline(x=span_cursor + 0.5, color="#bdc3c7", linestyle="-", linewidth=0.8, alpha=0.5)

    # Bottom: OSNR acumulado
    ax2.plot(span_ids, cum_osnr, color=COLORS["osnr"], linewidth=2.5, marker="o",
             markersize=3, label="OSNR acumulado (ASE+NLI)")
    ax2.plot(span_ids, cum_ase_only, color=COLORS["ase"], linewidth=1.5, linestyle="--",
             marker="s", markersize=2, label="OSNR apenas ASE")
    ax2.fill_between(span_ids, cum_ase_only, cum_osnr, alpha=0.15, color=COLORS["nli"],
                      label="Penalidade NLI")

    ax2.axhline(y=MODULATION_CATALOG["QPSK"].minimum_osnr, color=COLORS["qpsk"],
                linestyle="--", linewidth=1.5, label="QPSK mín. (12.6 dB)")

    ax2.set_xlabel("Span (amplificador)")
    ax2.set_ylabel("OSNR Acumulado (dB)")
    ax2.set_title("Evolução do OSNR ao Longo do Caminho")
    ax2.legend(loc="upper right", fontsize=8)

    # Anotar links
    span_cursor = 0
    for i, link_id in enumerate(longest_path.link_ids):
        link = topology.links[link_id]
        mid = span_cursor + len(link.spans) / 2
        ax2.text(mid + 0.5, ax2.get_ylim()[1] - 0.5,
                 f"{link.source_name[:3]}→{link.target_name[:3]}",
                 ha="center", fontsize=7, rotation=45, alpha=0.6)
        span_cursor += len(link.spans)

    path_label = " → ".join(longest_path.node_names)
    fig.suptitle(
        f"Perfil de Ruído Span-a-Span — {path_label}\n"
        f"({longest_path.length_km:.0f} km, {n_spans} spans)",
        fontsize=12, fontweight="bold", y=1.02,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "06_perfil_span.png")
    plt.close(fig)
    print(f"  ✓ 06_perfil_span.png")


# ── Gráfico 7: Margem vs Probabilidade de Bloqueio Teórica ──────────
def plot_margin_vs_blocking_theoretical(path_data):
    """Estima a fração de caminhos bloqueados para cada valor de margem."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    margins = np.arange(0.0, 5.1, 0.1)
    modulations_to_plot = ["QPSK", "8QAM", "16QAM"]
    total_paths = len(path_data)

    for mod_name in modulations_to_plot:
        mod = MODULATION_CATALOG[mod_name]
        blocked_fractions = []
        for margin in margins:
            threshold = mod.minimum_osnr + margin
            # Para cada par OD, verificar se ALGUM k-caminho atende
            od_pairs: dict[tuple, float] = {}
            for d in path_data:
                key = (d["path"].node_names[0], d["path"].node_names[-1])
                rev_key = (key[1], key[0])
                best_key = key if key not in od_pairs else key
                if rev_key in od_pairs:
                    best_key = rev_key
                if best_key not in od_pairs or d["osnr_db"] > od_pairs[best_key]:
                    od_pairs[best_key] = d["osnr_db"]
            blocked = sum(1 for osnr in od_pairs.values() if osnr < threshold)
            frac = blocked / len(od_pairs) * 100
            blocked_fractions.append(max(frac, 0.1))  # floor para log scale

        ax.plot(margins, blocked_fractions, linewidth=2.5, label=mod_name,
                color=MOD_COLORS[mod_name], marker="", markersize=3)

    ax.set_yscale("log")
    ax.set_xlabel("Margem de OSNR (dB)")
    ax.set_ylabel("Pares OD Bloqueados (%)")
    ax.set_title(f"Impacto da Margem na Viabilidade dos Pares OD — {TOPOLOGY_ID}")
    ax.legend(fontsize=10)
    ax.set_xlim(0, 5)
    ax.set_ylim(0.1, 110)
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_major_locator(mticker.FixedLocator([0.1, 0.5, 1, 2, 5, 10, 20, 50, 100]))
    ax.set_yticklabels(["0.1%", "0.5%", "1%", "2%", "5%", "10%", "20%", "50%", "100%"])

    # Anotar pontos críticos
    for mod_name in modulations_to_plot:
        mod = MODULATION_CATALOG[mod_name]
        od_pairs_best: dict[tuple, float] = {}
        for d in path_data:
            key = (d["path"].node_names[0], d["path"].node_names[-1])
            rev_key = (key[1], key[0])
            best_key = key if key not in od_pairs_best else key
            if rev_key in od_pairs_best:
                best_key = rev_key
            if best_key not in od_pairs_best or d["osnr_db"] > od_pairs_best[best_key]:
                od_pairs_best[best_key] = d["osnr_db"]
        # Encontrar margem onde blocking começa
        min_folga = min(osnr - mod.minimum_osnr for osnr in od_pairs_best.values())
        if min_folga > 0 and min_folga < 5:
            ax.annotate(
                f"{mod_name}: bloqueio\ncomeça em {min_folga:.1f} dB",
                xy=(min_folga, 0.2), xytext=(min_folga + 0.8, 3),
                arrowprops=dict(arrowstyle="->", color=MOD_COLORS[mod_name]),
                fontsize=8, color=MOD_COLORS[mod_name],
            )

    fig.savefig(OUTPUT_DIR / "07_margem_vs_bloqueio.png")
    plt.close(fig)
    print(f"  ✓ 07_margem_vs_bloqueio.png")


# ── Main ─────────────────────────────────────────────────────────────
def main():
    print(f"Carregando topologia {TOPOLOGY_ID}...")
    topology = TopologyModel.from_file(
        resolve_topology(TOPOLOGY_ID),
        topology_id=TOPOLOGY_ID,
        k_paths=DEFAULT_K_PATHS,
    )
    print(f"  Nós: {topology.node_count}")
    print(f"  Links: {topology.link_count}")
    print(f"  Caminhos: {topology.path_count}")
    print()

    modulations = get_modulations("QPSK,8QAM,16QAM,32QAM,64QAM")

    print("Calculando ruído para todos os caminhos...")
    path_data = get_unique_paths(topology)
    print(f"  Caminhos únicos analisados: {len(path_data)}")
    print()

    longest_path = max(topology.paths, key=lambda p: p.length_km)
    print(f"Caminho mais longo: {' → '.join(longest_path.node_names)} ({longest_path.length_km:.0f} km)")
    print()

    print(f"Gerando gráficos em {OUTPUT_DIR}/")
    plot_osnr_vs_length(path_data, modulations)
    plot_noise_decomposition(path_data)
    plot_acceptance_zones(path_data, modulations)
    plot_xpm_degradation(topology, longest_path)
    plot_viability_heatmap(topology, path_data)
    plot_span_profile(topology, longest_path)
    plot_margin_vs_blocking_theoretical(path_data)

    print()
    print(f"✅ Todos os gráficos salvos em: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
