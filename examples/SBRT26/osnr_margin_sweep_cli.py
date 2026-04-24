"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          SBRT26 · OSNR Margin Sweep · Interface de Linha de Comando         ║
╚══════════════════════════════════════════════════════════════════════════════╝

Versão interativa do experimento de varredura de margem OSNR.
A lógica de simulação é IDÊNTICA ao script original osnr_margin_sweep.py.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

# ── Verificação antecipada das dependências visuais ──────────────────────────
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.progress import (
        Progress,
        SpinnerColumn,
        BarColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
        MofNCompleteColumn,
        TaskProgressColumn,
    )
    from rich.prompt import Prompt, Confirm
    from rich.text import Text
    from rich.align import Align
    from rich.columns import Columns
    from rich import box
    from rich.style import Style
    from rich.rule import Rule
    from rich.live import Live
    from rich.layout import Layout
    from rich.padding import Padding
except ImportError:
    print("❌  Dependência 'rich' não encontrada. Execute:  pip install rich")
    sys.exit(1)

# ── Importações do simulador (mesmas do original) ────────────────────────────
from optical_networking_gym_v2 import ScenarioConfig, make_env
from optical_networking_gym_v2.defaults import (
    DEFAULT_K_PATHS,
    DEFAULT_LAUNCH_POWER_DBM,
    DEFAULT_LOAD,
    DEFAULT_MEAN_HOLDING_TIME,
    DEFAULT_MODULATIONS_TO_CONSIDER,
    DEFAULT_NUM_SPECTRUM_RESOURCES,
    DEFAULT_SEED,
)
from optical_networking_gym_v2.utils import experiment_scenarios as scenario_utils
from optical_networking_gym_v2.utils import experiment_utils as sweep_utils
from optical_networking_gym_v2.utils import sweep_reporting as report_utils

# ═══════════════════════════════════════════════════════════════════════════════
#  Constantes (idênticas ao original)
# ═══════════════════════════════════════════════════════════════════════════════

SCRIPT_DIR = Path(__file__).resolve().parent

DEFAULT_POLICY_NAME       = "first_fit"
DEFAULT_MARGINS           = (0.0, 1.0, 2.0, 3.0)
DEFAULT_EPISODES_PER_MARGIN = 6
DEFAULT_EPISODE_LENGTH    = 10000

MODULATION_INDEX_TO_NAME = sweep_utils.build_modulation_index_to_name(
    sweep_utils.DEFAULT_MODULATION_NAMES
)

EPISODE_BASE_FIELDS = [
    "date", "policy", "topology_id", "margin", "episode_index",
    "episode_seed", "episodes_per_margin", "requests_per_episode",
    "seed_base", "load", "mean_holding_time", "num_spectrum_resources",
    "k_paths", "launch_power_dbm", "measure_disruptions",
]
EPISODE_METRIC_FIELDS = [
    "services_processed", "services_accepted", "services_served",
    "service_blocking_rate", "service_served_rate", "bit_rate_blocking_rate",
    "blocked_due_to_resources", "blocked_due_to_osnr", "rejected",
    "episode_disrupted_services_count", "episode_disrupted_services_rate",
    "disrupted_or_dropped_services", "mean_osnr_accepted", "mean_osnr_final",
    "episode_time_s", *MODULATION_INDEX_TO_NAME.values(),
]
EPISODE_FIELDNAMES = EPISODE_BASE_FIELDS + EPISODE_METRIC_FIELDS

SUMMARY_BASE_FIELDS = [
    "date", "policy", "topology_id", "margin", "episodes",
    "requests_per_episode", "seed_base", "load", "mean_holding_time",
    "num_spectrum_resources", "k_paths", "launch_power_dbm", "measure_disruptions",
]
SUMMARY_METRIC_NAMES = [
    "services_accepted", "services_served", "service_blocking_rate",
    "service_served_rate", "bit_rate_blocking_rate", "mean_osnr_accepted",
    "mean_osnr_final", "episode_disrupted_services_count",
    "episode_disrupted_services_rate", "disrupted_or_dropped_services",
]
SUMMARY_FIELDNAMES = report_utils.build_summary_fieldnames(
    base_fields=SUMMARY_BASE_FIELDS,
    metric_names=SUMMARY_METRIC_NAMES,
)

# ═══════════════════════════════════════════════════════════════════════════════
#  Dataclasses (idênticos ao original)
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class MarginSweepExperiment:
    topology_id: str              = "nobel-eu"
    policy_name: str              = DEFAULT_POLICY_NAME
    margins: tuple[float, ...]    = DEFAULT_MARGINS
    episodes_per_margin: int      = DEFAULT_EPISODES_PER_MARGIN
    episode_length: int           = DEFAULT_EPISODE_LENGTH
    seed: int                     = DEFAULT_SEED
    load: float                   = DEFAULT_LOAD
    mean_holding_time: float      = DEFAULT_MEAN_HOLDING_TIME
    num_spectrum_resources: int   = DEFAULT_NUM_SPECTRUM_RESOURCES
    k_paths: int                  = DEFAULT_K_PATHS
    launch_power_dbm: float       = DEFAULT_LAUNCH_POWER_DBM
    modulations_to_consider: int  = DEFAULT_MODULATIONS_TO_CONSIDER
    measure_disruptions: bool     = True
    drop_on_disruption: bool      = True
    output_dir: Path              = SCRIPT_DIR / "resultados"

    def __post_init__(self) -> None:
        object.__setattr__(self, "margins", tuple(float(m) for m in self.margins))
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if not self.topology_id:
            raise ValueError("topology_id deve ser uma string não-vazia")
        if self.policy_name != DEFAULT_POLICY_NAME:
            raise ValueError(f"política {self.policy_name!r} não suportada; use: {DEFAULT_POLICY_NAME}")
        if not self.margins:
            raise ValueError("margins deve ser uma sequência não-vazia")
        if self.episodes_per_margin <= 0:
            raise ValueError("episodes_per_margin deve ser positivo")
        if self.episode_length <= 0:
            raise ValueError("episode_length deve ser positivo")
        if self.seed < 0:
            raise ValueError("seed deve ser não-negativo")
        if self.load <= 0:
            raise ValueError("load deve ser positivo")
        if self.mean_holding_time <= 0:
            raise ValueError("mean_holding_time deve ser positivo")
        if self.num_spectrum_resources <= 0:
            raise ValueError("num_spectrum_resources deve ser positivo")
        if self.k_paths <= 0:
            raise ValueError("k_paths deve ser positivo")
        if self.modulations_to_consider <= 0:
            raise ValueError("modulations_to_consider deve ser positivo")


@dataclass(frozen=True, slots=True)
class MarginSweepOutputs:
    episodes_csv: Path
    summary_csv: Path


# ═══════════════════════════════════════════════════════════════════════════════
#  Funções de simulação (lógica IDÊNTICA ao original)
# ═══════════════════════════════════════════════════════════════════════════════

def build_base_scenario(experiment: MarginSweepExperiment) -> ScenarioConfig:
    return scenario_utils.build_nobel_eu_graph_load_scenario(
        topology_id=experiment.topology_id,
        episode_length=experiment.episode_length,
        seed=experiment.seed,
        load=experiment.load,
        mean_holding_time=experiment.mean_holding_time,
        num_spectrum_resources=experiment.num_spectrum_resources,
        k_paths=experiment.k_paths,
        launch_power_dbm=experiment.launch_power_dbm,
        modulations_to_consider=experiment.modulations_to_consider,
        measure_disruptions=experiment.measure_disruptions,
        drop_on_disruption=experiment.drop_on_disruption,
    )


def build_episode_scenario(
    *,
    experiment: MarginSweepExperiment,
    base_scenario: ScenarioConfig,
    margin: float,
    episode_index: int,
) -> ScenarioConfig:
    episode_seed = experiment.seed + episode_index
    return replace(
        base_scenario,
        scenario_id=f"{experiment.topology_id}_margin_{margin:g}_seed{episode_seed}",
        seed=episode_seed,
        margin=float(margin),
    )


def build_env(*, scenario: ScenarioConfig):
    return make_env(config=scenario)


def run_single_episode(
    *,
    experiment: MarginSweepExperiment,
    base_scenario: ScenarioConfig,
    margin: float,
    episode_index: int,
    date_label: str,
    step_callback=None,          # callable() | None — chamado a cada env.step()
) -> dict[str, report_utils.Scalar]:
    if experiment.policy_name != DEFAULT_POLICY_NAME:
        raise ValueError(f"política não suportada: {experiment.policy_name!r}")

    episode_scenario = build_episode_scenario(
        experiment=experiment,
        base_scenario=base_scenario,
        margin=margin,
        episode_index=episode_index,
    )
    env = build_env(scenario=episode_scenario)
    episode_seed = int(episode_scenario.seed or 0)
    _, info = env.reset(seed=episode_seed)

    accepted_osnrs: list[float] = []
    started_at = time.perf_counter()
    while True:
        action = sweep_utils.select_informative_first_fit_policy(env, info)
        _, _, terminated, truncated, info = env.step(action)
        if step_callback is not None:          # ← avança barra de requisições
            step_callback()
        if str(info.get("status", "")) == "accepted":
            accepted_osnrs.append(float(info.get("osnr", 0.0)))
        if terminated or truncated:
            break
    episode_time_s = time.perf_counter() - started_at

    simulator = env.simulator
    if simulator.statistics is None:
        raise RuntimeError("estatísticas do simulador não disponíveis após o episódio")
    snapshot = simulator.statistics.snapshot()
    active_services = (
        () if simulator.state is None
        else simulator.state.active_services_by_id.values()
    )
    final_osnrs = [float(service.osnr) for service in active_services]

    row: dict[str, report_utils.Scalar] = {
        "date":                             date_label,
        "policy":                           experiment.policy_name,
        "topology_id":                      episode_scenario.topology_id,
        "margin":                           float(margin),
        "episode_index":                    int(episode_index),
        "episode_seed":                     episode_seed,
        "episodes_per_margin":              experiment.episodes_per_margin,
        "requests_per_episode":             int(episode_scenario.episode_length),
        "seed_base":                        int(experiment.seed),
        "load":                             float(episode_scenario.load),
        "mean_holding_time":                float(episode_scenario.mean_holding_time),
        "num_spectrum_resources":           int(episode_scenario.num_spectrum_resources),
        "k_paths":                          int(episode_scenario.k_paths),
        "launch_power_dbm":                 float(episode_scenario.launch_power_dbm),
        "measure_disruptions":              bool(episode_scenario.measure_disruptions),
        "services_processed":               int(snapshot.episode_services_processed),
        "services_accepted":                int(snapshot.episode_services_accepted),
        "services_served":                  int(snapshot.episode_services_served),
        "service_blocking_rate":            float(snapshot.episode_service_blocking_rate),
        "service_served_rate":              float(snapshot.episode_service_served_rate),
        "bit_rate_blocking_rate":           float(snapshot.episode_bit_rate_blocking_rate),
        "blocked_due_to_resources":         int(snapshot.episode_services_blocked_resources),
        "blocked_due_to_osnr":              int(snapshot.episode_services_blocked_qot),
        "rejected":                         int(snapshot.episode_services_rejected_by_agent),
        "episode_disrupted_services_count": int(snapshot.episode_disrupted_services),
        "episode_disrupted_services_rate":  float(snapshot.episode_disrupted_services_rate),
        "disrupted_or_dropped_services":    int(snapshot.episode_services_dropped_qot),
        "mean_osnr_accepted":               sweep_utils.float_mean(accepted_osnrs),
        "mean_osnr_final":                  sweep_utils.float_mean(final_osnrs),
        "episode_time_s":                   float(episode_time_s),
    }
    row.update(
        sweep_utils.episode_modulation_counts(
            snapshot,
            modulation_index_to_name=MODULATION_INDEX_TO_NAME,
        )
    )
    return row


def _build_summary_row(
    *,
    experiment: MarginSweepExperiment,
    base_scenario: ScenarioConfig,
    margin: float,
    date_label: str,
    episode_rows: list[dict[str, report_utils.Scalar]],
) -> dict[str, report_utils.Scalar]:
    row: dict[str, report_utils.Scalar] = {
        "date":                   date_label,
        "policy":                 experiment.policy_name,
        "topology_id":            base_scenario.topology_id,
        "margin":                 float(margin),
        "episodes":               int(experiment.episodes_per_margin),
        "requests_per_episode":   int(experiment.episode_length),
        "seed_base":              int(experiment.seed),
        "load":                   float(base_scenario.load),
        "mean_holding_time":      float(base_scenario.mean_holding_time),
        "num_spectrum_resources": int(base_scenario.num_spectrum_resources),
        "k_paths":                int(base_scenario.k_paths),
        "launch_power_dbm":       float(base_scenario.launch_power_dbm),
        "measure_disruptions":    bool(base_scenario.measure_disruptions),
    }
    row.update(
        report_utils.aggregate_summary_metrics(
            episode_rows,
            metric_names=SUMMARY_METRIC_NAMES,
        )
    )
    return row


def run_margin_sweep(
    experiment: MarginSweepExperiment | None = None,
    *,
    now: datetime | None = None,
    console: Console | None = None,
) -> MarginSweepOutputs:
    """Executa a varredura de margem com barra de progresso visual."""
    resolved_experiment = MarginSweepExperiment() if experiment is None else experiment
    base_scenario       = build_base_scenario(resolved_experiment)
    date_label          = report_utils.date_prefix(now)
    con                 = console or Console()

    total_margins    = len(resolved_experiment.margins)
    total_episodes   = total_margins * resolved_experiment.episodes_per_margin
    # Cada episódio processa aproximadamente episode_length requisições
    total_requests   = total_episodes * resolved_experiment.episode_length

    episode_rows: list[dict[str, report_utils.Scalar]] = []
    summary_rows: list[dict[str, report_utils.Scalar]] = []

    con.print()
    con.print(Rule("[bold cyan]▶  Iniciando varredura[/bold cyan]", style="cyan"))
    con.print()

    with Progress(
        SpinnerColumn(spinner_name="dots12", style="bold magenta"),
        TextColumn("[bold white]{task.description}"),
        BarColumn(
            bar_width=36,
            style="dark_violet",
            complete_style="bright_magenta",
            finished_style="bright_green",
            pulse_style="bold magenta",
        ),
        TaskProgressColumn(style="bright_white"),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TextColumn("⏳"),
        TimeRemainingColumn(),
        console=con,
        transient=False,
        expand=True,
    ) as progress:

        margin_task = progress.add_task(
            "[cyan]Margem[/cyan]",
            total=total_margins,
        )
        episode_task = progress.add_task(
            "[magenta]Episódio[/magenta]",
            total=total_episodes,
        )
        request_task = progress.add_task(
            "[bright_blue]Requisições[/bright_blue]",
            total=total_requests,
        )

        for m_idx, margin in enumerate(resolved_experiment.margins, start=1):
            progress.update(
                margin_task,
                description=(
                    f"[cyan]Margem [bold]{margin:+.1f} dB[/bold]"
                    f" ({m_idx}/{total_margins})[/cyan]"
                ),
            )

            margin_episode_rows: list[dict[str, report_utils.Scalar]] = []
            for ep_idx in range(resolved_experiment.episodes_per_margin):
                ep_num = ep_idx + 1
                progress.update(
                    episode_task,
                    description=(
                        f"[magenta]Ep [bold]{ep_num}/{resolved_experiment.episodes_per_margin}[/bold]"
                        f" · margem [bold]{margin:+.1f} dB[/bold][/magenta]"
                    ),
                )
                # Reinicia barra de requisições para este episódio
                progress.reset(
                    request_task,
                    total=resolved_experiment.episode_length,
                    description=f"[bright_blue]Requisições  ep {ep_num}[/bright_blue]",
                    completed=0,
                    visible=True,
                )

                def _step_cb(_p=progress, _t=request_task):
                    _p.advance(_t)

                row = run_single_episode(
                    experiment=resolved_experiment,
                    base_scenario=base_scenario,
                    margin=float(margin),
                    episode_index=ep_idx,
                    date_label=date_label,
                    step_callback=_step_cb,
                )
                margin_episode_rows.append(row)

                # Garante que a barra de requisições chegue a 100%
                progress.update(request_task, completed=resolved_experiment.episode_length)

                # Mini-resumo em linha
                sbr  = row["service_blocking_rate"]
                osnr = row["mean_osnr_accepted"]
                con.print(
                    f"   [dim]↳ ep {ep_num:02d}[/dim]  "
                    f"bloqueio=[yellow]{sbr:.4f}[/yellow]  "
                    f"OSNR⟨aceit.⟩=[green]{osnr:.2f} dB[/green]  "
                    f"tempo=[grey50]{row['episode_time_s']:.2f}s[/grey50]"
                )

                progress.advance(episode_task)

            # Esconde barra de requisições entre margens
            progress.update(request_task, visible=False)

            episode_rows.extend(margin_episode_rows)
            summary_rows.append(
                _build_summary_row(
                    experiment=resolved_experiment,
                    base_scenario=base_scenario,
                    margin=float(margin),
                    date_label=date_label,
                    episode_rows=margin_episode_rows,
                )
            )
            progress.advance(margin_task)
            con.print()

    # ── Gravação dos CSVs ────────────────────────────────────────────────────
    # Nome inclui a carga: ex. 30-03-300-margin-episodes.csv
    _prefix = report_utils.date_prefix(now)
    _load   = int(resolved_experiment.load)
    episodes_csv = resolved_experiment.output_dir / f"{_prefix}-{_load}-margin-episodes.csv"
    summary_csv  = resolved_experiment.output_dir / f"{_prefix}-{_load}-margin-summary.csv"
    report_utils.write_csv_rows(
        path=episodes_csv,
        fieldnames=EPISODE_FIELDNAMES,
        rows=episode_rows,
    )
    report_utils.write_csv_rows(
        path=summary_csv,
        fieldnames=SUMMARY_FIELDNAMES,
        rows=summary_rows,
    )
    return MarginSweepOutputs(episodes_csv=episodes_csv, summary_csv=summary_csv)


# ═══════════════════════════════════════════════════════════════════════════════
#  Interface de linha de comando interativa
# ═══════════════════════════════════════════════════════════════════════════════

BANNER = r"""
   ██████╗     ███╗   ██╗     ██████╗         ██╗   ██╗██████╗
  ██╔═══██╗    ████╗  ██║    ██╔════╝         ██║   ██║╚════██╗
  ██║   ██║    ██╔██╗ ██║    ██║  ███╗        ██║   ██║ █████╔╝
  ██║   ██║    ██║╚██╗██║    ██║   ██║        ╚██╗ ██╔╝██╔═══╝
  ╚██████╔╝    ██║ ╚████║    ╚██████╔╝    ██╗  ╚████╔╝ ███████╗
   ╚═════╝     ╚═╝  ╚═══╝     ╚═════╝    ╚═╝   ╚═══╝  ╚══════╝
  ─────────── Optical Networking Gym · Version 2 ────────────
"""

MENU_ITEMS = [
    ("1", "Topologia             [--topology-id]",         "topology_id"),
    ("2", "Margens OSNR (dB)     [--margins]",             "margins"),
    ("3", "Episódios por margem  [--episodes-per-margin]", "episodes_per_margin"),
    ("4", "Requisições/episódio  [--request-count]",       "episode_length"),
    ("5", "Diretório de saída    [--output-dir]",          "output_dir"),
    ("6", "Política              [--policy]",               "policy_name"),
    ("7", "Carga da topologia    [--load]  (Erlang)",      "load"),
    ("8", "▶  Iniciar simulação"),
    ("0", "Sair"),
]


def _term_width(console: Console, fallback: int = 88) -> int:
    try:
        return max(60, console.width or fallback)
    except Exception:
        return fallback


def print_banner(console: Console) -> None:
    w = _term_width(console)
    console.print()
    console.print(Align.center(
        Panel(
            Align.center(Text(BANNER, style="bold bright_magenta")),
            subtitle="[dim cyan]SBRT 2026 · OSNR Margin Sweep · Interface Interativa[/dim cyan]",
            border_style="bright_magenta",
            padding=(0, 2),
            width=min(w, 90),
        ),
        vertical="middle",
    ))
    console.print()


def _fmt_margins(margins: tuple[float, ...]) -> str:
    return "  ".join(f"[bold cyan]{m:+.1f}[/bold cyan]" for m in margins)


def _make_config_table(cfg: dict, console: Console) -> Table:
    """Renderiza a tabela de parâmetros configurados."""
    w = _term_width(console)
    tbl = Table(
        title="[bold white]⚙  Parâmetros da Simulação[/bold white]",
        box=box.DOUBLE_EDGE,
        border_style="bright_magenta",
        header_style="bold bright_cyan",
        show_lines=True,
        expand=False,
        width=min(w - 4, 86),
    )
    tbl.add_column("#",          style="bold dim",          justify="center", width=3)
    tbl.add_column("Parâmetro",  style="bold white",        min_width=28)
    tbl.add_column("Flag CLI",   style="dim cyan",          min_width=22)
    tbl.add_column("Valor",      style="bold yellow",       min_width=20)

    rows = [
        ("1", "Topologia",              "--topology-id",           str(cfg["topology_id"])),
        ("2", "Margens OSNR (dB)",      "--margins",               "  ".join(f"{m:+.1f}" for m in cfg["margins"])),
        ("3", "Episódios / margem",     "--episodes-per-margin",   str(cfg["episodes_per_margin"])),
        ("4", "Requisições / episódio", "--request-count",         str(cfg["episode_length"])),
        ("5", "Diretório de saída",     "--output-dir",            str(cfg["output_dir"])),
        ("6", "Política",               "--policy",                str(cfg["policy_name"])),
        ("7", "Carga (Erlang)",         "--load",                  f"{cfg['load']:.1f}"),
    ]
    for num, label, flag, val in rows:
        tbl.add_row(num, label, flag, val)

    return tbl


def _make_stats_panel(cfg: dict) -> Panel:
    total = len(cfg["margins"]) * cfg["episodes_per_margin"]
    reqs  = total * cfg["episode_length"]
    txt = (
        f"Total de episódios: [bold bright_white]{total}[/bold bright_white]   "
        f"Total de requisições: [bold bright_white]{reqs:,}[/bold bright_white]"
    )
    return Panel(
        Align.center(txt),
        title="[bold]📊 Estimativa[/bold]",
        border_style="cyan",
        padding=(0, 2),
    )


def _print_menu(console: Console, cfg: dict) -> None:
    console.clear()
    print_banner(console)
    console.print(Align.center(_make_config_table(cfg, console)))
    console.print()
    console.print(Align.center(_make_stats_panel(cfg)))
    console.print()

    # Menu de opções
    menu = Table(
        box=box.MINIMAL,
        show_header=False,
        border_style="dim",
        expand=False,
    )
    menu.add_column(justify="center", style="bold bright_magenta", width=4)
    menu.add_column(style="white")

    for item in MENU_ITEMS:
        key   = item[0]
        label = item[1]
        color = "bright_green" if key == "8" else ("bright_red" if key == "0" else "bright_magenta")
        menu.add_row(f"[{color}][{key}][/{color}]", label)

    console.print(Align.center(menu))
    console.print()


def _ask(console: Console, prompt: str, default: str = "") -> str:
    return Prompt.ask(
        f"  [bold bright_cyan]❯[/bold bright_cyan] {prompt}",
        default=default,
        console=console,
    )


def _edit_topology(cfg: dict, console: Console) -> dict:
    val = _ask(console, "Nova topologia", default=str(cfg["topology_id"]))
    return {**cfg, "topology_id": val.strip() or cfg["topology_id"]}


def _edit_margins(cfg: dict, console: Console) -> dict:
    console.print(
        "  [dim]Informe as margens separadas por espaço ou vírgula "
        "(ex: [cyan]0 0.5 1.0 1.5 2.0 2.5 3.0[/cyan])[/dim]"
    )
    current = " ".join(str(m) for m in cfg["margins"])
    raw = _ask(console, "Margens OSNR (dB)", default=current)
    try:
        raw_clean = raw.replace(",", " ")
        margins = tuple(float(x) for x in raw_clean.split() if x)
        if not margins:
            raise ValueError
        return {**cfg, "margins": margins}
    except ValueError:
        console.print("  [red]⚠  Valores inválidos — mantendo configuração anterior.[/red]")
        return cfg


def _edit_episodes(cfg: dict, console: Console) -> dict:
    raw = _ask(console, "Episódios por margem", default=str(cfg["episodes_per_margin"]))
    try:
        val = int(raw)
        assert val > 0
        return {**cfg, "episodes_per_margin": val}
    except (ValueError, AssertionError):
        console.print("  [red]⚠  Valor inválido — deve ser inteiro positivo.[/red]")
        return cfg


def _edit_episode_length(cfg: dict, console: Console) -> dict:
    raw = _ask(console, "Requisições por episódio", default=str(cfg["episode_length"]))
    try:
        val = int(raw)
        assert val > 0
        return {**cfg, "episode_length": val}
    except (ValueError, AssertionError):
        console.print("  [red]⚠  Valor inválido — deve ser inteiro positivo.[/red]")
        return cfg


def _edit_output_dir(cfg: dict, console: Console) -> dict:
    raw = _ask(console, "Diretório de saída", default=str(cfg["output_dir"]))
    path = Path(raw).expanduser()
    if not path.exists():
        confirm = Confirm.ask(
            f"  [yellow]Diretório não existe. Criar?[/yellow]",
            console=console,
            default=True,
        )
        if confirm:
            path.mkdir(parents=True, exist_ok=True)
            console.print(f"  [green]✓ Criado: {path}[/green]")
        else:
            console.print("  [dim]Mantendo diretório anterior.[/dim]")
            return cfg
    return {**cfg, "output_dir": path}


def _edit_policy(cfg: dict, console: Console) -> dict:
    console.print(f"  [dim]Política disponível: [cyan]{DEFAULT_POLICY_NAME}[/cyan][/dim]")
    _ask(console, "Pressione ENTER para confirmar", default=DEFAULT_POLICY_NAME)
    return {**cfg, "policy_name": DEFAULT_POLICY_NAME}


def _edit_load(cfg: dict, console: Console) -> dict:
    console.print(
        "  [dim]Carga de tráfego em Erlang da topologia "
        "(ex: [cyan]50  100  200  300  400[/cyan])[/dim]"
    )
    raw = _ask(console, "Carga (Erlang)", default=str(cfg["load"]))
    try:
        val = float(raw)
        if val <= 0:
            raise ValueError
        return {**cfg, "load": val}
    except ValueError:
        console.print("  [red]⚠  Valor inválido — deve ser número positivo (ex: 150.0).[/red]")
        return cfg


def _print_final_summary(outputs: MarginSweepOutputs, console: Console) -> None:
    console.print()
    console.print(Rule("[bold bright_green]✅  Simulação concluída com sucesso![/bold bright_green]",
                       style="bright_green"))
    console.print()

    tbl = Table(
        box=box.ROUNDED,
        border_style="bright_green",
        show_header=False,
        padding=(0, 2),
        expand=False,
    )
    tbl.add_column(style="dim green",         min_width=18)
    tbl.add_column(style="bold bright_white", min_width=50)
    tbl.add_row("📄 Episódios CSV", str(outputs.episodes_csv))
    tbl.add_row("📋 Resumo CSV",    str(outputs.summary_csv))

    console.print(Align.center(tbl))
    console.print()


# ── Parser (todos os add_argument do original) ───────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="SBRT26 OSNR margin sweep with first-fit — Interface Interativa CLI.",
    )
    parser.add_argument("--policy",               default=DEFAULT_POLICY_NAME,
                        choices=(DEFAULT_POLICY_NAME,))
    parser.add_argument("--topology-id",          default="nobel-eu")
    parser.add_argument("--episodes-per-margin",  type=int,   default=DEFAULT_EPISODES_PER_MARGIN)
    parser.add_argument("--request-count",        type=int,   default=DEFAULT_EPISODE_LENGTH)
    parser.add_argument("--output-dir",           type=Path,  default=SCRIPT_DIR / "resultados")
    parser.add_argument("--margins",              type=float, nargs="*", default=None)
    parser.add_argument("--load",                 type=float, default=DEFAULT_LOAD,
                        help=f"Carga de tráfego em Erlang (padrão: {DEFAULT_LOAD}).")
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        default=False,
        help="Executa diretamente sem abrir o menu interativo (modo script).",
    )
    return parser


# ═══════════════════════════════════════════════════════════════════════════════
#  main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args    = build_parser().parse_args()
    console = Console(highlight=False)

    # Estado inicial de configuração vindo dos argumentos CLI
    cfg: dict = {
        "topology_id":         args.topology_id,
        "policy_name":         args.policy,
        "margins":             DEFAULT_MARGINS if args.margins is None else tuple(args.margins),
        "episodes_per_margin": args.episodes_per_margin,
        "episode_length":      args.request_count,
        "output_dir":          args.output_dir,
        "load":                args.load,
    }

    # ── Modo não-interativo (equivalente exato ao original) ──────────────────
    if args.no_interactive:
        experiment = MarginSweepExperiment(**cfg)
        outputs    = run_margin_sweep(experiment=experiment, console=console)
        console.print(f"Episódios salvos em: {outputs.episodes_csv}")
        console.print(f"Resumo salvo em:     {outputs.summary_csv}")
        return

    # ── Modo interativo ──────────────────────────────────────────────────────
    while True:
        _print_menu(console, cfg)
        choice = _ask(console, "[bold]Escolha uma opção[/bold]", default="7").strip()

        if choice == "1":
            cfg = _edit_topology(cfg, console)
        elif choice == "2":
            cfg = _edit_margins(cfg, console)
        elif choice == "3":
            cfg = _edit_episodes(cfg, console)
        elif choice == "4":
            cfg = _edit_episode_length(cfg, console)
        elif choice == "5":
            cfg = _edit_output_dir(cfg, console)
        elif choice == "6":
            cfg = _edit_policy(cfg, console)
        elif choice == "7":
            cfg = _edit_load(cfg, console)
        elif choice == "8":
            # ── Confirmar e executar ─────────────────────────────────────────
            console.clear()
            print_banner(console)
            console.print(Align.center(_make_config_table(cfg, console)))
            console.print()
            console.print(Align.center(_make_stats_panel(cfg)))
            console.print()

            ok = Confirm.ask(
                "  [bold bright_green]Iniciar a simulação com estes parâmetros?[/bold bright_green]",
                console=console,
                default=True,
            )
            if not ok:
                continue

            try:
                experiment = MarginSweepExperiment(**cfg)
            except ValueError as exc:
                console.print(f"\n  [bold red]Erro na configuração:[/bold red] {exc}\n")
                time.sleep(2)
                continue

            outputs = run_margin_sweep(experiment=experiment, console=console)
            _print_final_summary(outputs, console)

            again = Confirm.ask(
                "  Deseja executar outra simulação?",
                console=console,
                default=False,
            )
            if not again:
                break

        elif choice == "0":
            console.print("\n  [dim]Encerrando. Até logo! 👋[/dim]\n")
            break
        else:
            console.print(f"  [yellow]⚠  Opção [bold]{choice!r}[/bold] inválida.[/yellow]")
            time.sleep(1)


__all__ = [
    "MarginSweepExperiment",
    "MarginSweepOutputs",
    "build_base_scenario",
    "build_env",
    "build_episode_scenario",
    "run_margin_sweep",
    "run_single_episode",
]


if __name__ == "__main__":
    main()
