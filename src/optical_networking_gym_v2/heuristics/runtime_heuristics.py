from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from optical_networking_gym_v2.contracts import ActionSelection, CandidateRewardMetrics
from optical_networking_gym_v2.contracts.enums import MaskMode
from optical_networking_gym_v2.network.topology import TopologyModel
from optical_networking_gym_v2.runtime.action_codec import decode_action, encode_action, reject_action
from optical_networking_gym_v2.runtime.request_analysis import PATH_FEATURE_NAMES, RequestAnalysis
from optical_networking_gym_v2.runtime.runtime_state import RuntimeState

if TYPE_CHECKING:
    from optical_networking_gym_v2.config.scenario import ScenarioConfig
    from optical_networking_gym_v2.contracts import ServiceRequest
    from optical_networking_gym_v2.envs.optical_env import OpticalEnv
    from optical_networking_gym_v2.runtime.simulator import Simulator


@dataclass(frozen=True, slots=True)
class RuntimeHeuristicContext:
    simulator: Simulator
    config: ScenarioConfig
    topology: TopologyModel
    state: RuntimeState
    request: ServiceRequest
    analysis: RequestAnalysis
    action_mask: np.ndarray | None

    @property
    def reject_action(self) -> int:
        return reject_action(self.config)

    def decode_action(self, action: int) -> ActionSelection | None:
        decoded = decode_action(self.config, action)
        if decoded is None:
            return None
        if decoded.path_index >= len(self.analysis.paths):
            return None
        if decoded.modulation_offset >= len(self.analysis.modulation_indices):
            return None
        return ActionSelection(
            path_index=int(decoded.path_index),
            modulation_index=int(self.analysis.modulation_indices[decoded.modulation_offset]),
            initial_slot=int(decoded.initial_slot),
        )

    def selected_candidate_metrics(self, action: int) -> CandidateRewardMetrics | None:
        selection = self.decode_action(action)
        if selection is None:
            return None
        return self.analysis.selected_candidate_metrics(
            path_index=selection.path_index,
            modulation_index=selection.modulation_index,
            initial_slot=selection.initial_slot,
        )


def _resolve_simulator(source: OpticalEnv | Simulator | RuntimeHeuristicContext):
    if isinstance(source, RuntimeHeuristicContext):
        return source.simulator

    simulator = getattr(source, "simulator", None)
    if simulator is not None and hasattr(simulator, "current_analysis") and hasattr(simulator, "topology"):
        return simulator
    if hasattr(source, "current_analysis") and hasattr(source, "topology") and hasattr(source, "config"):
        return source
    raise TypeError("runtime heuristic requires an OpticalEnv, Simulator, or RuntimeHeuristicContext")


def build_runtime_heuristic_context(
    source: OpticalEnv | Simulator | RuntimeHeuristicContext,
) -> RuntimeHeuristicContext:
    if isinstance(source, RuntimeHeuristicContext):
        return source

    simulator = _resolve_simulator(source)
    if simulator.state is None or simulator.current_request is None:
        raise RuntimeError("reset() must be called before evaluating runtime heuristics")

    analysis = simulator.current_analysis
    if analysis is None:
        analysis = simulator.analysis_engine.build(simulator.state, simulator.current_request)

    return RuntimeHeuristicContext(
        simulator=simulator,
        config=simulator.config,
        topology=simulator.topology,
        state=simulator.state,
        request=simulator.current_request,
        analysis=analysis,
        action_mask=simulator.current_mask,
    )


def _valid_start_flags(context: RuntimeHeuristicContext) -> np.ndarray:
    if context.config.mask_mode is MaskMode.RESOURCE_ONLY:
        return context.analysis.resource_valid_starts
    return context.analysis.qot_valid_starts


_PATH_LINK_UTIL_MEAN_INDEX = int(PATH_FEATURE_NAMES.index("path_link_util_mean"))
_PATH_LINK_UTIL_MAX_INDEX = int(PATH_FEATURE_NAMES.index("path_link_util_max"))
_PATH_COMMON_FREE_RATIO_INDEX = int(PATH_FEATURE_NAMES.index("path_common_free_ratio"))


def select_first_fit_decision(
    source: OpticalEnv | Simulator | RuntimeHeuristicContext,
) -> tuple[int, bool, bool]:
    context = build_runtime_heuristic_context(source)
    blocked_due_to_resources = False
    blocked_due_to_qot = False
    use_resource_only = context.config.mask_mode is MaskMode.RESOURCE_ONLY

    for path_index, _path in enumerate(context.analysis.paths):
        for modulation_offset, _modulation_index in enumerate(context.analysis.modulation_indices):
            resource_candidates = np.flatnonzero(
                context.analysis.resource_valid_starts[path_index, modulation_offset, :]
            )
            if resource_candidates.size == 0:
                blocked_due_to_resources = True
                continue

            if use_resource_only:
                return (
                    encode_action(
                        context.config,
                        path_index=path_index,
                        modulation_offset=modulation_offset,
                        initial_slot=int(resource_candidates[0]),
                    ),
                    False,
                    False,
                )

            qot_candidates = np.flatnonzero(context.analysis.qot_valid_starts[path_index, modulation_offset, :])
            if qot_candidates.size == 0:
                blocked_due_to_qot = True
                continue

            return (
                encode_action(
                    context.config,
                    path_index=path_index,
                    modulation_offset=modulation_offset,
                    initial_slot=int(qot_candidates[0]),
                ),
                False,
                False,
            )

    if blocked_due_to_qot:
        blocked_due_to_resources = False
    return context.reject_action, blocked_due_to_resources, blocked_due_to_qot


def select_first_fit_action(source: OpticalEnv | Simulator | RuntimeHeuristicContext) -> int:
    action, _, _ = select_first_fit_decision(source)
    return action


def select_first_fit_informative_action(
    source: OpticalEnv | Simulator | RuntimeHeuristicContext,
) -> int:
    """Variante informativa do first-fit.

    Ao contrário de :func:`select_first_fit_action` (versão baseada em máscara),
    que retorna sempre ``reject_action`` quando não há ação válida — fazendo o
    simulador registrar tudo como ``REJECTED_BY_AGENT`` — esta função retorna
    uma **ação real** que permite ao simulador distinguir corretamente:

    * ``BLOCKED_RESOURCES`` — todos os slots estão ocupados em todos os caminhos.
    * ``BLOCKED_QOT``       — há slots livres, mas nenhum passa no critério de OSNR.

    Quando existe uma alocação válida, o comportamento é idêntico ao first-fit
    convencional.
    """
    context = build_runtime_heuristic_context(source)
    use_resource_only = context.config.mask_mode is MaskMode.RESOURCE_ONLY

    # Primeira combinação (caminho, modulação, slot) que tem recursos mas falha no QoT.
    # Será usada para acionar BLOCKED_QOT no simulador caso não haja opção válida.
    first_qot_block_path: int | None = None
    first_qot_block_modulation: int | None = None
    first_qot_block_slot: int | None = None

    for path_index, _path in enumerate(context.analysis.paths):
        for modulation_offset, _modulation_index in enumerate(context.analysis.modulation_indices):
            resource_candidates = np.flatnonzero(
                context.analysis.resource_valid_starts[path_index, modulation_offset, :]
            )
            if resource_candidates.size == 0:
                # Sem espectro livre para esta combinação — continua.
                continue

            if use_resource_only:
                # Modo sem verificação de QoT: qualquer slot com recursos é válido.
                return encode_action(
                    context.config,
                    path_index=path_index,
                    modulation_offset=modulation_offset,
                    initial_slot=int(resource_candidates[0]),
                )

            qot_candidates = np.flatnonzero(
                context.analysis.qot_valid_starts[path_index, modulation_offset, :]
            )
            if qot_candidates.size == 0:
                # Há recursos, mas nenhum slot passa no critério de OSNR.
                # Guarda a primeira ocorrência para uso posterior.
                if first_qot_block_path is None:
                    first_qot_block_path = path_index
                    first_qot_block_modulation = modulation_offset
                    first_qot_block_slot = int(resource_candidates[0])
                continue

            # Slot válido tanto em recursos quanto em QoT → aceita.
            return encode_action(
                context.config,
                path_index=path_index,
                modulation_offset=modulation_offset,
                initial_slot=int(qot_candidates[0]),
            )

    # Nenhuma ação totalmente válida encontrada.
    if first_qot_block_path is not None:
        # Há espectro livre, mas o QoT falha em todas as opções.
        # Retorna a ação com recursos mas sem QoT: o simulador registrará BLOCKED_QOT.
        return encode_action(
            context.config,
            path_index=first_qot_block_path,
            modulation_offset=first_qot_block_modulation,  # type: ignore[arg-type]
            initial_slot=first_qot_block_slot,             # type: ignore[arg-type]
        )

    # Todos os slots de todos os caminhos estão ocupados.
    # Qualquer ação decodificável falhará na verificação de resource_valid_starts
    # e o simulador registrará BLOCKED_RESOURCES.
    return encode_action(context.config, path_index=0, modulation_offset=0, initial_slot=0)


def select_random_action(
    source: OpticalEnv | Simulator | RuntimeHeuristicContext,
    *,
    rng: np.random.Generator | None = None,
) -> int:
    context = build_runtime_heuristic_context(source)
    generator = rng if rng is not None else np.random.default_rng()
    valid_flags = _valid_start_flags(context)
    selected_action = context.reject_action
    valid_count = 0

    for path_index, _path in enumerate(context.analysis.paths):
        for modulation_offset, _modulation_index in enumerate(context.analysis.modulation_indices):
            candidate_indices = np.flatnonzero(valid_flags[path_index, modulation_offset, :])
            for initial_slot in candidate_indices:
                valid_count += 1
                if int(generator.integers(valid_count)) != 0:
                    continue
                selected_action = encode_action(
                    context.config,
                    path_index=path_index,
                    modulation_offset=modulation_offset,
                    initial_slot=int(initial_slot),
                )

    return selected_action


def select_load_balancing_action(
    source: OpticalEnv | Simulator | RuntimeHeuristicContext,
) -> int:
    context = build_runtime_heuristic_context(source)
    valid_flags = _valid_start_flags(context)
    best_key: tuple[float, float, float, float, float, int] | None = None
    best_action = context.reject_action

    for path_index, _path in enumerate(context.analysis.paths):
        path_features = context.analysis.path_features[path_index]
        path_link_util_mean = float(path_features[_PATH_LINK_UTIL_MEAN_INDEX])
        path_link_util_max = float(path_features[_PATH_LINK_UTIL_MAX_INDEX])
        path_common_free_ratio = float(path_features[_PATH_COMMON_FREE_RATIO_INDEX])

        for modulation_offset, _modulation_index in enumerate(context.analysis.modulation_indices):
            candidate_indices = np.flatnonzero(valid_flags[path_index, modulation_offset, :])
            for initial_slot in candidate_indices.tolist():
                action = encode_action(
                    context.config,
                    path_index=path_index,
                    modulation_offset=modulation_offset,
                    initial_slot=int(initial_slot),
                )
                metrics = context.selected_candidate_metrics(action)
                if metrics is None:
                    continue
                candidate_key = (
                    path_link_util_max,
                    path_link_util_mean,
                    -path_common_free_ratio,
                    float(metrics.fragmentation_damage_num_blocks),
                    float(metrics.fragmentation_damage_largest_block),
                    int(action),
                )
                if best_key is None or candidate_key < best_key:
                    best_key = candidate_key
                    best_action = int(action)

    return best_action


__all__ = [
    "RuntimeHeuristicContext",
    "build_runtime_heuristic_context",
    "select_first_fit_action",
    "select_first_fit_decision",
    "select_first_fit_informative_action",
    "select_load_balancing_action",
    "select_random_action",
]
