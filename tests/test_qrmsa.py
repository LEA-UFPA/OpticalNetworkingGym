# tests/test_qrmsa.py
# -*- coding: utf-8 -*-

import os
import copy
import math
import random
import numpy as np
import pytest

from typing import Tuple

# Imports do optical-networking-gym
from optical_networking_gym.topology import Modulation, get_topology
from optical_networking_gym.wrappers.qrmsa_gym import QRMSAEnvWrapper
# Alvo correto para monkeypatch de OSNR
import optical_networking_gym.core.osnr as osnr_mod


# ---------------------------------------------------------------------
# Helpers / Fixtures
# ---------------------------------------------------------------------

def _find_topology_file():
    """Tenta nobel-eu.xml e cai para ring_4.txt. Se nada existir, marca skip."""
    base = os.path.join("examples", "topologies")
    candidates = [
        ("nobel-eu", os.path.join(base, "nobel-eu.xml")),
        ("ring_4",   os.path.join(base, "ring_4.txt")),
    ]
    for name, path in candidates:
        if os.path.exists(path):
            return name, path
    pytest.skip("Nenhum arquivo de topologia encontrado em examples/topologies (nobel-eu.xml ou ring_4.txt).")


def _define_modulations() -> Tuple[Modulation, ...]:
    return (
        Modulation("QPSK",  200_000, 2,  minimum_osnr=6.72, inband_xt=-17),
        Modulation("16QAM", 500,     4,  minimum_osnr=13.24, inband_xt=-23),
    )


@pytest.fixture(scope="module")
def small_env_args():
    topo_name, topo_file = _find_topology_file()
    mods = _define_modulations()

    topology = get_topology(
        topo_file, topo_name, mods,
        max_span_length=80, default_attenuation=0.2,
        default_noise_figure=4.5, k_paths=3
    )

    seed = 10
    random.seed(seed)
    np.random.seed(seed)

    return dict(
        topology=topology,
        seed=seed,
        allow_rejection=True,
        load=300,
        episode_length=10,
        num_spectrum_resources=10,
        channel_width=12.5,
        launch_power_dbm=0,
        frequency_slot_bandwidth=12.5e9,
        frequency_start=3e8/1565e-9,
        bandwidth=10*12.5e9,
        bit_rate_selection="discrete",
        bit_rates=(10, 40),
        margin=0,
        measure_disruptions=False,
        file_name="",
        k_paths=3,
        modulations_to_consider=2,
        defragmentation=False,
        n_defrag_services=0,
        gen_observation=True,
    )


@pytest.fixture()
def env(small_env_args):
    e = QRMSAEnvWrapper(**small_env_args)
    yield e
    try:
        e.close()
    except Exception:
        pass


# ---------------------------------------------------------------------
# Testes de inicialização e reset
# ---------------------------------------------------------------------

def test_reset_estado_inicial(env):
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape
    assert "mask" in info and info["mask"].shape[0] == env.action_space.n

    n = env.env.num_spectrum_resources
    expected = "1" * n
    for u, v in env.env.topology.edges():
        idx = env.env.topology[u][v]["index"]
        s = "".join(map(str, env.env.topology.graph["available_slots"][idx].tolist()))
        assert s == expected


def test_reject_action_valida(env):
    _, info = env.reset()
    reject_idx = env.env.action_space.n - 1
    assert info["mask"][reject_idx] == 1


# ---------------------------------------------------------------------
# Codificação/Decodificação de ações + invariantes
# ---------------------------------------------------------------------

def test_encoded_decimal_to_array_invariantes(env):
    _, info = env.reset()
    max_check = min(20, env.env.action_space.n - 1)
    for a in range(max_check):
        k, m, s = env.env.encoded_decimal_to_array(a)
        assert 0 <= k < env.env.k_paths
        assert 0 <= m < len(env.env.modulations)
        assert 0 <= s < env.env.num_spectrum_resources


def test_decimal_to_array_respeita_allowed_mods(env):
    env.reset()
    env.env.max_modulation_idx = len(env.env.modulations) - 1  # topo
    a = 0
    k, m, s = env.env.decimal_to_array(a)
    # allowed_mods conforme lógica do env
    if env.env.max_modulation_idx > 1:
        allowed_mods = list(
            range(env.env.max_modulation_idx,
                  env.env.max_modulation_idx - (env.env.modulations_to_consider - 1) - 1,
                  -1)
        )
    else:
        allowed_mods = list(range(0, env.env.modulations_to_consider))
    assert m in allowed_mods
    assert 0 <= k < env.env.k_paths
    assert 0 <= s < env.env.num_spectrum_resources


# ---------------------------------------------------------------------
# _get_candidates, get_available_slots, is_path_free, guard band
# ---------------------------------------------------------------------

def test_get_available_slots_intersection_e_guardband(env):
    env.reset()
    svc = env.env.current_service
    path = env.env.k_shortest_paths[svc.source, svc.destination][0]

    # Ocupa slot 0 no primeiro link
    n1, n2 = path.node_list[0], path.node_list[1]
    idx = env.env.topology[n1][n2]["index"]
    env.env.topology.graph["available_slots"][idx, 0] = 0

    inter = env.env.get_available_slots(path)
    assert inter[0] == 0

    env.env.topology.graph["available_slots"][idx, 1:] = 1
    inter2 = env.env.get_available_slots(path)

    modulation = env.env.modulations[0]
    num_slots = env.env.get_number_slots(svc, modulation)
    cand = env.env._get_candidates(inter2, num_slots, env.env.num_spectrum_resources)
    assert isinstance(cand, list)
    assert 0 not in cand
    assert any(c >= 1 for c in cand)


def test_is_path_free_coerencia_com_candidates(env):
    env.reset()
    svc = env.env.current_service
    path = env.env.k_shortest_paths[svc.source, svc.destination][0]
    available = env.env.get_available_slots(path)
    modulation = env.env.modulations[0]
    num_slots = env.env.get_number_slots(svc, modulation)

    cands = env.env._get_candidates(available, num_slots, env.env.num_spectrum_resources)
    assert len(cands) > 0

    c0 = cands[0]
    assert env.env.is_path_free(path, c0, num_slots) is True

    # ocupa adjacente ao fim para quebrar guard band
    end = c0 + num_slots
    if end < env.env.num_spectrum_resources:
        for i in range(len(path.node_list) - 1):
            u, v = path.node_list[i], path.node_list[i + 1]
            idx = env.env.topology[u][v]["index"]
            env.env.topology.graph["available_slots"][idx, end] = 0
        assert env.env.is_path_free(path, c0, num_slots) is False


# ---------------------------------------------------------------------
# Provisionamento, release e alocações por ID (se exposto)
# ---------------------------------------------------------------------

def test_provisionamento_e_release_atualizam_estruturas(env):
    obs, info = env.reset()
    svc = env.env.current_service

    reject_idx = env.env.action_space.n - 1
    valid = [i for i, v in enumerate(info["mask"][:-1]) if v == 1]
    if not valid:
        pytest.skip("Sem ação válida para alocação neste step")

    a = valid[0]
    k, m, s = env.env.encoded_decimal_to_array(a)
    path = env.env.k_shortest_paths[svc.source, svc.destination][k]
    modulation = env.env.modulations[m]
    ns = env.env.get_number_slots(svc, modulation)

    _obs2, reward, done, trunc, info2 = env.step(a)
    if reward < 0:
        pytest.skip("Ação resultou em rejeição; não é possível verificar alocação")

    u, v = path.node_list[0], path.node_list[1]
    idx = env.env.topology[u][v]["index"]
    seg = env.env.topology.graph["available_slots"][idx, s:s+ns]
    assert np.all(seg == 0), "Slots deveriam estar ocupados após provisionamento"

    # Checagem opcional: spectrum_slots_allocation pode não existir nesta build
    if hasattr(env.env, "spectrum_slots_allocation"):
        seg_ids = env.env.spectrum_slots_allocation[idx, s:s+ns]
        assert np.all(seg_ids == svc.service_id), "Allocation por ID do serviço incorreta"

    # Release e checagem
    env.env._release_path(svc)
    seg2 = env.env.topology.graph["available_slots"][idx, s:s+ns]
    assert np.all(seg2 == 1)
    if hasattr(env.env, "spectrum_slots_allocation"):
        seg_ids2 = env.env.spectrum_slots_allocation[idx, s:s+ns]
        assert np.all(seg_ids2 == -1)


# ---------------------------------------------------------------------
# Máscara: consistência mínima com recursos (OSNR pode variar)
# ---------------------------------------------------------------------

def test_mascara_consistencia_amostrada(env):
    obs, info = env.reset()
    svc = env.env.current_service
    checked = 0
    for a in range(min(env.env.action_space.n - 1, 50)):
        k, m, s = env.env.encoded_decimal_to_array(a)
        if k >= env.env.k_paths or m >= len(env.env.modulations) or s >= env.env.num_spectrum_resources:
            continue
        path = env.env.k_shortest_paths[svc.source, svc.destination][k]
        modulation = env.env.modulations[m]
        ns = env.env.get_number_slots(svc, modulation)
        if ns <= 0:
            continue
        avail = env.env.get_available_slots(path)
        cands = env.env._get_candidates(avail, ns, env.env.num_spectrum_resources)
        resource_ok = (s in cands)

        if info["mask"][a] == 1:
            assert resource_ok, f"Máscara 1 com recurso inválido (action={a})"
        checked += 1
    assert checked > 0


# ---------------------------------------------------------------------
# get_available_blocks / _get_spectrum_slots
# ---------------------------------------------------------------------

def test_get_available_blocks_e_spectrum_slots(env):
    env.reset()
    svc = env.env.current_service
    path = env.env.k_shortest_paths[svc.source, svc.destination][0]
    mod = env.env.modulations[0]
    ns = env.env.get_number_slots(svc, mod)

    init_idx, lengths = env.env.get_available_blocks(0, ns, j=2)
    assert len(init_idx) == len(lengths)
    assert len(init_idx) <= 2
    assert np.all(lengths >= ns)

    spec_list = env.env._get_spectrum_slots(0)
    assert isinstance(spec_list, list) and len(spec_list) == len(path.links)
    for arr in spec_list:
        assert isinstance(arr, np.ndarray)


# ---------------------------------------------------------------------
# reward / limites / retorno
# ---------------------------------------------------------------------

def test_step_reward_intervalo(env):
    _, info = env.reset()
    reject_idx = env.env.action_space.n - 1
    valid = [i for i, v in enumerate(info["mask"][:-1]) if v == 1]
    a = valid[0] if valid else reject_idx
    _obs2, reward, done, trunc, info2 = env.step(a)
    assert isinstance(reward, (int, float, np.floating))
    assert -6.0 <= float(reward) <= 3.0


# ---------------------------------------------------------------------
# _update_link_stats: evitar NaN provocando uso antes
# ---------------------------------------------------------------------

def test_update_link_stats_intervalos(env):
    env.reset()
    # cria uso: ocupa slot 0 do primeiro link de alguma rota
    svc = env.env.current_service
    path = env.env.k_shortest_paths[svc.source, svc.destination][0]
    u, v = path.node_list[0], path.node_list[1]
    idx = env.env.topology[u][v]["index"]
    env.env.topology.graph["available_slots"][idx, 0] = 0

    env.env.current_time = 1.0
    env.env._update_link_stats(u, v)
    link = env.env.topology[u][v]

    assert math.isfinite(link["utilization"])
    assert math.isfinite(link["external_fragmentation"])
    assert 0.0 <= link["utilization"] <= 1.0
    # Algumas builds podem produzir 0/0 -> NaN se sem uso; garantimos uso acima
    assert 0.0 <= link["external_fragmentation"] <= 1.0
    assert math.isfinite(link["compactness"])
    assert link["last_update"] == env.env.current_time


# ---------------------------------------------------------------------
# set_load / _get_node_pair / _get_network_compactness (compatíveis)
# ---------------------------------------------------------------------

def test_set_load_calculo(env):
    # Garante que set_load ajusta parâmetros; checa mean_service_inter_arrival_time se existir
    env.env.set_load(load=100.0, mean_service_holding_time=10.0)
    assert env.env.load == 100.0
    assert env.env.mean_service_holding_time == 10.0
    if hasattr(env.env, "mean_service_inter_arrival_time"):
        expected = 1.0 / (100.0 / 10.0)
        assert abs(env.env.mean_service_inter_arrival_time - expected) < 1e-9


def test_get_node_pair_distintos(env):
    env.reset()
    if not hasattr(env.env, "_get_node_pair"):
        pytest.skip("_get_node_pair não está exposto nesta build")
    src, src_id, dst, dst_id = env.env._get_node_pair()
    assert src != dst
    assert isinstance(src_id, int)
    assert isinstance(dst_id, (str, int))  # versões podem usar str ou int


def test_network_compactness_inicial(env):
    env.reset()
    val = env.env._get_network_compactness()
    assert val > 0.0
    assert math.isfinite(val)


# ---------------------------------------------------------------------
# Defragmentação (com monkeypatch de OSNR para estabilizar) — se existir
# ---------------------------------------------------------------------

def test_defragmentacao_move_para_esquerda(env, monkeypatch):
    if not hasattr(env.env, "defragment"):
        pytest.skip("defragment não está exposto nesta build")

    # OSNR alto para permitir realocação
    def osnr_high(_env, _svc):
        return (1e6, 0.0, 0.0)

    monkeypatch.setattr(osnr_mod, "calculate_osnr", osnr_high, raising=True)

    obs, info = env.reset()
    svc = env.env.current_service
    path = env.env.k_shortest_paths[svc.source, svc.destination][0]
    u0, v0 = path.node_list[0], path.node_list[1]
    idx0 = env.env.topology[u0][v0]["index"]
    # bloqueia slot 0 para empurrar alocação para a direita
    env.env.topology.graph["available_slots"][idx0, 0] = 0

    reject_idx = env.env.action_space.n - 1
    valid = [i for i, v in enumerate(info["mask"][:-1]) if v == 1]
    if not valid:
        pytest.skip("Sem ação válida para alocação neste step para defragmentação")
    a = valid[0]
    _obs2, reward, done, trunc, info2 = env.step(a)
    if reward < 0:
        pytest.skip("Ação resultou em rejeição; não há serviço para defragmentar")

    allocated = svc
    old_slot = allocated.initial_slot

    # libera slot 0
    env.env.topology.graph["available_slots"][idx0, 0] = 1

    env.env.defragment(10)
    assert allocated.initial_slot <= old_slot, "Defragment não moveu para a esquerda quando possível"


# ---------------------------------------------------------------------
# get_max_modulation_index (com monkeypatch de OSNR)
# ---------------------------------------------------------------------

def test_get_max_modulation_index_alcanca_modulacao_mais_alta(env, monkeypatch):
    def osnr_high(_env, _svc):
        return (1e6, 0.0, 0.0)

    monkeypatch.setattr(osnr_mod, "calculate_osnr", osnr_high, raising=True)

    env.reset()
    env.env.get_max_modulation_index()
    assert env.env.max_modulation_idx == len(env.env.modulations) - 1
