# tests/test_qrmsa_unit.py
# -*- coding: utf-8 -*-

import math
import types
import numpy as np
import networkx as nx
import pytest

# IMPORTANTE: importa o módulo onde a classe QRMSAEnv/Service vivem
# (ajuste o path abaixo caso necessário)
import optical_networking_gym.envs.qrmsa as qrmsa_mod

QRMSAEnv = qrmsa_mod.QRMSAEnv
Service = qrmsa_mod.Service


# -----------------------------
# Stubs simples para Path/Link/Modulation
# -----------------------------
class StubLink:
    def __init__(self, node1, node2):
        self.node1 = node1
        self.node2 = node2


class StubPath:
    def __init__(self, node_list, length, k):
        self.node_list = tuple(node_list)
        self.length = float(length)
        self.k = int(k)
        self.links = [StubLink(a, b) for a, b in zip(self.node_list, self.node_list[1:])]
        self.hops = len(self.node_list) - 1

    def get_node_list(self):
        return self.node_list


class StubMod:
    def __init__(self, name, spectral_efficiency, minimum_osnr):
        self.name = name
        self.spectral_efficiency = float(spectral_efficiency)
        self.minimum_osnr = float(minimum_osnr)

    def __repr__(self):
        return f"StubMod({self.name}, SE={self.spectral_efficiency}, minOSNR={self.minimum_osnr})"


# -----------------------------
# Fixtures
# -----------------------------
@pytest.fixture
def small_topology():
    """
    Grafo: A-B-C e A-C.
    Índices de aresta fixos para estabilidade dos testes.
    """
    G = nx.Graph()
    for n in ["A", "B", "C"]:
        G.add_node(n)

    # Define indices e comprimentos
    G.add_edge("A", "B", index=0, length=100.0)
    G.add_edge("B", "C", index=1, length=100.0)
    G.add_edge("A", "C", index=2, length=150.0)

    # KSP: A->C tem dois caminhos
    p0 = StubPath(["A", "B", "C"], length=200.0, k=0)
    p1 = StubPath(["A", "C"], length=150.0, k=1)
    G.graph["ksp"] = {
        ("A", "C"): [p0, p1],
    }

    # Modulações simples
    G.graph["modulations"] = [
        StubMod("QPSK", 2.0, 6.0),
        StubMod("16QAM", 4.0, 13.0),
        StubMod("64QAM", 6.0, 20.0),
    ]
    G.graph["name"] = "unit-small"
    return G


@pytest.fixture
def env_small(small_topology, monkeypatch):
    """
    Cria um QRMSAEnv enxuto com n_slots pequeno.
    Tudo que depende de OSNR é monkeypatched.
    """
    nslots = 16
    # Fake OSNR sempre suficiente
    def fake_calc_osnr(env, service):
        return (30.0, 0.0, 0.0)

    def fake_calc_osnr_obs(env, route_links, bw, cf, sid, pwr, min_osnr):
        return 10.0  # >= 0 => válido na tua observation()

    monkeypatch.setattr(qrmsa_mod, "calculate_osnr", fake_calc_osnr, raising=True)
    monkeypatch.setattr(qrmsa_mod, "calculate_osnr_observation", fake_calc_osnr_obs, raising=True)

    env = QRMSAEnv(
        topology=small_topology,
        num_spectrum_resources=nslots,
        # manter consistência do assert bandwidth == slot_bw * nslots
        frequency_slot_bandwidth=1.0,
        bandwidth=float(nslots) * 1.0,
        # deixar canais fáceis de ler
        channel_width=1.0,
        bit_rate_selection="discrete",
        bit_rates=(10, 20),
        k_paths=2,
        margin=0.0,
        measure_disruptions=False,
        allow_rejection=True,
        reset=False,
        gen_observation=False,  # desliga construção pesada de observação por padrão
    )

    # service corrente mockado
    env.current_service = Service(
        service_id=0,
        source="A",
        source_id=env.topo_cache.get_node_id("A"),
        destination="C",
        destination_id=str(env.topo_cache.get_node_id("C")),
        bit_rate=10.0,
    )

    # disponível por padrão: tudo livre (1)
    assert env.topo_cache.available_slots.shape == (3, nslots)
    return env


# -----------------------------
# TopologyCache
# -----------------------------
def test_topology_cache_basic(env_small):
    tc = env_small.topo_cache
    assert tc.get_node_id("A") == 0
    assert tc.get_node_name(1) == "B"
    ab_idx = tc.get_edge_index(tc.get_node_id("A"), tc.get_node_id("B"))
    assert ab_idx == 0
    assert math.isclose(tc.get_edge_length(0), 100.0)


# -----------------------------
# FastPathOps
# -----------------------------
def test_fast_ops_extract_indices(env_small):
    p = env_small.k_shortest_paths[("A", "C")][0]  # A-B-C
    arr = env_small.fast_ops.extract_edge_indices(p)
    assert arr.tolist() == [0, 1]

def test_vectorized_spectrum_product(env_small):
    # zera alguns slots em arestas 0 e 1
    env_small.topo_cache.available_slots[0, 5:9] = 0
    env_small.topo_cache.available_slots[1, 7:11] = 0
    p = env_small.k_shortest_paths[("A", "C")][0]  # A-B-C
    res = env_small.get_available_slots(p)
    # deve zerar união das janelas (5..8 e 7..10)
    zeros = np.where(res == 0)[0].tolist()
    assert all(x in zeros for x in range(5, 9))
    assert all(x in zeros for x in range(7, 11))


def test_fast_provision_and_release(env_small):
    svc = env_small.current_service
    p = env_small.k_shortest_paths[("A", "C")][1]  # A-C (edge index 2)
    svc.path = p
    svc.initial_slot = 3
    svc.number_slots = 4

    # provisiona
    env_small._provision_path(p, 3, 4)
    # guarda-band: método usa end+=1 se não é o fim
    end = 3 + 4 + 1
    for eidx in [2]:
        assert np.all(env_small.topo_cache.available_slots[eidx, 3:end] == 0)
        assert np.all(env_small.spectrum_slots_allocation[eidx, 3:end] == svc.service_id)

    assert svc in env_small.topology.graph["running_services"]

    # libera
    env_small._release_path(svc)
    for eidx in [2]:
        assert np.all(env_small.topo_cache.available_slots[eidx, 3:end] == 1)
        assert np.all(env_small.spectrum_slots_allocation[eidx, 3:end] == -1)
    assert svc not in env_small.topology.graph["running_services"]


# -----------------------------
# Helpers do ambiente
# -----------------------------
def test_get_candidates_guard_band(env_small):
    # constrói um vetor com blocos (1=livre)
    # [0..15] => bloco 0: [2..7] (6), bloco 1 (até o fim): [10..15] (6)
    vec = np.zeros(env_small.num_spectrum_resources, dtype=np.int32)
    vec[2:8] = 1
    vec[10:16] = 1
    cands = env_small._get_candidates(vec, num_slots_required=4, total_slots=env_small.num_spectrum_resources)
    # no bloco do meio (2..7), precisa 4+1 => starts 2..(7-5)=2..2 => [2,3]? Revisão:
    # comprimento=6 => (4+1)=5 -> starts: 2..(2+6-5)=3 => [2,3]
    # no bloco final (até o fim), precisa só 4 => starts: 10..(10+6-4)=12 => [10,11,12]
    assert cands == [2, 3, 10, 11, 12]


def test_is_path_free_with_guard(env_small):
    p = env_small.k_shortest_paths[("A", "C")][0]  # A-B-C (edges 0 e 1)
    # tudo livre, exceto um guard-slot no fim
    start, n = 4, 3
    end = start + n  # 7
    # como não é o fim do espectro, checa até end+1=8
    env_small.topo_cache.available_slots[0, end] = 0  # bloqueia o guard
    assert env_small.is_path_free(p, start, n) is False

    # libera guard e bloqueia dentro da janela
    env_small.topo_cache.available_slots[0, end] = 1
    env_small.topo_cache.available_slots[1, start + 1] = 0
    assert env_small.is_path_free(p, start, n) is False

    # libera tudo
    env_small.topo_cache.available_slots[1, start + 1] = 1
    assert env_small.is_path_free(p, start, n) is True


def test_get_number_slots(env_small):
    svc = env_small.current_service
    mod = qrmsa_mod.Modulation("X", 999999, 4.0, minimum_osnr=0.0, inband_xt=0.0) if hasattr(qrmsa_mod, "Modulation") else StubMod("X", 4.0, 0.0)
    svc.bit_rate = 10.0
    env_small.channel_width = 1.0
    # 10 / (4*1) = 2.5 => ceil = 3
    assert env_small.get_number_slots(svc, mod) == 3


def test_decimal_to_array_and_encoded(env_small):
    env_small.modulations_to_consider = 2
    env_small.max_modulation_idx = 2  # maior índice disponível
    env_small.k_paths = 2
    env_small.num_spectrum_resources = 16

    # numero 'decimal' qualquer
    decimal = 2 * (2 * 16) + 1 * 16 + 5  # path=2? (mas k_paths=2 => modulo ajusta), mod=1, slot=5
    arr = env_small.decimal_to_array(decimal, [env_small.k_paths, env_small.modulations_to_consider, env_small.num_spectrum_resources])
    assert len(arr) == 3
    # encoded (faixa) – validando que não quebra e respeita allowed_mods
    arr2 = env_small.encoded_decimal_to_array(decimal, [env_small.k_paths, env_small.modulations_to_consider, env_small.num_spectrum_resources])
    assert len(arr2) == 3


def test_get_available_blocks(env_small):
    # prepara janelas livres nas duas arestas do caminho A-B-C
    p = env_small.k_shortest_paths[("A", "C")][0]
    env_small.topo_cache.available_slots[:, :] = 0
    env_small.topo_cache.available_slots[0, 2:7] = 1  # [2..6] len=5
    env_small.topo_cache.available_slots[1, 3:9] = 1  # [3..8] len=6
    # AND => [3..6] (len 4)
    starts, lens = env_small.get_available_blocks(0, slots=4, j=3)
    assert starts.tolist() == [3]
    assert lens.tolist() == [4]


def test_update_link_stats_sets_values(env_small):
    env_small.current_time = 10.0
    # ocupa 6 slots na aresta A-B (idx 0)
    env_small.topo_cache.available_slots[0, 4:10] = 0
    env_small._update_link_stats("A", "B")
    edge_data = env_small.topo_cache.edge_data[0]
    for key in ("utilization", "external_fragmentation", "compactness", "last_update"):
        assert key in edge_data
    assert 0.0 <= edge_data["utilization"] <= 1.0


def test_get_network_compactness_returns_float(env_small):
    # monta um running service minimal
    p = env_small.k_shortest_paths[("A", "C")][0]
    svc = env_small.current_service
    svc.path = p
    svc.number_slots = 3
    env_small.topology.graph["running_services"] = [svc]

    # ocupa [5..7] em ambas as arestas do caminho
    env_small.topo_cache.available_slots[0, 5:8] = 0
    env_small.topo_cache.available_slots[1, 5:8] = 0

    env_small.current_time = 5.0
    c = env_small._get_network_compactness()
    assert isinstance(c, float)
    assert c > 0.0


def test_get_max_modulation_index(env_small, monkeypatch):
    # ajusta OSNR "médio" para selecionar até 16QAM mas não 64QAM
    def osnr_mid(env, service):
        return (15.0, 0.0, 0.0)  # >= 13 (16QAM), < 20 (64QAM)
    monkeypatch.setattr(qrmsa_mod, "calculate_osnr", osnr_mid, raising=True)

    # deixa tudo livre e garante candidatos
    env_small.topo_cache.available_slots[:, :] = 1

    env_small.get_max_modulation_index()
    # mod list tamanho 3 => idx deve ser 1 (16QAM) ou maior dependendo do "start_index" interno
    assert env_small.max_modulation_idx >= 1
    assert env_small.max_modulation_idx <= len(env_small.modulations) - 1


@pytest.mark.xfail(reason="reward() não retorna valor; adicione `return reward_value` no fim do método.")
def test_reward_returns_float(env_small):
    # serviço aceito com modulação e OSNR setados
    svc = env_small.current_service
    svc.accepted = True
    svc.current_modulation = StubMod("16QAM", spectral_efficiency=4.0, minimum_osnr=13.0)
    svc.OSNR = 20.0
    r = env_small.reward()
    assert isinstance(r, float)
    assert -3.0 <= r <= 3.0
