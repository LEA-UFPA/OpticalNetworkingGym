# tests/test_qrmsa.py

import math
import os
import random
from typing import Tuple

import numpy as np
import pytest

import optical_networking_gym.core.osnr as osnr_mod
from optical_networking_gym.envs.qrmsa import Service
from optical_networking_gym.topology import Modulation, get_topology
from optical_networking_gym.wrappers.qrmsa_gym import QRMSAEnvWrapper


def _find_topology_file() -> tuple[str, str]:
    base = os.path.join("examples", "topologies")
    candidates = [
        ("nobel-eu", os.path.join(base, "nobel-eu.xml")),
        ("ring_4", os.path.join(base, "ring_4.txt")),
    ]
    for name, path in candidates:
        if os.path.exists(path):
            return name, path
    pytest.skip("Nenhum arquivo de topologia encontrado em examples/topologies")


def _define_modulations() -> Tuple[Modulation, ...]:
    return (
        Modulation("QPSK", 200_000, 2, minimum_osnr=6.72, inband_xt=-17),
        Modulation("16QAM", 500, 4, minimum_osnr=13.24, inband_xt=-23),
    )


def _find_resource_feasible_action(env: QRMSAEnvWrapper) -> int | None:
    """Encontra uma ação com recursos válidos sem considerar OSNR."""
    svc = env.env.current_service
    for action in range(env.env.action_space.n - 1):
        k, m, s = env.env.encoded_decimal_to_array(action)
        if (
            k >= env.env.k_paths
            or m >= len(env.env.modulations)
            or s >= env.env.num_spectrum_resources
        ):
            continue
        path = env.env.k_shortest_paths[svc.source, svc.destination][k]
        modulation = env.env.modulations[m]
        ns = env.env.get_number_slots(svc, modulation)
        if ns <= 0:
            continue
        available = env.env.get_available_slots(path)
        candidates = env.env._get_candidates(
            available, ns, env.env.num_spectrum_resources
        )
        if s in candidates:
            return action
    return None


def _make_service(
    env: QRMSAEnvWrapper,
    service_id: int,
    source: str,
    destination: str,
    bit_rate: float = 40.0,
    holding_time: float = 1e6,
) -> Service:
    node_indices = env.env.topology.graph.get(
        "node_indices", list(env.env.topology.nodes())
    )
    source_id = env.env.topology.nodes[source].get("id", node_indices.index(source))
    destination_id = env.env.topology.nodes[destination].get(
        "id", node_indices.index(destination)
    )
    return Service(
        service_id=service_id,
        source=source,
        source_id=source_id,
        destination=destination,
        destination_id=str(destination_id),
        arrival_time=env.env.current_time,
        holding_time=holding_time,
        bit_rate=bit_rate,
    )


def _configure_service_candidate(
    env: QRMSAEnvWrapper,
    service: Service,
    path,
    initial_slot: int,
    modulation: Modulation,
) -> int:
    number_slots = env.env.get_number_slots(service, modulation)
    service.path = path
    service.initial_slot = initial_slot
    service.number_slots = number_slots
    service.center_frequency = (
        env.env.frequency_start
        + (env.env.frequency_slot_bandwidth * initial_slot)
        + (env.env.frequency_slot_bandwidth * (number_slots / 2.0))
    )
    service.bandwidth = env.env.frequency_slot_bandwidth * number_slots
    service.launch_power = env.env.launch_power
    service.current_modulation = modulation
    return number_slots


def _find_action_for_choice(
    env: QRMSAEnvWrapper, route_idx: int, modulation_idx: int, slot: int
) -> int | None:
    for action in range(env.env.action_space.n - 1):
        k, m, s = env.env.encoded_decimal_to_array(action)
        if (k, m, s) == (route_idx, modulation_idx, slot):
            return action
    return None


def _find_overlap_scenario(
    env: QRMSAEnvWrapper, modulation_idx: int = 0, bit_rate: float = 40.0
) -> tuple[str, str, int]:
    modulation = env.env.modulations[modulation_idx]
    for (source, destination), paths in env.env.k_shortest_paths.items():
        for route_idx, path in enumerate(paths):
            probe = _make_service(
                env,
                service_id=9000 + route_idx,
                source=source,
                destination=destination,
                bit_rate=bit_rate,
            )
            number_slots = env.env.get_number_slots(probe, modulation)
            if number_slots <= 0:
                continue
            candidates = env.env._get_candidates(
                env.env.get_available_slots(path),
                number_slots,
                env.env.num_spectrum_resources,
            )
            if len(candidates) >= 2:
                return source, destination, route_idx
    pytest.skip("Nenhum cenario com duas alocacoes no mesmo caminho foi encontrado")


def _allocate_custom_service(
    env: QRMSAEnvWrapper,
    source: str,
    destination: str,
    route_idx: int,
    modulation_idx: int,
    service_id: int,
    bit_rate: float = 40.0,
    preferred_slot: int | None = None,
) -> tuple[Service, int]:
    path = env.env.k_shortest_paths[source, destination][route_idx]
    modulation = env.env.modulations[modulation_idx]
    probe = _make_service(env, service_id, source, destination, bit_rate=bit_rate)
    number_slots = env.env.get_number_slots(probe, modulation)
    candidates = env.env._get_candidates(
        env.env.get_available_slots(path),
        number_slots,
        env.env.num_spectrum_resources,
    )
    if not candidates:
        pytest.skip("Sem slots candidatos para o servico customizado")

    ordered_candidates = list(candidates)
    if preferred_slot is not None and preferred_slot in ordered_candidates:
        ordered_candidates.remove(preferred_slot)
        ordered_candidates.insert(0, preferred_slot)

    chosen_slot = None
    for slot in ordered_candidates:
        probe = _make_service(env, service_id, source, destination, bit_rate=bit_rate)
        _configure_service_candidate(env, probe, path, slot, modulation)
        osnr, _, _ = osnr_mod.calculate_osnr(env.env, probe, env.env.qot_constraint)
        if osnr >= modulation.minimum_osnr + env.env.margin:
            chosen_slot = slot
            break

    if chosen_slot is None:
        pytest.skip("Nenhum slot candidato atendeu ao limiar de QoT")

    action = _find_action_for_choice(env, route_idx, modulation_idx, chosen_slot)
    assert action is not None

    service = _make_service(env, service_id, source, destination, bit_rate=bit_rate)
    _configure_service_candidate(env, service, path, chosen_slot, modulation)
    env.env.current_service = service
    _obs, reward, _done, _trunc, _info = env.step(action)

    if reward < 0 or not service.accepted:
        pytest.fail(f"Falha ao alocar servico customizado no slot {chosen_slot}")

    return service, chosen_slot


@pytest.fixture(scope="module")
def small_env_args():
    topo_name, topo_file = _find_topology_file()
    mods = _define_modulations()

    topology = get_topology(
        topo_file,
        topo_name,
        mods,
        max_span_length=80,
        default_attenuation=0.2,
        default_noise_figure=4.5,
        k_paths=3,
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
        frequency_start=3e8 / 1565e-9,
        bandwidth=10 * 12.5e9,
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
        rl_mode=True,
    )


@pytest.fixture()
def env(small_env_args):
    e = QRMSAEnvWrapper(**small_env_args)
    yield e
    try:
        e.close()
    except Exception:
        pass


@pytest.fixture()
def disruption_env(small_env_args):
    args = dict(small_env_args)
    args["measure_disruptions"] = True
    e = QRMSAEnvWrapper(**args)
    yield e
    try:
        e.close()
    except Exception:
        pass


def test_reset_estado_inicial(env):
    obs, info = env.reset()
    assert obs.shape == env.observation_space.shape
    assert "mask" in info
    assert info["mask"].shape[0] == env.action_space.n

    n = env.env.num_spectrum_resources
    expected = "1" * n
    for u, v in env.env.topology.edges():
        idx = env.env.topology[u][v]["index"]
        slots = env.env.topology.graph["available_slots"][idx].tolist()
        assert "".join(map(str, slots)) == expected


def test_reject_action_valida(env):
    _, info = env.reset()
    reject_idx = env.env.action_space.n - 1
    assert info["mask"][reject_idx] == 1


def test_encoded_decimal_to_array_invariantes(env):
    env.reset()
    max_check = min(20, env.env.action_space.n - 1)
    for a in range(max_check):
        k, m, s = env.env.encoded_decimal_to_array(a)
        assert 0 <= k < env.env.k_paths
        assert 0 <= m < len(env.env.modulations)
        assert 0 <= s < env.env.num_spectrum_resources


def test_decimal_to_array_respeita_allowed_mods(env):
    env.reset()
    env.env.max_modulation_idx = len(env.env.modulations) - 1
    k, m, s = env.env.decimal_to_array(0)
    if env.env.max_modulation_idx > 1:
        allowed_mods = list(
            range(
                env.env.max_modulation_idx,
                env.env.max_modulation_idx - (env.env.modulations_to_consider - 1) - 1,
                -1,
            )
        )
    else:
        allowed_mods = list(range(0, env.env.modulations_to_consider))
    assert m in allowed_mods
    assert 0 <= k < env.env.k_paths
    assert 0 <= s < env.env.num_spectrum_resources


def test_get_available_slots_intersection_e_guardband(env):
    env.reset()
    svc = env.env.current_service
    path = env.env.k_shortest_paths[svc.source, svc.destination][0]

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

    end = c0 + num_slots
    if end < env.env.num_spectrum_resources:
        for i in range(len(path.node_list) - 1):
            u, v = path.node_list[i], path.node_list[i + 1]
            idx = env.env.topology[u][v]["index"]
            env.env.topology.graph["available_slots"][idx, end] = 0
        assert env.env.is_path_free(path, c0, num_slots) is False


def test_provisionamento_e_release_atualizam_estruturas(env):
    _, info = env.reset()
    svc = env.env.current_service
    valid = [i for i, v in enumerate(info["mask"][:-1]) if v == 1]
    if not valid:
        pytest.skip("Sem ação válida para alocação neste step")

    a = valid[0]
    k, m, s = env.env.encoded_decimal_to_array(a)
    path = env.env.k_shortest_paths[svc.source, svc.destination][k]
    modulation = env.env.modulations[m]
    ns = env.env.get_number_slots(svc, modulation)

    _obs2, reward, _done, _trunc, _info2 = env.step(a)
    if reward < 0:
        pytest.skip("Ação resultou em rejeição; não é possível verificar alocação")

    u, v = path.node_list[0], path.node_list[1]
    idx = env.env.topology[u][v]["index"]
    seg = env.env.topology.graph["available_slots"][idx, s : s + ns]
    assert np.all(seg == 0)

    if hasattr(env.env, "spectrum_slots_allocation"):
        seg_ids = env.env.spectrum_slots_allocation[idx, s : s + ns]
        assert np.all(seg_ids == svc.service_id)

    env.env._release_path(svc)
    seg2 = env.env.topology.graph["available_slots"][idx, s : s + ns]
    assert np.all(seg2 == 1)
    if hasattr(env.env, "spectrum_slots_allocation"):
        seg_ids2 = env.env.spectrum_slots_allocation[idx, s : s + ns]
        assert np.all(seg_ids2 == -1)


def test_mascara_consistencia_amostrada(env):
    _, info = env.reset()
    svc = env.env.current_service
    checked = 0
    for a in range(min(env.env.action_space.n - 1, 50)):
        k, m, s = env.env.encoded_decimal_to_array(a)
        if (
            k >= env.env.k_paths
            or m >= len(env.env.modulations)
            or s >= env.env.num_spectrum_resources
        ):
            continue
        path = env.env.k_shortest_paths[svc.source, svc.destination][k]
        modulation = env.env.modulations[m]
        ns = env.env.get_number_slots(svc, modulation)
        if ns <= 0:
            continue
        avail = env.env.get_available_slots(path)
        cands = env.env._get_candidates(avail, ns, env.env.num_spectrum_resources)
        resource_ok = s in cands
        if info["mask"][a] == 1:
            assert resource_ok, f"Máscara marcou ação inválida por recurso (action={a})"
        checked += 1
    assert checked > 0


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


def test_step_reward_intervalo(env):
    _, info = env.reset()
    reject_idx = env.env.action_space.n - 1
    valid = [i for i, v in enumerate(info["mask"][:-1]) if v == 1]
    a = valid[0] if valid else reject_idx
    _obs2, reward, _done, _trunc, _info2 = env.step(a)
    assert isinstance(reward, (int, float, np.floating))
    assert -6.0 <= float(reward) <= 3.0


def test_gsnr_mock_controla_mascara(env):
    def osnr_high(_env, _svc, qot_constraint="ASE+NLI"):
        return (1e6, 0.0, 0.0)

    def osnr_low(_env, _svc, qot_constraint="ASE+NLI"):
        return (-1e6, 0.0, 0.0)

    env.env.osnr_calculator = osnr_high
    _obs, info_high = env.reset()
    assert int(np.sum(info_high["mask"][:-1])) > 0

    env.env.osnr_calculator = osnr_low
    _obs, info_low = env.reset()
    assert int(np.sum(info_low["mask"][:-1])) == 0
    assert info_low["mask"][-1] == 1


def test_step_rejeita_quando_gsnr_mock_baixo(env):
    def osnr_low(_env, _svc, qot_constraint="ASE+NLI"):
        return (-1e6, 0.0, 0.0)

    env.env.osnr_calculator = osnr_low
    env.reset()
    action = _find_resource_feasible_action(env)
    if action is None:
        pytest.skip("Não foi possível encontrar ação com recurso válido")

    _obs, reward, _done, _trunc, info = env.step(action)
    assert reward < 0
    assert info["osnr"] < info["osnr_req"]


def test_margem_osnr_controla_mascara(env):
    env.env.margin = -1000.0
    _obs, info_relaxado = env.reset()
    valid_relaxado = int(np.sum(info_relaxado["mask"][:-1]))
    assert valid_relaxado > 0

    env.env.margin = 1000.0
    _obs, info_restrito = env.reset()
    valid_restrito = int(np.sum(info_restrito["mask"][:-1]))
    assert valid_restrito == 0
    assert valid_restrito <= valid_relaxado
    assert info_restrito["mask"][-1] == 1


def test_step_rejeita_quando_qot_exigido_muito_alto(env):
    env.env.margin = 1000.0
    env.reset()
    action = _find_resource_feasible_action(env)
    if action is None:
        pytest.skip("Não foi possível encontrar ação com recurso válido")

    _obs, reward, _done, _trunc, info = env.step(action)
    assert reward < 0
    assert info["osnr"] < info["osnr_req"]


def test_compute_slot_osnr_vectorized_consistente_com_individual(env):
    env.reset()
    svc = env.env.current_service
    path = env.env.k_shortest_paths[svc.source, svc.destination][0]
    available = env.env.get_available_slots(path).astype(np.int32, copy=False)

    vec_osnr = osnr_mod.compute_slot_osnr_vectorized(
        env.env, path, available, env.env.qot_constraint
    )
    free_slots = np.where(available == 1)[0]
    if free_slots.size == 0:
        pytest.skip("Sem slots livres para validar OSNR vetorizado")

    for slot in free_slots[:3]:
        temp = type("TempService", (), {})()
        temp.path = path
        temp.initial_slot = int(slot)
        temp.number_slots = 1
        temp.center_frequency = (
            env.env.frequency_start
            + (env.env.frequency_slot_bandwidth * slot)
            + (env.env.frequency_slot_bandwidth / 2.0)
        )
        temp.bandwidth = env.env.channel_width * 1e9
        temp.launch_power = env.env.launch_power
        temp.service_id = -999999

        indiv_gsnr, _ase, _nli = osnr_mod.calculate_osnr(
            env.env, temp, env.env.qot_constraint
        )
        assert np.isclose(vec_osnr[slot], indiv_gsnr, atol=1e-6)


def test_validate_osnr_vectorized_retorna_true(env):
    env.reset()
    svc = env.env.current_service
    path = env.env.k_shortest_paths[svc.source, svc.destination][0]
    available = env.env.get_available_slots(path).astype(np.int32, copy=False)
    assert osnr_mod.validate_osnr_vectorized(
        env.env,
        path,
        available,
        tolerance=1e-6,
        qot_constraint=env.env.qot_constraint,
    )


def test_update_link_stats_intervalos(env):
    env.reset()
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
    assert 0.0 <= link["external_fragmentation"] <= 1.0
    assert math.isfinite(link["compactness"])
    assert link["last_update"] == env.env.current_time


def test_set_load_calculo(env):
    env.env.set_load(load=100.0, mean_service_holding_time=10.0)
    assert env.env.load == 100.0
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
    assert isinstance(dst_id, (str, int))


def test_network_compactness_inicial(env):
    env.reset()
    val = env.env._get_network_compactness()
    assert val > 0.0
    assert math.isfinite(val)


def test_defragmentacao_move_para_esquerda(env):
    if not hasattr(env.env, "defragment"):
        pytest.skip("defragment não está exposto nesta build")

    def osnr_high(_env, _svc, qot_constraint="ASE+NLI"):
        return (1e6, 0.0, 0.0)

    env.env.osnr_calculator = osnr_high

    env.reset()
    action = _find_resource_feasible_action(env)
    if action is None:
        pytest.skip("Sem ação com recursos válidos para testar defragmentação")

    allocated = env.env.current_service
    _obs2, reward, _done, _trunc, _info2 = env.step(action)
    if reward < 0:
        pytest.skip("Ação resultou em rejeição; não há serviço para defragmentar")

    old_slot = allocated.initial_slot
    if old_slot <= 0:
        pytest.skip("Serviço já está no início do espectro; sem movimento para validar")

    path = allocated.path
    for i in range(len(path.node_list) - 1):
        u, v = path.node_list[i], path.node_list[i + 1]
        idx = env.env.topology[u][v]["index"]
        env.env.topology.graph["available_slots"][idx, :old_slot] = 1

    env.env.defragment(10)
    assert allocated.initial_slot <= old_slot


def test_get_max_modulation_index_alcanca_modulacao_mais_alta(env):
    def osnr_high(_env, _svc, qot_constraint="ASE+NLI"):
        return (1e6, 0.0, 0.0)

    env.env.osnr_calculator = osnr_high
    env.reset()
    env.env.get_max_modulation_index()
    assert env.env.max_modulation_idx == len(env.env.modulations) - 1


def test_measure_disruptions_atualiza_osnr_dos_servicos_impactados(disruption_env):
    disruption_env.reset()
    source, destination, route_idx = _find_overlap_scenario(
        disruption_env, modulation_idx=0, bit_rate=40.0
    )

    first_service, first_slot = _allocate_custom_service(
        disruption_env,
        source=source,
        destination=destination,
        route_idx=route_idx,
        modulation_idx=0,
        service_id=5001,
        bit_rate=40.0,
    )
    first_osnr_before = first_service.OSNR

    second_service, _second_slot = _allocate_custom_service(
        disruption_env,
        source=source,
        destination=destination,
        route_idx=route_idx,
        modulation_idx=0,
        service_id=5002,
        bit_rate=40.0,
        preferred_slot=first_slot + first_service.number_slots + 1,
    )

    expected_osnr, expected_ase, expected_nli = osnr_mod.calculate_osnr(
        disruption_env.env, first_service, disruption_env.env.qot_constraint
    )

    assert second_service.accepted is True
    assert not math.isclose(first_osnr_before, expected_osnr, abs_tol=1e-6)
    assert math.isclose(first_service.OSNR, expected_osnr, abs_tol=1e-6)
    assert math.isclose(first_service.ASE, expected_ase, abs_tol=1e-6)
    assert math.isclose(first_service.NLI, expected_nli, abs_tol=1e-6)


def test_measure_disruptions_restaura_qot_apos_release(disruption_env):
    disruption_env.reset()
    source, destination, route_idx = _find_overlap_scenario(
        disruption_env, modulation_idx=0, bit_rate=40.0
    )

    first_service, first_slot = _allocate_custom_service(
        disruption_env,
        source=source,
        destination=destination,
        route_idx=route_idx,
        modulation_idx=0,
        service_id=5101,
        bit_rate=40.0,
    )
    first_osnr_before = first_service.OSNR

    second_service, _second_slot = _allocate_custom_service(
        disruption_env,
        source=source,
        destination=destination,
        route_idx=route_idx,
        modulation_idx=0,
        service_id=5102,
        bit_rate=40.0,
        preferred_slot=first_slot + first_service.number_slots + 1,
    )

    disruption_env.env._release_path(second_service)
    expected_osnr, expected_ase, expected_nli = osnr_mod.calculate_osnr(
        disruption_env.env, first_service, disruption_env.env.qot_constraint
    )

    assert math.isclose(first_service.OSNR, expected_osnr, abs_tol=1e-6)
    assert math.isclose(first_service.ASE, expected_ase, abs_tol=1e-6)
    assert math.isclose(first_service.NLI, expected_nli, abs_tol=1e-6)
    assert math.isclose(first_service.OSNR, first_osnr_before, abs_tol=1e-6)


def test_measure_disruptions_desligado_mantem_snapshot_qot(small_env_args):
    args = dict(small_env_args)
    args["measure_disruptions"] = False
    local_env = QRMSAEnvWrapper(**args)
    try:
        local_env.reset()
        source, destination, route_idx = _find_overlap_scenario(
            local_env, modulation_idx=0, bit_rate=40.0
        )

        first_service, first_slot = _allocate_custom_service(
            local_env,
            source=source,
            destination=destination,
            route_idx=route_idx,
            modulation_idx=0,
            service_id=5201,
            bit_rate=40.0,
        )
        first_osnr_before = first_service.OSNR

        _second_service, _second_slot = _allocate_custom_service(
            local_env,
            source=source,
            destination=destination,
            route_idx=route_idx,
            modulation_idx=0,
            service_id=5202,
            bit_rate=40.0,
            preferred_slot=first_slot + first_service.number_slots + 1,
        )

        expected_osnr, _expected_ase, _expected_nli = osnr_mod.calculate_osnr(
            local_env.env, first_service, local_env.env.qot_constraint
        )

        assert math.isclose(first_service.OSNR, first_osnr_before, abs_tol=1e-6)
        assert not math.isclose(first_service.OSNR, expected_osnr, abs_tol=1e-6)
    finally:
        local_env.close()
