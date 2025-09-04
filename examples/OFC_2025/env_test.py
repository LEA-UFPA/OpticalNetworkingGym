#!/usr/bin/env python3
# debug_qrmsa_verbose.py

import os
import csv
import time
import random
import copy
import numpy as np
import gymnasium as gym

from datetime import datetime
from typing import Tuple

# Imports do optical-networking-gym
from optical_networking_gym.topology import Modulation, get_topology
from optical_networking_gym.wrappers.qrmsa_gym import QRMSAEnvWrapper
from optical_networking_gym.heuristics.heuristics import shortest_available_path_first_fit_best_modulation
from optical_networking_gym.core.osnr import calculate_osnr


# 1) Definição das modulações
def define_modulations() -> Tuple[Modulation, ...]:
    return (
        Modulation("QPSK",   200_000, 2, minimum_osnr=6.72, inband_xt=-17),
        Modulation("16QAM",    500, 4, minimum_osnr=13.24, inband_xt=-23),
    )


def create_environment():
    topology_name = "nobel-eu"
    topo_file = os.path.join("examples", "topologies", f"{topology_name}.xml")
    mods = define_modulations()
    topology = get_topology(
        topo_file, topology_name, mods,
        max_span_length=80, default_attenuation=0.2,
        default_noise_figure=4.5, k_paths=5
    )
    seed = 10
    random.seed(seed)
    np.random.seed(seed)

    env_args = dict(
        topology=topology,
        seed=seed,
        allow_rejection=True,
        load=300,
        episode_length=10,
        num_spectrum_resources=10,
        launch_power_dbm=0,
        frequency_slot_bandwidth=12.5e9,
        frequency_start=3e8/1565e-9,
        bandwidth=10*12.5e9,
        bit_rate_selection="discrete",
        bit_rates=(10,40),
        margin=0,
        measure_disruptions=False,
        file_name="",
        k_paths=3,
        modulations_to_consider=4,
        defragmentation=False,
        n_defrag_services=0,
        gen_observation=True,
    )
    return topology, env_args


# 3) Função de debug
def debug_run():
    topology, env_args = create_environment()
    env = QRMSAEnvWrapper(**env_args)

    obs, info = env.reset()
    svc = env.env.current_service

    def assert_all_links_empty(env):
        n = env.env.num_spectrum_resources
        expected = "1" * n
        for u, v in env.env.topology.edges():
            idx = env.env.topology[u][v]["index"]
            slots = env.env.topology.graph["available_slots"][idx]
            s = "".join(map(str, slots.tolist()))
            assert s == expected, f"Link {u}->{v} not full: {s}"

    assert_all_links_empty(env)

    print("\n--- Serviço atual ---")
    print(f"ID={svc.service_id}, src={svc.source}->{svc.destination}, bit_rate={svc.bit_rate}")
    print("Máscara válida de ações (1=válido):", info["mask"].astype(int))

    # escolhe ação e mostra detalhes
    action = shortest_available_path_first_fit_best_modulation(info["mask"])
    print(f"\n>> Heurística selecionou ação {action}")

    path_i, mod_i, slot_i = env.env.encoded_decimal_to_array(action)
    print(f">> Decoded action: path={path_i}, mod={mod_i}, slot={slot_i}")

    path = env.env.k_shortest_paths[svc.source, svc.destination][path_i]
    modulation = env.env.modulations[mod_i]
    num_slots = env.env.get_number_slots(svc, modulation)
    available = env.env.get_available_slots(path)
    cand = env.env._get_candidates(available, num_slots, env.env.num_spectrum_resources)

    print("\n--- Cálculo de recursos ---")
    print(f"Slots requeridos: {num_slots}")
    print(f"Slots disponíveis na rota: {available.tolist()[:50]}...")  # print parcial
    print(f"Candidatos válidos (inicial slots): {cand}")

    # antes da alocação, mostra links envolvidos
    print("\n=== LINKS ANTES DA ALOCAÇÃO ===")
    for i, (u,v) in enumerate(zip(path.node_list, path.node_list[1:])):
        idx = env.env.topology[u][v]["index"]
        slots = env.env.topology.graph["available_slots"][idx]
        print(f" {u}->{v} slots: {''.join(map(str, slots.tolist()))[:50]}...")

    # configura serviço para calcular OSNR
    svc.path = path
    svc.initial_slot = slot_i
    svc.number_slots = num_slots
    svc.center_frequency = (
        env.env.frequency_start +
        env.env.frequency_slot_bandwidth*(slot_i + num_slots/2)
    )
    svc.bandwidth = env.env.frequency_slot_bandwidth * num_slots
    svc.launch_power = env.env.launch_power

    osnr, ase, nli = calculate_osnr(env.env, svc)
    print(f"\n--- Cálculo de OSNR ---")
    print(f"OSNR = {osnr:.2f}, ASE = {ase:.2f}, NLI = {nli:.2f}")
    print(f"Relação mínima exigida = {modulation.minimum_osnr + env.env.margin:.2f}")

    # executa o step
    obs2, reward, done, trunc, info2 = env.step(action)
    print(f"\n>> Step(action={action}) => reward={reward:.2f}, done={done}")

    # depois da alocação, mostra links envolvidos
    print("\n=== LINKS DEPOIS DA ALOCAÇÃO ===")
    for u, v in zip(path.node_list, path.node_list[1:]):
        idx = env.env.topology[u][v]["index"]
        slots = env.env.topology.graph["available_slots"][idx]
        running_services = env.env.topology[u][v]["running_services"]
        print(f" {u}->{v} slots: {''.join(map(str,slots.tolist()))[:50]},  #services={[s.service_id for s in running_services]}")

    print("\n>> Nova máscara:", info2["mask"].astype(int))

    test_allocation_action_0(env)
    test_allocation_action_10(env)
    test_reject_action(env)
    test_action_encoding(env)
    test_get_available_slots_intersection(env)
    test_release_path(env)
    test_candidates_basic(env)
    test_multiple_steps(env)
    test_full_mask_validation(env, steps=10)
    print("\n=== RESUMO ===\nTodos os testes adicionais executados.")


def test_allocation_action_0(env):
    """Testa a alocação com action=0."""
    action = 0
    
    # Realiza reset e obtém novo serviço
    obs, info = env.reset()
    svc = env.env.current_service
    
    # Decodifica a ação para obter path, modulation e slot corretos
    path_i, mod_i, slot_i = env.env.encoded_decimal_to_array(action)
    
    # Obtém o path e modulation corretos
    path = env.env.k_shortest_paths[svc.source, svc.destination][path_i]
    modulation = env.env.modulations[mod_i]
    num_slots = env.env.get_number_slots(svc, modulation)

    print(f"\n=== TESTE ACTION=0 ===")
    print(f"Serviço: {svc.source}->{svc.destination}, bit_rate={svc.bit_rate}")
    print(f"Action {action} decodificado: path={path_i}, mod={mod_i}, slot={slot_i}")
    print(f"Slots necessários: {num_slots}")

    # Executa a ação
    obs, reward, done, trunc, info = env.step(action)

    # Verifica se a alocação foi realizada corretamente apenas se foi aceita
    if reward >= 0:  # Serviço foi aceito
        for u, v in zip(path.node_list, path.node_list[1:]):
            idx = env.env.topology[u][v]["index"]
            slots = env.env.topology.graph["available_slots"][idx]
            expected_slots = slots[slot_i:slot_i+num_slots]
            assert all(expected_slots == 0), f"Slots não alocados corretamente para link {u}->{v}: slots[{slot_i}:{slot_i+num_slots}] = {expected_slots}"
        print("✓ Alocação verificada com sucesso!")
    else:
        print("⚠ Serviço foi rejeitado - não verificando alocação")


def test_allocation_action_10(env):
    """Testa a alocação com action=10."""
    action = 10
    
    # Realiza reset e obtém novo serviço
    obs, info = env.reset()
    svc = env.env.current_service
    
    # Verifica se a ação 10 é válida
    if not info["mask"][action]:
        print(f"\n=== TESTE ACTION=10 ===")
        print("⚠ Ação 10 não é válida para este serviço - pulando teste")
        return
    
    # Decodifica a ação para obter path, modulation e slot corretos
    path_i, mod_i, slot_i = env.env.encoded_decimal_to_array(action)
    
    # Obtém o path e modulation corretos
    path = env.env.k_shortest_paths[svc.source, svc.destination][path_i]
    modulation = env.env.modulations[mod_i]
    num_slots = env.env.get_number_slots(svc, modulation)

    print(f"\n=== TESTE ACTION=10 ===")
    print(f"Serviço: {svc.source}->{svc.destination}, bit_rate={svc.bit_rate}")
    print(f"Action {action} decodificado: path={path_i}, mod={mod_i}, slot={slot_i}")
    print(f"Slots necessários: {num_slots}")

    # Executa a ação
    obs, reward, done, trunc, info = env.step(action)

    # Verifica se a alocação foi realizada corretamente apenas se foi aceita
    if reward >= 0:  # Serviço foi aceito
        for u, v in zip(path.node_list, path.node_list[1:]):
            idx = env.env.topology[u][v]["index"]
            slots = env.env.topology.graph["available_slots"][idx]
            expected_slots = slots[slot_i:slot_i+num_slots]
            assert all(expected_slots == 0), f"Slots não alocados corretamente para link {u}->{v}: slots[{slot_i}:{slot_i+num_slots}] = {expected_slots}"
        print("✓ Alocação verificada com sucesso!")
    else:
        print("⚠ Serviço foi rejeitado - não verificando alocação")


def test_reject_action(env):
    """Testa a ação de rejeição (último índice)."""
    obs, info = env.reset()
    reject_action = env.env.action_space.n - 1
    assert info["mask"][reject_action] == 1, "Máscara não marcou rejeição como válida"
    _obs2, reward, done, trunc, info2 = env.step(reject_action)
    assert reward < 0, "Rejeição deveria gerar recompensa negativa"
    print("✓ Ação de rejeição testada")


def test_action_encoding(env):
    """Testa consistência de encoded_decimal_to_array para diversas ações válidas."""
    obs, info = env.reset()
    max_check = min(15, env.env.action_space.n - 2)  # evita percorrer espaço grande
    for a in range(max_check):
        if info["mask"][a] == 0:
            continue
        arr = env.env.encoded_decimal_to_array(a)
        assert 0 <= arr[0] < env.env.k_paths, "k_path fora do intervalo"
        assert 0 <= arr[1] < len(env.env.modulations), "mod fora do intervalo"
        assert 0 <= arr[2] < env.env.num_spectrum_resources, "slot fora do intervalo"
    print("✓ Encoding de ações testado (amostragem)")


def test_get_available_slots_intersection(env):
    """Força ocupação em um link e verifica interseção (get_available_slots)."""
    obs, info = env.reset()
    svc = env.env.current_service
    path = env.env.k_shortest_paths[svc.source, svc.destination][0]
    # Marca slot 0 ocupado no primeiro link
    n1, n2 = path.node_list[0], path.node_list[1]
    idx = env.env.topology[n1][n2]["index"]
    env.env.topology.graph["available_slots"][idx, 0] = 0
    inter = env.env.get_available_slots(path)
    assert inter[0] in (0, 1), "Valor inesperado em interseção"
    # Como só um link alterado, se ele ficou 0 deve propagar
    if env.env.topology.graph["available_slots"][idx, 0] == 0:
        assert inter[0] == 0, "Interseção não refletiu ocupação"
    print("✓ Interseção de slots testada")


def test_release_path(env):
    """Provisiona manualmente e testa _release_path."""
    obs, info = env.reset()
    svc = env.env.current_service  # serviço que será tentado alocar
    # Escolhe primeira ação válida não-rejeição
    reject_action = env.env.action_space.n - 1
    valid_actions = [i for i,v in enumerate(info["mask"][:-1]) if v==1]
    if not valid_actions:
        print("⚠ Nenhuma ação válida para testar release")
        return
    action = valid_actions[0]
    # Decodifica para saber slot e path
    path_i, mod_i, slot_i = env.env.encoded_decimal_to_array(action)
    path = env.env.k_shortest_paths[svc.source, svc.destination][path_i]
    modulation = env.env.modulations[mod_i]
    slots_needed = env.env.get_number_slots(svc, modulation)
    allocated_service = svc  # guarda referência antes do step
    _obs2, reward, done, trunc, info2 = env.step(action)
    if reward < 0:
        print("⚠ Serviço rejeitado; não é possível testar release")
        return
    # Verificações pós-provisionamento antes de liberar
    n1, n2 = path.node_list[0], path.node_list[1]
    idx = env.env.topology[n1][n2]["index"]
    before_release = env.env.topology.graph["available_slots"][idx, slot_i:slot_i+slots_needed].copy()
    assert all(before_release == 0), "Slots deveriam estar ocupados antes do release"
    # Libera usando o serviço alocado (não o current_service, que já avançou)
    env.env._release_path(allocated_service)
    after_release = env.env.topology.graph["available_slots"][idx, slot_i:slot_i+slots_needed]
    assert all(after_release == 1), "Slots não foram liberados corretamente"
    print("✓ Release de path testado")


def test_candidates_basic(env):
    """Testa _get_candidates com vetor sintético."""
    # Vetor: 1 1 0 1 1 1 0 0 1 1 1 1
    avail = np.array([1,1,0,1,1,1,0,0,1,1,1,1], dtype=int)
    num_slots = 2
    cand = env.env._get_candidates(avail, num_slots, len(avail))
    assert isinstance(cand, list), "Retorno deve ser lista"
    # Garantir que inícios óbvios presentes (0 e 3 e 9 podem depender de guarda)
    assert any(c in cand for c in (0,3,9)), "Candidatos esperados ausentes"
    print("✓ _get_candidates testado")


def test_multiple_steps(env):
    """Executa várias alocações até receber rejeição ou preencher alguns slots."""
    obs, info = env.reset()
    steps = 0
    accepted = 0
    while steps < 5:
        mask = info["mask"]
        # Escolhe primeira ação válida não-rejeição
        reject_action = env.env.action_space.n - 1
        valid_actions = [i for i,v in enumerate(mask[:-1]) if v==1]
        if not valid_actions:
            break
        action = valid_actions[0]
        obs, reward, done, trunc, info = env.step(action)
        if reward >= 0:
            accepted += 1
        steps += 1
        if done:
            break
    assert steps > 0, "Nenhum step executado"
    assert accepted >= 0, "Contador de aceitos inválido"
    print(f"✓ Execução múltipla de steps ({steps} steps, {accepted} aceitos)")


def test_full_mask_validation(env, steps=10):
    """Valida toda a máscara de ações por N steps.

    Para cada step:
      - Para cada ação (exceto rejeição):
          * Se mask[a]==1 então (slots livres & OSNR >= threshold) devem ser verdade.
          * Se mask[a]==0 então NÃO pode ocorrer (slots livres & OSNR >= threshold) simultaneamente.
      - Executa a primeira ação válida (ou rejeição se nenhuma) para avançar.
    """
    print("\n=== TESTE FULL MASK (10 steps) ===")
    obs, info = env.reset()
    reject_action = env.env.action_space.n - 1
    for step_i in range(steps):
        mask = info["mask"]
        # Rejeição sempre deve ser 1
        assert mask[reject_action] == 1, "Ação de rejeição deve ser sempre válida na máscara"

        svc = env.env.current_service
        source, destination = svc.source, svc.destination
        errors = []
        # Percorre todas as ações exceto rejeição
        for action in range(env.env.action_space.n - 1):
            # Decodifica
            path_i, mod_i, slot_i = env.env.encoded_decimal_to_array(action)
            # Proteção contra índices fora (caso espaço maior que k_paths * mods * slots)
            if path_i >= env.env.k_paths or mod_i >= len(env.env.modulations) or slot_i >= env.env.num_spectrum_resources:
                # Ação não mapeada para combinação real: deve estar 0 na máscara
                if mask[action] == 1:
                    errors.append(f"Action {action}: decodificação inválida mas máscara=1")
                continue
            path = env.env.k_shortest_paths[source, destination][path_i]
            modulation = env.env.modulations[mod_i]
            num_slots = env.env.get_number_slots(svc, modulation)
            if num_slots <= 0:
                # Se slots necessários <=0 não devemos permitir ação
                if mask[action] == 1:
                    errors.append(f"Action {action}: num_slots<=0 mas máscara=1")
                continue
            available = env.env.get_available_slots(path)
            candidates = env.env._get_candidates(available, num_slots, env.env.num_spectrum_resources)
            resource_valid = slot_i in candidates
            # Simula serviço (cópia) para OSNR
            service_copy = copy.deepcopy(svc)
            service_copy.path = path
            service_copy.initial_slot = slot_i
            service_copy.number_slots = num_slots
            service_copy.center_frequency = (
                env.env.frequency_start +
                (env.env.frequency_slot_bandwidth * slot_i) +
                (env.env.frequency_slot_bandwidth * (num_slots / 2))
            )
            service_copy.bandwidth = env.env.frequency_slot_bandwidth * num_slots
            service_copy.launch_power = env.env.launch_power
            service_copy.current_modulation = modulation
            osnr, _, _ = calculate_osnr(env.env, service_copy)
            osnr_valid = osnr >= (modulation.minimum_osnr + env.env.margin)
            both_valid = resource_valid and osnr_valid
            if mask[action] == 1 and not both_valid:
                errors.append(
                    f"Action {action} marcada 1 mas inválida (resource={resource_valid}, osnr={osnr_valid}, osnr={osnr:.2f}, mod={modulation.name})"
                )
            if mask[action] == 0 and both_valid:
                errors.append(
                    f"Action {action} marcada 0 mas seria válida (osnr={osnr:.2f}, mod={modulation.name})"
                )
        assert not errors, "Inconsistências na máscara:\n" + "\n".join(errors[:10]) + ("\n..." if len(errors) > 10 else "")
        # Avança ambiente: escolhe primeira ação válida (ou rejeição)
        valid_non_reject = [a for a in range(env.env.action_space.n - 1) if mask[a] == 1]
        if valid_non_reject:
            chosen = valid_non_reject[0]
        else:
            chosen = reject_action
        obs, reward, done, trunc, info = env.step(chosen)
        if done:
            # Reinicia se episódio acabou antes dos 10 steps
            if step_i < steps - 1:
                obs, info = env.reset()
        print(f" Step {step_i+1}: máscara validada, ação tomada={chosen}, reward={reward:.2f}")
    print("✓ Máscara completa validada por", steps, "steps")


if __name__ == "__main__":
    debug_run()
