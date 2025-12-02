import math
from typing import Optional
import numpy as np
from optical_networking_gym.topology import Modulation, Path
from optical_networking_gym.utils import rle, link_shannon_entropy_, fragmentation_route_cuts, fragmentation_route_rss
from optical_networking_gym.core.osnr import calculate_osnr
from optical_networking_gym.envs.qrmsa import QRMSAEnv
from gymnasium import Env
import pyswarms as ps

from gymnasium import Env
from optical_networking_gym.envs.qrmsa import QRMSAEnv

def get_qrmsa_env(env: Env) -> QRMSAEnv:
    """
    Percorre os wrappers do ambiente até encontrar a instância base de QRMSAEnv.

    Args:
        env (gym.Env): O ambiente potencialmente envolvido em múltiplos wrappers.

    Returns:
        QRMSAEnv: A instância base de QRMSAEnv.

    Raises:
        ValueError: Se QRMSAEnv não for encontrado na cadeia de wrappers.
    """
    while not isinstance(env, QRMSAEnv):
        if hasattr(env, 'env'):
            env = env.env
        else:
            raise ValueError("QRMSAEnv não foi encontrado na cadeia de wrappers do ambiente.")
    return env


def get_action_index(env: QRMSAEnv, path_index: int, modulation_index: int, initial_slot: int) -> int:
    """
    Converte (path_index, modulation_index, initial_slot) em um índice de ação inteiro.
    Para ambientes multiband, usa get_multiband_action_index.
    
    Args:
        env (QRMSAEnv): O ambiente QRMSAEnv.
        path_index (int): Índice da rota.
        modulation_index (int): Índice absoluto da modulação.
        initial_slot (int): Slot inicial para alocação (global para multiband).
    
    Returns:
        int: Índice da ação correspondente.
    """
    # Verificar se é ambiente multiband
    if hasattr(env, 'bands') and env.bands:
        # Para multiband, precisa determinar a banda e slot relativo
        for band_idx, band in enumerate(env.bands):
            if band.contains_slot_range(initial_slot, 1):  # Verifica se o slot está na banda
                slot_in_band = band.global_to_local(initial_slot)
                return get_multiband_action_index(env, path_index, band_idx, modulation_index, slot_in_band)
        raise ValueError(f"Slot global {initial_slot} não pertence a nenhuma banda")
    else:
        # Ambiente single-band original
        relative_modulation_index = env.max_modulation_idx - modulation_index
        return (path_index * env.modulations_to_consider * env.num_spectrum_resources +
                relative_modulation_index * env.num_spectrum_resources +
                initial_slot)


def get_multiband_action_index(env: QRMSAEnv, path_index: int, band_index: int, 
                              modulation_index: int, slot_in_band: int) -> int:
    """
    Converte (path_index, band_index, modulation_index, slot_in_band) em um índice de ação 
    para ambiente multiband.
    
    Args:
        env (QRMSAEnv): O ambiente QRMSAEnv.
        path_index (int): Índice da rota.
        band_index (int): Índice da banda.
        modulation_index (int): Índice absoluto da modulação.
        slot_in_band (int): Slot relativo dentro da banda (não global).
    
    Returns:
        int: Índice da ação correspondente.
    """
    # Converter o índice absoluto da modulação para o relativo
    relative_modulation_index = env.max_modulation_idx - modulation_index
    
    # CORREÇÃO CRÍTICA: O ambiente assume que todas as bandas têm o mesmo número de slots
    # Para o cálculo da ação. Verificar se é realmente o caso ou usar o valor da primeira banda.
    # Baseado no código do ambiente (line 1166 e 1363 em qrmsa.pyx):
    slots_per_band = env.bands[0].num_slots if env.bands else env.num_spectrum_resources
    
    # IMPORTANTE: O ambiente decodifica assumindo slots_per_band uniformes, então precisamos
    # usar o mesmo valor aqui. Se a banda atual tem um número diferente de slots,
    # precisamos validar se slot_in_band está dentro do range da primeira banda.
    if slot_in_band >= slots_per_band:
        raise ValueError(f"slot_in_band {slot_in_band} excede slots_per_band {slots_per_band} "
                        f"para banda {band_index}. O ambiente assume bandas uniformes.")
    
    # Fórmula baseada na estrutura do action space:
    # action_space_size = (k_paths * num_bands * modulations_to_consider * slots_per_band) + 1
    action_index = (path_index * env.num_bands * env.modulations_to_consider * slots_per_band +
                   band_index * env.modulations_to_consider * slots_per_band +
                   relative_modulation_index * slots_per_band +
                   slot_in_band)
    
    return action_index

# def decimal_to_array(env: QRMSAEnv, decimal: int, max_values: list[int] = None) -> list[int]:
#     if max_values is None:
#         max_values = [env.k_paths, len(env.modulations), env.num_spectrum_resources]
    
#     array = []
#     for max_val in reversed(max_values):
#         array.insert(0, decimal % max_val)
#         decimal //= max_val
#     print(f"Decimal converted to array {array}")

#     # Mapeia o índice relativo de modulação para o índice absoluto
#     allowed_mods = list(range(env.max_modulation_idx, env.max_modulation_idx - env.modulations_to_consider, -1))
    
#     # O mapeamento de modulação acontece para o segundo índice (array[1])
#     array[1] = allowed_mods[array[1]]
#     print(f"Modulation index mapped to absolute index: {array}")
#     return array



def heuristic_from_mask(env: Env, mask: np.ndarray) -> int:
    print("========== Iniciando heuristic_from_mask ==========")
    total_actions = len(mask)
    print("Total de ações a testar:", total_actions)

    valid_actions = []
    errors = []

    for action_index in range(total_actions):
        # Ação de rejeição deve ser sempre válida na máscara
        if action_index // 31 == 0:
            print("a")

        if action_index == env.action_space.n - 1:
            assert mask[action_index] == 1, "Erro: Ação de rejeição deve ser sempre válida"
            continue

        # Decodifica a ação e imprime os detalhes
        decoded = env.encoded_decimal_to_array(action_index)
        print(action_index,": Decodificação da ação:", decoded)
        path_index, modulation_index, candidate_index = decoded

        # Cria o ambiente de simulação (supondo que get_qrmsa_env retorne uma instância fresca)
        qrmsa_env = get_qrmsa_env(env)
        
        # Seleciona o caminho de acordo com os k-shortest paths
        path = qrmsa_env.k_shortest_paths[
            qrmsa_env.current_service.source,
            qrmsa_env.current_service.destination
        ][path_index]

        # Verifica se a modulação obtida é a esperada
        modulation = qrmsa_env.modulations[modulation_index]
        interval = action_index // 320
        print("*" * 30)
        # Define as modulações esperadas para os dois casos:
        if env.max_modulation_idx > 1:
            # Caso padrão: usamos a modulação de índice max_modulation_idx (a "maior") para intervalos pares
            # e a modulação imediatamente inferior (índice max_modulation_idx - 1) para intervalos ímpares.
            higher_mod = qrmsa_env.modulations[env.max_modulation_idx]
            lower_mod  = qrmsa_env.modulations[env.max_modulation_idx - 1]
        else:
            # Se max_modulation_idx é 0, isto indica que a maior modulação permitida é a primeira da lista (por exemplo, 8QAM).
            # Então, para intervalos pares usamos a próxima modulação (índice 1, ex: 16QAM) e para ímpares a própria.
            higher_mod = qrmsa_env.modulations[1]
            lower_mod  = qrmsa_env.modulations[0]

        # print(f"Expected modulations:\n- Para intervalo par: {higher_mod}\n- Para intervalo ímpar: {lower_mod}")
        # print("*" * 30)

        # Seleciona a modulação esperada com base no valor de 'interval'
        if interval % 2 == 0:
            expected_modulation = higher_mod
        else:
            expected_modulation = lower_mod

        assert modulation == expected_modulation, (
            f"Erro: A modulação {modulation.name} não é a esperada {expected_modulation.name} para o índice {modulation_index}."
        )

        # Determina o número de slots, slots disponíveis e os candidatos válidos
        number_slots = qrmsa_env.get_number_slots(qrmsa_env.current_service, modulation)
        available_slots = qrmsa_env.get_available_slots(path)
        valid_starts = qrmsa_env._get_candidates(available_slots, number_slots, env.num_spectrum_resources)
        
        resource_valid = candidate_index in valid_starts
        if not resource_valid:
            assert mask[action_index] == 0, (
                f"Erro: Ação {action_index} bloqueada por recurso, mas marcada como válida na máscara."
            )
            continue

        # Configura os parâmetros do serviço atual para a simulação
        service = qrmsa_env.current_service
        service.path = path
        service.initial_slot = candidate_index
        service.number_slots = number_slots
        service.center_frequency = (
            qrmsa_env.frequency_start +
            (qrmsa_env.frequency_slot_bandwidth * candidate_index) +
            (qrmsa_env.frequency_slot_bandwidth * (number_slots / 2))
        )
        service.bandwidth = qrmsa_env.frequency_slot_bandwidth * number_slots
        service.launch_power = qrmsa_env.launch_power

        # Calcula o OSNR e valida
        osnr, _, _ = calculate_osnr(qrmsa_env, service)

        threshold = modulation.minimum_osnr + qrmsa_env.margin
        osnr_valid = osnr >= threshold
        if not osnr_valid:
            assert mask[action_index] == 0, (
                f"Erro: Ação {action_index} bloqueada por OSNR, mas marcada como válida na máscara."
            )
            continue

        # Caso a ação seja considerada válida pela simulação, registra a ação
        if mask[action_index] == 1 and not (resource_valid and osnr_valid):
            error_msg = (
                f"Erro: Ação {action_index} marcada como válida na máscara, mas simulação indica ação inválida "
                f"(Resource valid: {resource_valid}, OSNR valid: {osnr_valid}, OSNR: {osnr:.2f})."
            )
            print(error_msg)
            errors.append(error_msg)
        elif mask[action_index] == 0 and (resource_valid and osnr_valid):
            error_msg = (
                f"Erro: Ação {action_index} marcada como inválida na máscara, mas simulação indica ação válida "
                f"(Resource valid: {resource_valid}, OSNR valid: {osnr_valid}, OSNR: {osnr:.2f})."
            )
            print(error_msg)
            errors.append(error_msg)

        valid_actions.append(action_index)
        print(f"Ação {action_index} é considerada válida.")

    print("\n========== Resumo da validação ==========")
    if errors:
        print("Foram encontrados os seguintes erros:")
        for err in errors:
            print(" -", err)
    selected_action = np.random.choice(total_actions)
    return selected_action



def heuristic_shortest_available_path_first_fit_best_modulation(env: Env) -> int:
    blocked_due_to_resources, blocked_due_to_osnr = False, False
    sim_env = get_qrmsa_env(env)
    source = sim_env.current_service.source
    destination = sim_env.current_service.destination
    k_paths = sim_env.k_shortest_paths[source, destination]
    for path_idx, path in enumerate(k_paths):
        for modulation_idx in range(sim_env.max_modulation_idx, -1, -1):
            modulation = sim_env.modulations[modulation_idx]

            required_slots = sim_env.get_number_slots(sim_env.current_service, modulation)
            if required_slots <= 0:
                continue 
            available_slots = sim_env.get_available_slots(path)
            valid_starts = sim_env._get_candidates(available_slots, required_slots, sim_env.num_spectrum_resources)
            
            if not valid_starts:
                blocked_due_to_resources = True
                continue 
            candidate_start = valid_starts[0]
            service = sim_env.current_service
            service.path = path
            service.initial_slot = candidate_start
            service.number_slots = required_slots
            service.current_modulation = modulation
            
            # Definir banda atual para ambientes multibanda
            if hasattr(sim_env, 'bands') and sim_env.bands:
                # Encontrar qual banda contém o slot inicial
                for band in sim_env.bands:
                    if band.contains_slot_range(candidate_start, required_slots):
                        service.current_band = band
                        service.center_frequency = band.center_frequency_hz_from_global(
                            candidate_start, required_slots
                        )
                        break
                else:
                    # Se nenhuma banda contém o slot, usar a primeira banda como fallback
                    service.current_band = sim_env.bands[0]
                    service.center_frequency = (
                        sim_env.frequency_start +
                        (sim_env.frequency_slot_bandwidth * candidate_start) +
                        (sim_env.frequency_slot_bandwidth * (required_slots / 2)))
            else:
                # Ambiente single-band tradicional
                service.current_band = None
                service.center_frequency = (
                        sim_env.frequency_start +
                        (sim_env.frequency_slot_bandwidth * candidate_start) +
                        (sim_env.frequency_slot_bandwidth * (required_slots / 2)))
            
            service.bandwidth = sim_env.frequency_slot_bandwidth * required_slots
            service.launch_power = sim_env.launch_power
            
            osnr, _, _ = calculate_osnr(sim_env, service)
            
            threshold = modulation.minimum_osnr + sim_env.margin
            if osnr >= threshold:
                action_index = get_action_index(sim_env, path_idx, modulation_idx, candidate_start)
                return action_index, False, False 
            else:
                blocked_due_to_osnr = True
                if blocked_due_to_resources:
                    blocked_due_to_resources = False
    
    return env.action_space.n - 1, blocked_due_to_resources, blocked_due_to_osnr 


def heuristic_highest_snr(env: Env) -> int:
    best_osnr = -np.inf
    best_action = None
    any_blocked_resources = False
    any_blocked_osnr = False

    sim_env = get_qrmsa_env(env)
    source = sim_env.current_service.source
    destination = sim_env.current_service.destination
    k_paths = sim_env.k_shortest_paths[source, destination]

    for path_idx, path in enumerate(k_paths):
        for modulation_idx in range(sim_env.max_modulation_idx, -1, -1):
            modulation = sim_env.modulations[modulation_idx]

            required_slots = sim_env.get_number_slots(sim_env.current_service, modulation)
            if required_slots <= 0:
                continue

            available_slots = sim_env.get_available_slots(path)
            valid_starts = sim_env._get_candidates(available_slots, required_slots, sim_env.num_spectrum_resources)

            if not valid_starts:
                any_blocked_resources = True
                continue

            for candidate_start in valid_starts:
                service = sim_env.current_service
                service.path = path
                service.initial_slot = candidate_start
                service.number_slots = required_slots
                service.current_modulation = modulation
                service.center_frequency = (
                    sim_env.frequency_start +
                    (sim_env.frequency_slot_bandwidth * candidate_start) +
                    (sim_env.frequency_slot_bandwidth * (required_slots / 2))
                )
                service.bandwidth = sim_env.frequency_slot_bandwidth * required_slots
                service.launch_power = sim_env.launch_power

                osnr, _, _ = calculate_osnr(sim_env, service)
                threshold = modulation.minimum_osnr + sim_env.margin

                if osnr >= threshold:
                    if osnr > best_osnr or (osnr == best_osnr and best_action is None):
                        action_index = get_action_index(sim_env, path_idx, modulation_idx, candidate_start)
                        best_osnr = osnr
                        best_action = action_index
                else:
                    any_blocked_osnr = True

    if best_action is not None:
        return best_action, False, False 
    else:
        if any_blocked_osnr:
            any_blocked_resources = False
        return env.action_space.n - 1, any_blocked_resources, any_blocked_osnr

def heuristic_lowest_fragmentation(env: Env) -> tuple[int, bool, bool]:
    sim_env = get_qrmsa_env(env)
    service = sim_env.current_service
    source, destination = service.source, service.destination
    k_paths = sim_env.k_shortest_paths[source, destination]

    best_score = math.inf
    best_action = None
    any_blocked_resources = False
    any_blocked_osnr       = False

    for path_idx, path in enumerate(k_paths):
        raw = sim_env._get_spectrum_slots(path_idx)

        if isinstance(raw, list):
            base_spectra = np.stack(raw, axis=0)
        else:
            arr = np.array(raw)
            if arr.ndim == 1:
                base_spectra = arr[None, :]    # (1, n_slots)
            elif arr.ndim == 2:
                base_spectra = arr
            else:
                raise ValueError(f"Formato inesperado em _get_spectrum_slots: ndim={arr.ndim}")

        for mod_idx in range(sim_env.max_modulation_idx, -1, -1):
            modulation = sim_env.modulations[mod_idx]
            req_slots  = sim_env.get_number_slots(service, modulation)+1
            if req_slots <= 0:
                continue

            available = sim_env.get_available_slots(path)
            candidates = sim_env._get_candidates(
                available, req_slots, sim_env.num_spectrum_resources
            )
            if not candidates:
                any_blocked_resources = True
                continue

            for start in candidates:
                # 3) simula a alocação: clone e pinta o bloco como 1=ocupado
                tmp = base_spectra.copy()
                tmp[:, start:start+req_slots] = 1

                # 4) calcula métricas de fragmentação
                # 4a) entropia média
                entropies = [ link_shannon_entropy_(row.tolist()) for row in tmp ]
                se   = sum(entropies) / len(entropies) if entropies else 0.0
                # 4b) cuts
                cuts = fragmentation_route_cuts([row.tolist() for row in tmp])
                # 4c) rss
                rss  = fragmentation_route_rss([row.tolist() for row in tmp])

                score = 0.33 * se + 0.33 * cuts + 0.34 * rss

                # 5) verifica OSNR
                service.path             = path
                service.initial_slot     = start
                service.number_slots     = req_slots
                service.current_modulation = modulation
                service.center_frequency = (
                    sim_env.frequency_start
                    + sim_env.frequency_slot_bandwidth * start
                    + sim_env.frequency_slot_bandwidth * (req_slots / 2)
                )
                service.bandwidth    = sim_env.frequency_slot_bandwidth * req_slots
                service.launch_power = sim_env.launch_power

                osnr, _, _ = calculate_osnr(sim_env, service)
                if osnr < modulation.minimum_osnr + sim_env.margin:
                    any_blocked_osnr = True
                    continue

                # 6) escolhe o candidato de menor fragmentação
                if score < best_score:
                    best_score  = score
                    best_action = get_action_index(sim_env, path_idx, mod_idx, start)

    # 7) retorna conforme convenção
    if best_action is not None:
        return best_action, False, False
    else:
        if any_blocked_osnr:
            any_blocked_resources = False
        return env.action_space.n - 1, any_blocked_resources, any_blocked_osnr




def shortest_available_path_first_fit_best_modulation(
    mask: np.ndarray,
) -> Optional[int]:
    return int(np.where(mask == 1)[0][0])

def rnd(
    mask: np.ndarray,
) -> Optional[int]:
    valid_actions = np.where(mask == 1)[0]
    return int(np.random.choice(valid_actions))


def shortest_available_path_lowest_spectrum_best_modulation(
    env: Env,
) -> Optional[int]:
    """
    Seleciona a rota mais curta disponível com a menor utilização espectral e a melhor modulação.
    
    Args:
        env (gym.Env): O ambiente potencialmente envolvido em wrappers.
    
    Returns:
        Optional[int]: Índice da ação correspondente, ou a ação de rejeição se permitido, ou None.
    """
    qrmsa_env: QRMSAEnv = get_qrmsa_env(env) 

    for idp, path in enumerate(qrmsa_env.k_shortest_paths[
        qrmsa_env.current_service.source,
        qrmsa_env.current_service.destination,
    ]):
        available_slots = qrmsa_env.get_available_slots(path)
        for idm, modulation in zip(
            range(len(qrmsa_env.modulations) - 1, -1, -1),
            reversed(qrmsa_env.modulations)
        ):
            number_slots = qrmsa_env.get_number_slots(qrmsa_env.current_service, modulation) + 2 

            initial_indices, values, lengths = rle(available_slots)
            sufficient_indices = np.where(lengths >= number_slots)
            available_indices = np.where(values == 1)
            final_indices = np.intersect1d(available_indices, sufficient_indices)

            if final_indices.size > 0:
                qrmsa_env.current_service.blocked_due_to_resources = False
                initial_slot = initial_indices[final_indices][0]
                if initial_slot > 0:
                    initial_slot += 1  # guard band

                # Atualizar parâmetros do serviço
                qrmsa_env.current_service.path = path
                qrmsa_env.current_service.initial_slot = initial_slot
                qrmsa_env.current_service.number_slots = number_slots
                qrmsa_env.current_service.center_frequency = (
                    qrmsa_env.frequency_start +
                    (qrmsa_env.frequency_slot_bandwidth * initial_slot) +
                    (qrmsa_env.frequency_slot_bandwidth * (number_slots / 2))
                )
                qrmsa_env.current_service.bandwidth = qrmsa_env.frequency_slot_bandwidth * number_slots
                qrmsa_env.current_service.launch_power = qrmsa_env.launch_power

                # Calcular OSNR
                osnr, _, _ = calculate_osnr(qrmsa_env, qrmsa_env.current_service)
                if osnr >= modulation.minimum_osnr + qrmsa_env.margin:
                    # Converter para índice de ação
                    action = get_action_index(qrmsa_env, idp, idm, initial_slot)
                    return action, False, False  # ou uma ação padrão específica

    # Se nenhuma ação válida encontrada, retornar a ação de rejeição se permitido
    return qrmsa_env.reject_action, False, False  # ou uma ação padrão específica

def best_modulation_load_balancing(
    env: Env,
) -> Optional[int]:
    """
    Balanceia a carga selecionando a melhor modulação e minimizando a carga na rota.
    
    Args:
        env (gym.Env): O ambiente potencialmente envolvido em wrappers.
    
    Returns:
        Optional[int]: Índice da ação correspondente, ou a ação de rejeição se permitido, ou None.
    """
    qrmsa_env: QRMSAEnv = get_qrmsa_env(env)  # Descompactar o ambiente
    solution = None
    lowest_load = float('inf')

    for idm, modulation in zip(
        range(len(qrmsa_env.modulations) - 1, -1, -1),
        reversed(qrmsa_env.modulations)
    ):
        for idp, path in enumerate(qrmsa_env.k_shortest_paths[
            qrmsa_env.current_service.source,
            qrmsa_env.current_service.destination,
        ]):
            available_slots = qrmsa_env.get_available_slots(path)
            number_slots = qrmsa_env.get_number_slots(qrmsa_env.current_service, modulation)  # +2 para guard band

            initial_indices, values, lengths = rle(available_slots)
            sufficient_indices = np.where(lengths >= number_slots+1)
            available_indices = np.where(values == 1)
            final_indices = np.intersect1d(available_indices, sufficient_indices)

            if final_indices.size > 0:
                qrmsa_env.current_service.blocked_due_to_resources = False
                initial_slot = initial_indices[final_indices][0]

                # Atualizar parâmetros do serviço
                qrmsa_env.current_service.path = path
                qrmsa_env.current_service.initial_slot = initial_slot
                qrmsa_env.current_service.number_slots = number_slots
                qrmsa_env.current_service.center_frequency = qrmsa_env.frequency_start + \
                    (qrmsa_env.frequency_slot_bandwidth * initial_slot) + \
                    (qrmsa_env.frequency_slot_bandwidth * (number_slots / 2))
                qrmsa_env.current_service.bandwidth = qrmsa_env.frequency_slot_bandwidth * number_slots
                qrmsa_env.current_service.launch_power = qrmsa_env.launch_power

                # Calcular OSNR
                osnr, _, _ = calculate_osnr(qrmsa_env, qrmsa_env.current_service)
                if osnr >= modulation.minimum_osnr + qrmsa_env.margin:
                    # Converter para índice de ação
                    action = get_action_index(qrmsa_env, idp, idm, initial_slot)
                    return action, False, False  # ou uma ação padrão específica

    # Se nenhuma ação válida encontrada, retornar a ação de rejeição se permitido
    return qrmsa_env.reject_action, False, False

def load_balancing_best_modulation(
    env: Env,
) -> Optional[int]:
    """
    Balanceia a carga selecionando a melhor modulação com a menor carga na rota.
    
    Args:
        env (gym.Env): O ambiente potencialmente envolvido em wrappers.
    
    Returns:
        Optional[int]: Índice da ação correspondente, ou a ação de rejeição se permitido, ou None.
    """
    qrmsa_env: QRMSAEnv = get_qrmsa_env(env)  # Descompactar o ambiente
    solution = None
    lowest_load = float('inf')

    for idp, path in enumerate(qrmsa_env.k_shortest_paths[
        qrmsa_env.current_service.source,
        qrmsa_env.current_service.destination,
    ]):
        available_slots = qrmsa_env.get_available_slots(path)
        current_load = available_slots.sum() / np.sqrt(len(path.links))
        if current_load >= lowest_load:
            continue  # não é uma rota melhor

        for idm, modulation in zip(
            range(len(qrmsa_env.modulations) - 1, -1, -1),
            reversed(qrmsa_env.modulations)
        ):
            number_slots = qrmsa_env.get_number_slots(qrmsa_env.current_service, modulation) + 2  # +2 para guard band

            initial_indices, values, lengths = rle(available_slots)
            sufficient_indices = np.where(lengths >= number_slots)
            available_indices = np.where(values == 1)
            final_indices = np.intersect1d(available_indices, sufficient_indices)

            if final_indices.size > 0:
                qrmsa_env.current_service.blocked_due_to_resources = False
                initial_slot = initial_indices[final_indices][0]
                if initial_slot > 0:
                    initial_slot += 1  # guard band

                # Atualizar parâmetros do serviço
                qrmsa_env.current_service.path = path
                qrmsa_env.current_service.initial_slot = initial_slot
                qrmsa_env.current_service.number_slots = number_slots
                qrmsa_env.current_service.center_frequency = (
                    qrmsa_env.frequency_start +
                    (qrmsa_env.frequency_slot_bandwidth * initial_slot) +
                    (qrmsa_env.frequency_slot_bandwidth * (number_slots / 2))
                )
                qrmsa_env.current_service.bandwidth = qrmsa_env.frequency_slot_bandwidth * number_slots
                qrmsa_env.current_service.launch_power = qrmsa_env.launch_power

                # Calcular OSNR
                osnr, _, _ = calculate_osnr(qrmsa_env, qrmsa_env.current_service)
                if osnr >= modulation.minimum_osnr + qrmsa_env.margin and current_load < lowest_load:
                    lowest_load = current_load
                    solution = get_action_index(qrmsa_env, idp, idm, initial_slot)
                    break  # Mover para a próxima rota após encontrar uma modulação melhor

    # Retornar a melhor solução encontrada
    if solution is not None:
        return solution, False, False

    return qrmsa_env.reject_action, False, False  # ou uma ação padrão específica


def _get_largest_contiguous_block(available_slots: np.ndarray) -> int:
    """Função auxiliar para encontrar o maior bloco contíguo de slots livres."""
    if not np.any(available_slots):
        return 0
    
    initial_indices, values, lengths = rle(available_slots)
    
    max_len = 0
    for i in range(len(values)):
        if values[i] == 1 and lengths[i] > max_len:
            max_len = lengths[i]
            
    return max_len

import copy

def heuristic_mscl_sequential_simplified(env: Env) -> tuple[int, bool, bool]:
    """
    Implementa a heurística MSCL Sequencial de forma otimizada para testes.

    A heurística itera pelas rotas em sequência. Para a primeira rota que tiver
    recursos, ela encontra a melhor alocação (baseada em fragmentação mínima)
    APENAS NESSA ROTA e retorna a ação imediatamente.
    """
    sim_env = get_qrmsa_env(env)
    current_service = sim_env.current_service

    any_blocked_resources = False
    any_blocked_osnr = False

    # Itera pelas rotas candidatas em sequência
    for path_idx, path in enumerate(sim_env.k_shortest_paths[current_service.source, current_service.destination]):

        best_score_for_this_path = -1
        best_action_for_this_path = None

        # Para a rota atual, encontra a melhor combinação de modulação e slot
        for mod_idx in range(sim_env.max_modulation_idx, -1, -1):
            modulation = sim_env.modulations[mod_idx]
            required_slots = sim_env.get_number_slots(current_service, modulation)

            if required_slots <= 0:
                continue

            available_slots = sim_env.get_available_slots(path)
            candidate_starts = sim_env._get_candidates(available_slots, required_slots, sim_env.num_spectrum_resources)

            if not candidate_starts:
                any_blocked_resources = True
                continue

            # Para manter a robustez, usamos o primeiro slot candidato (First-Fit)
            start_slot = candidate_starts[0]

            # Verifica o OSNR
            service_copy = copy.deepcopy(current_service)
            service_copy.path = path
            service_copy.initial_slot = start_slot
            service_copy.number_slots = required_slots
            service_copy.current_modulation = modulation
            service_copy.bandwidth = sim_env.frequency_slot_bandwidth * required_slots
            service_copy.launch_power = sim_env.launch_power
            service_copy.center_frequency = (
                sim_env.frequency_start +
                (sim_env.frequency_slot_bandwidth * start_slot) +
                (sim_env.frequency_slot_bandwidth * (required_slots / 2))
            )

            osnr, _, _ = calculate_osnr(sim_env, service_copy)

            if osnr >= modulation.minimum_osnr + sim_env.margin:
                # Calcula a pontuação de fragmentação para esta opção válida
                temp_slots = available_slots.copy()
                temp_slots[start_slot : start_slot + required_slots] = 0
                current_score = _get_largest_contiguous_block(temp_slots)

                # Se esta for a melhor opção encontrada ATÉ AGORA PARA ESTA ROTA
                if current_score > best_score_for_this_path:
                    best_score_for_this_path = current_score
                    best_action_for_this_path = get_action_index(sim_env, path_idx, mod_idx, start_slot)
            else:
                any_blocked_osnr = True

        # FIM DO LAÇO DE MODULAÇÕES
        # Se uma ação válida foi encontrada para esta rota, retorna-a imediatamente.
        if best_action_for_this_path is not None:
            return best_action_for_this_path, False, False

    # Se o laço terminar sem nenhuma solução encontrada em nenhuma rota
    if any_blocked_osnr and not any_blocked_resources:
        any_blocked_resources = False
    return sim_env.action_space.n - 1, any_blocked_resources, any_blocked_osnr

def heuristic_priority_band_C_then_L(env: Env) -> tuple[int, bool, bool]:
    """
    Tenta alocar o serviço na banda C enquanto a ocupação for < 90%.
    Se >= 90%, tenta na banda L.
    """
    sim_env = get_qrmsa_env(env)
    source = sim_env.current_service.source
    destination = sim_env.current_service.destination
    k_paths = sim_env.k_shortest_paths[source, destination]

    # Identifica os índices das bandas C e L
    band_C_idx = None
    band_L_idx = None
    for idx, band in enumerate(getattr(sim_env, "bands", getattr(sim_env, "_bands", []))):
        if band.name.upper() == "C":
            band_C_idx = idx
        elif band.name.upper() == "L":
            band_L_idx = idx

    def band_occupancy(band_idx):
        # Calcula ocupação da banda (slots ocupados / total de slots)
        total_slots = sim_env.bands[band_idx].num_slots
        occupied = 0
        for path in k_paths:
            slots = sim_env.get_available_slots(path)[sim_env.bands[band_idx].offset : sim_env.bands[band_idx].offset + total_slots]
            occupied += total_slots - np.sum(slots)
        # Média de ocupação entre todos os caminhos
        return occupied / (total_slots * len(k_paths))

    # Decide se tenta na banda C ou L
    if band_C_idx is not None and band_occupancy(band_C_idx) < 0.9:
        result = try_allocate_in_band(sim_env, k_paths, band_C_idx)
        if result is not None:
            return result

    # Se ocupação >= 90% ou não conseguiu, tenta na banda L
    if band_L_idx is not None:
        result = try_allocate_in_band(sim_env, k_paths, band_L_idx)
        if result is not None:
            return result

    # Se não conseguir em nenhuma, retorna ação de rejeição
    return env.action_space.n - 1, True, True

def try_allocate_in_band(sim_env, k_paths, band_idx):
    for path_idx, path in enumerate(k_paths):
        for modulation_idx in range(sim_env.max_modulation_idx, -1, -1):
            modulation = sim_env.modulations[modulation_idx]
            required_slots = sim_env.get_number_slots(sim_env.current_service, modulation)
            if required_slots <= 0:
                continue
            
            band = sim_env.bands[band_idx]
            available_slots = sim_env.get_available_slots(path)
            
            # Usar _get_candidates_in_band para obter candidatos válidos na banda
            try:
                valid_starts = sim_env._get_candidates_in_band(available_slots, required_slots, band)
            except Exception as e:
                print(f"Erro ao chamar _get_candidates_in_band: {e}")
                continue
                
            if not valid_starts:
                continue
            
            # O valid_starts já retorna slots globais
            global_candidate_start = valid_starts[0]
            
            # Verificar se o candidato global + required_slots está dentro da banda
            global_candidate_end = global_candidate_start + required_slots
            if not (band.slot_start <= global_candidate_start < band.slot_end and 
                    band.slot_start < global_candidate_end <= band.slot_end):
                continue
            
            # SIMPLIFICAÇÃO: usar diretamente o approach funcional do ambiente
            # Como o ambiente assume slots uniformes, mapear global_candidate_start 
            # para o slot relativo dentro da faixa da banda
            slot_in_band = global_candidate_start - band.slot_start
            
            # Verificar se o slot relativo está dentro dos limites da banda atual
            # CORREÇÃO: usar os limites da banda atual, não da primeira banda
            if slot_in_band < 0 or slot_in_band >= band.num_slots:
                continue
            
            # Verificação adicional: se ambiente assume slots uniformes, verificar se estamos dentro do limite uniforme
            slots_per_band = sim_env.bands[0].num_slots if sim_env.bands else sim_env.num_spectrum_resources
            if slot_in_band >= slots_per_band:
                continue
            
            service = sim_env.current_service
            service.path = path
            service.initial_slot = global_candidate_start
            service.number_slots = required_slots
            service.current_modulation = modulation
            service.current_band = band
            
            # Calcular frequência central usando a banda
            try:
                service.center_frequency = band.center_frequency_hz_from_global(
                    global_candidate_start, required_slots
                )
            except Exception as e:
                print(f"Erro ao calcular frequência central: {e}")
                # Fallback para cálculo tradicional
                service.center_frequency = (
                    sim_env.frequency_start +
                    (sim_env.frequency_slot_bandwidth * global_candidate_start) +
                    (sim_env.frequency_slot_bandwidth * (required_slots / 2))
                )
            
            service.bandwidth = sim_env.frequency_slot_bandwidth * required_slots
            service.launch_power = sim_env.launch_power
            
            # Verificar OSNR
            try:
                osnr, _, _ = calculate_osnr(sim_env, service)
                threshold = modulation.minimum_osnr + sim_env.margin
                
                if osnr >= threshold:
                    # Usar get_multiband_action_index diretamente
                    action_index = get_multiband_action_index(
                        sim_env, path_idx, band_idx, modulation_idx, slot_in_band
                    )
                    
                    
                    # Testar decodificação da ação para confirmar
                    try:
                        decoded = sim_env.decimal_to_array(action_index)
                        
                        # Verificar se o slot global está dentro da banda
                        target_band = sim_env.bands[decoded[1]]
                        calculated_global_slot = decoded[3]
                        if not (target_band.slot_start <= calculated_global_slot < target_band.slot_end):
                            continue
                            
                    except Exception as e:
                        continue
                    
                    # Verificar se a ação está válida
                    if action_index >= sim_env.action_space.n:
                        print(f"ERRO: action_index {action_index} >= action_space.n {sim_env.action_space.n}")
                        continue
                    
                    return action_index, False, False
            except Exception as e:
                print(f"Erro no cálculo de OSNR: {e}")
                continue
    return None


def shortest_available_path_first_fit_best_modulation_best_band(env: Env) -> tuple[int, bool, bool]:

    blocked_due_to_resources, blocked_due_to_osnr = False, False
    sim_env = get_qrmsa_env(env)
    source = sim_env.current_service.source
    destination = sim_env.current_service.destination
    k_paths = sim_env.k_shortest_paths[source, destination]
    
    # Verificar se o ambiente tem bandas
    if not hasattr(sim_env, 'bands') or not sim_env.bands:
        print("AVISO: Ambiente não tem bandas configuradas, usando heurística single-band")
        return heuristic_shortest_available_path_first_fit_best_modulation(env)
    
    # Iterar bandas preferindo menor atenuação (attenuation_normalized menor primeiro)
    # Não altera ordem global em sim_env.bands, apenas cria uma lista ordenada localmente.
    bands = sorted(sim_env.bands, key=lambda b: getattr(b, 'attenuation_normalized', float('inf')))
    
    for path_idx, path in enumerate(k_paths):
        for modulation_idx in range(sim_env.max_modulation_idx, -1, -1):
            modulation = sim_env.modulations[modulation_idx]
            required_slots = sim_env.get_number_slots(sim_env.current_service, modulation)
            
            if required_slots <= 0:
                continue 
            
            # Ao usar enumerate aqui, precisamos do índice original do band dentro de sim_env.bands
            # porque get_multiband_action_index espera o índice da banda na lista do ambiente.
            # Portanto, mapeamos cada banda ordenada para seu índice original.
            for band in bands:
                # Recuperar o índice original na lista do ambiente
                try:
                    band_idx = sim_env.bands.index(band)
                except ValueError:
                    # Fallback: se não encontrado (improvável), usar 0
                    band_idx = 0
                # Obter slots disponíveis do caminho
                available_slots = sim_env.get_available_slots(path)
                
                # Usar _get_candidates_in_band corretamente (apenas 3 parâmetros)
                try:
                    valid_starts = sim_env._get_candidates_in_band(available_slots, required_slots, band)
                except Exception as e:
                    print(f"Erro ao chamar _get_candidates_in_band: {e}")
                    blocked_due_to_resources = True
                    continue

                if not valid_starts:
                    blocked_due_to_resources = True
                    continue 
                
                # Usar primeiro candidato (first-fit)
                global_candidate_start = valid_starts[0]
                
                # SIMPLIFICAÇÃO: usar diretamente o approach funcional do ambiente
                # Como o ambiente assume slots uniformes, mapear global_candidate_start 
                # para o slot relativo dentro da faixa da banda
                slot_in_band = global_candidate_start - band.slot_start
                
                # Verificar se o slot relativo está dentro dos limites da banda uniforme
                if slot_in_band < 0 or slot_in_band >= sim_env.bands[0].num_slots:
                    continue
                
                # Configurar o serviço para teste de OSNR
                service = sim_env.current_service
                service.path = path
                service.initial_slot = global_candidate_start
                service.number_slots = required_slots
                service.current_modulation = modulation
                service.current_band = band
                
                # Calcular frequência central usando a banda
                try:
                    service.center_frequency = band.center_frequency_hz_from_global(
                        global_candidate_start, required_slots
                    )
                except Exception as e:
                    print(f"Erro ao calcular frequência central: {e}")
                    # Fallback para cálculo tradicional
                    service.center_frequency = (
                        sim_env.frequency_start +
                        (sim_env.frequency_slot_bandwidth * global_candidate_start) +
                        (sim_env.frequency_slot_bandwidth * (required_slots / 2))
                    )
                
                service.bandwidth = sim_env.frequency_slot_bandwidth * required_slots
                service.launch_power = sim_env.launch_power
                
                # Verificar OSNR
                try:
                    osnr, _, _ = calculate_osnr(sim_env, service)
                    
                    threshold = modulation.minimum_osnr + sim_env.margin
                    
                    if osnr >= threshold:
                        # Calcular action index corretamente
                        action_index = get_multiband_action_index(
                            sim_env, path_idx, band_idx, modulation_idx, slot_in_band
                        )
                        
                        # Verificar se a ação está válida
                        if action_index >= sim_env.action_space.n:
                            print(f"ERRO: action_index {action_index} >= action_space.n {sim_env.action_space.n}")
                            continue
                        
                        return action_index, False, False
                    else:
                        blocked_due_to_osnr = True
                        # Reset blocked_due_to_resources se há problema de OSNR
                        if blocked_due_to_resources:
                            blocked_due_to_resources = False
                except Exception as e:
                    print(f"Erro no cálculo de OSNR: {e}")
                    blocked_due_to_osnr = True
    
    return env.action_space.n - 1, blocked_due_to_resources, blocked_due_to_osnr 



def shortest_path_best_modulation_best_band_particle_swarm_optimization(env: Env) -> tuple[int, bool, bool]:
    """
    Heurística que utiliza Particle Swarm Optimization (PSO) para selecionar a melhor
    alocação de recursos em uma rede óptica com múltiplas bandas.
    """
    sim_env = get_qrmsa_env(env)
    source = sim_env.current_service.source
    destination = sim_env.current_service.destination
    k_paths = sim_env.k_shortest_paths[source, destination]

    # Verificar se o ambiente tem bandas
    if not hasattr(sim_env, 'bands') or not sim_env.bands:
        print("AVISO: Ambiente não tem bandas configuradas, usando heurística single-band")
        return heuristic_shortest_available_path_first_fit_best_modulation(env)

    # Definir função objetivo para PSO
    def objective_function(x):
        fitness = []
        for particle in x:
            path_idx = int(particle[0])
            band_idx = int(particle[1])
            modulation_idx = int(particle[2])
            slot_start = int(particle[3])

            path = k_paths[path_idx]
            band = sim_env.bands[band_idx]
            modulation = sim_env.modulations[modulation_idx]
            required_slots = sim_env.get_number_slots(sim_env.current_service, modulation)

            available_slots = sim_env.get_available_slots(path)
            valid_starts = sim_env._get_candidates_in_band(available_slots, required_slots, band)

            if slot_start not in valid_starts:
                fitness.append(1e6)  # Penalidade alta para soluções inválidas
                continue

            service = sim_env.current_service
            service.path = path
            service.initial_slot = slot_start
            service.number_slots = required_slots
            service.current_modulation = modulation
            service.current_band = band
            service.center_frequency = band.center_frequency_hz_from_global(slot_start, required_slots)
            service.bandwidth = sim_env.frequency_slot_bandwidth * required_slots
            service.launch_power = sim_env.launch_power

            osnr, _, _ = calculate_osnr(sim_env, service)
            threshold = modulation.minimum_osnr + sim_env.margin

            if osnr >= threshold:
                fitness.append(0)  # Boa solução
            else:
                fitness.append(1e5)  # Penalidade para baixa OSNR

        return np.array(fitness)

    # Configurar PSO
    num_particles = 30
    dimensions = 4  # path_idx, band_idx, modulation_idx, slot_start
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

    # Construir limites para partículas (inteiros convertidos em float para o otimizador)
    n_paths = len(k_paths)
    n_bands = len(sim_env.bands)
    n_mods = len(sim_env.modulations)
    slots_total = sim_env.num_spectrum_resources

    # lower and upper bounds for each dimension
    lb = [0, 0, 0, 0]
    ub = [max(0, n_paths - 1), max(0, n_bands - 1), max(0, n_mods - 1), max(0, slots_total - 1)]

    bounds = (np.array(lb, dtype=float), np.array(ub, dtype=float))

    # Função objetivo que implementa as fórmulas pedidas:
    # fragmentacao = 0.4*shannon + 0.3*cuts + 0.3*RSS
    # ATT = média de attenuation_normalized ao longo dos spans da rota e banda (usamos band.attenuation_normalized)
    # slots = tamanho_do_bloco + numero_de_slots_disponiveis
    # obj = 0.3*fragmentacao + 0.4*ATT + 0.3*Nslots
    def objective_fragmentation(x):
        """Recebe matriz (n_particles, dimensions) e retorna vetor de fitness (n_particles,).

        Menor fitness = melhor solução.
        """
        fitness = []
        for particle in x:
            try:
                path_idx = int(np.clip(np.round(particle[0]), 0, n_paths - 1))
                band_idx = int(np.clip(np.round(particle[1]), 0, n_bands - 1))
                modulation_idx = int(np.clip(np.round(particle[2]), 0, n_mods - 1))
                slot_start = int(np.clip(np.round(particle[3]), 0, slots_total - 1))

                path = k_paths[path_idx]
                band = sim_env.bands[band_idx]
                modulation = sim_env.modulations[modulation_idx]

                required_slots = sim_env.get_number_slots(sim_env.current_service, modulation)

                # checar candidatos dentro da banda
                available_slots = sim_env.get_available_slots(path)
                valid_starts = sim_env._get_candidates_in_band(available_slots, required_slots, band)

                if slot_start not in valid_starts:
                    # penalidade alta para soluções inválidas
                    fitness.append(1e6)
                    continue

                # calcular fragmentação usando utilitários
                # shannon: entropia por link/rota
                try:
                    shannon = link_shannon_entropy_(available_slots)
                except Exception:
                    # fallback: usar entropia simples baseada em proporção de ocupação
                    occupancy = 1.0 - (np.sum(available_slots) / len(available_slots))
                    shannon = occupancy

                try:
                    cuts = fragmentation_route_cuts(available_slots)
                except Exception:
                    cuts = float(np.sum(np.abs(np.diff(available_slots))))

                try:
                    rss = fragmentation_route_rss(available_slots)
                except Exception:
                    rss = float(np.sum(available_slots))

                fragmentacao = 0.4 * shannon + 0.3 * cuts + 0.3 * rss

                # ATT: usamos a atenuacao da banda como proxy; também incorporamos distância média da rota
                try:
                    att_band = getattr(band, 'attenuation_normalized', 0.0)
                except Exception:
                    att_band = 0.0

                # slots metric: bloco disponível (tamanho do bloco encontrado) + numero de slots livres totais na rota
                initial_indices, values, lengths = rle(available_slots)
                # identificar bloco que contém slot_start (global index)
                block_size = 0
                for idx_i, start_idx in enumerate(initial_indices):
                    if values[idx_i] == 1:
                        s = start_idx
                        e = start_idx + lengths[idx_i]
                        if s <= slot_start < e:
                            block_size = lengths[idx_i]
                            break

                n_slots_free = int(np.sum(available_slots))
                slots_metric = float(block_size + n_slots_free)

                # normalizar componentes para escalas semelhantes
                # fragmentacao assumida entre 0..1 (se funções retornarem maiores, escalonar)
                frag_norm = float(fragmentacao)
                att_norm = float(att_band)  # já em 1/m pequeno; ponderar diretamente
                slots_norm = slots_metric / max(1.0, slots_total)

                obj = 0.3 * frag_norm + 0.4 * att_norm + 0.3 * slots_norm

                fitness.append(obj)
            except Exception as e:
                # penalidade por erro
                fitness.append(1e6)

        return np.array(fitness)

    # executar PSO (Global Best)
    try:
        optimizer = ps.single.GlobalBestPSO(n_particles=num_particles, dimensions=dimensions, options=options, bounds=bounds)
        best_cost, best_pos = optimizer.optimize(objective_fragmentation, iters=50, verbose=False)
    except Exception as e:
        # Falha em executar PSO: fallback para heurística simples
        print(f"PSO falhou: {e}, usando heurística fallback")
        return shortest_available_path_first_fit_best_modulation_best_band(env)

    # converter best_pos para índices inteiros
    best_path = int(np.clip(np.round(best_pos[0]), 0, n_paths - 1))
    best_band = int(np.clip(np.round(best_pos[1]), 0, n_bands - 1))
    best_mod = int(np.clip(np.round(best_pos[2]), 0, n_mods - 1))
    best_slot = int(np.clip(np.round(best_pos[3]), 0, slots_total - 1))

    # validar e calcular action index
    path = k_paths[best_path]
    band = sim_env.bands[best_band]
    modulation = sim_env.modulations[best_mod]
    required_slots = sim_env.get_number_slots(sim_env.current_service, modulation)
    available_slots = sim_env.get_available_slots(path)
    valid_starts = sim_env._get_candidates_in_band(available_slots, required_slots, band)

    if best_slot not in valid_starts:
        # escolher primeiro candidato válido como fallback
        if len(valid_starts) == 0:
            return env.action_space.n - 1, True, False
        best_slot = int(valid_starts[0])

    # mapear slot global -> slot in band
    slot_in_band = best_slot - band.slot_start
    try:
        action_index = get_multiband_action_index(sim_env, best_path, best_band, best_mod, slot_in_band)
    except Exception:
        return env.action_space.n - 1, True, False

    if action_index >= sim_env.action_space.n:
        return env.action_space.n - 1, True, False

    return int(action_index), False, False

# manter compatibilidade de nome exportando alias
pso_fragmentation_band_aware = shortest_path_best_modulation_best_band_particle_swarm_optimization