from typing import Optional
import numpy as np
from optical_networking_gym.topology import Modulation, Path
from optical_networking_gym.utils import rle
from optical_networking_gym.core.osnr import calculate_osnr
from optical_networking_gym.envs.qrmsa import QRMSAEnv
from gymnasium import Env

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
    
    Args:
        env (QRMSAEnv): O ambiente QRMSAEnv.
        path_index (int): Índice da rota.
        modulation_index (int): Índice absoluto da modulação.
        initial_slot (int): Slot inicial para alocação.
    
    Returns:
        int: Índice da ação correspondente.
    """
    # Converter o índice absoluto da modulação para o relativo
    relative_modulation_index = env.max_modulation_idx - modulation_index
    
    return (path_index * env.modulations_to_consider * env.num_spectrum_resources +
            relative_modulation_index * env.num_spectrum_resources +
            initial_slot)

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
        # print(f"OSNR calculado para ação {action_index}: {osnr:.2f}")

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



def shortest_available_path_first_fit_best_modulation(
    mask: np.ndarray,
    # env: Env,
) -> Optional[int]:
    return int(np.where(mask == 1)[0][0])

    # """
    # Seleciona a rota mais curta disponível com a primeira alocação possível e a melhor modulação.

    # Args:
    #     env (gym.Env): O ambiente potencialmente envolvido em wrappers.

    # Returns:
    #     Optional[int]: Índice da ação correspondente, ou a ação de rejeição se permitido, ou None.
    # """
    # qrmsa_env: QRMSAEnv = get_qrmsa_env(env)  # Descompactar o ambiente
    # bl_resource = True
    # bl_osnr = False
    # # print("Inicializando bl_resource como True e bl_osnr como False")

    # # Itera por cada rota possível entre source e destination
    # for idp, path in enumerate(qrmsa_env.k_shortest_paths[
    #     qrmsa_env.current_service.source,
    #     qrmsa_env.current_service.destination,
    # ]):
    #     # print(f"\nIterando sobre o caminho {idp}: {path}")
    #     available_slots = qrmsa_env.get_available_slots(path)
    #     # print(f"Slots disponíveis para o caminho {idp}: {available_slots}")

    #     # Executa a RLE para identificar blocos contíguos na lista de slots disponíveis
    #     initial_indices, values, lengths = rle(available_slots)
    #     # print(f"RLE result - initial_indices: {initial_indices}, values: {values}, lengths: {lengths}")

    #     # Itera pelas modulações, da melhor para a pior
    #     for idm, modulation in zip(range(len(qrmsa_env.modulations) - 1, -1, -1),
    #                                 reversed(qrmsa_env.modulations)):
    #         # print(f"\nIterando sobre a modulação {idm}: {modulation}")
    #         # Número de slots requeridos para o serviço com a modulação atual (incluindo eventuais requisitos de guarda)
    #         number_slots = qrmsa_env.get_number_slots(qrmsa_env.current_service, modulation)
    #         # print(f"Número de slots requeridos para a modulação {idm}: {number_slots}")

    #         # Filtra os blocos (candidatos) manualmente, gerando TODOS os índices possíveis dentro do bloco
    #         candidatos = []
    #         for start, val, length in zip(initial_indices, values, lengths):
    #             if val == 1:
    #                 # Se o bloco se estende até o final do espectro, ignora o guard band
    #                 if start + length == env.num_spectrum_resources:
    #                     if length >= number_slots:
    #                         for candidate in range(start, start + length - number_slots + 1):
    #                             candidatos.append(candidate)
    #                             # print(f"Adicionando candidato {candidate} (sem guard band - bloco até o final) a partir do bloco que inicia em {start} com comprimento {length}")
    #                 else:
    #                     # Caso contrário, exige um slot extra para o guard band
    #                     if length >= (number_slots + 1):
    #                         for candidate in range(start, start + length - (number_slots + 1) + 1):
    #                             candidatos.append(candidate)
    #                             # print(f"Adicionando candidato {candidate} (guard band incluído) a partir do bloco que inicia em {start} com comprimento {length}")

    #         # print(f"Candidatos encontrados para modulação {idm}: {candidatos}")

    #         if len(candidatos) > 0:
    #             bl_resource = False
    #             # print("Atualizando bl_resource para False")
    #             qrmsa_env.current_service.blocked_due_to_resources = False

    #             # Itera por cada candidato (initial slot) encontrado
    #             for candidate in candidatos:
    #                 # print(f"\nTestando candidato com initial_slot {candidate}")
    #                 # Atualiza os parâmetros do serviço para o candidato atual
    #                 qrmsa_env.current_service.path = path
    #                 qrmsa_env.current_service.initial_slot = candidate
    #                 qrmsa_env.current_service.number_slots = number_slots
    #                 qrmsa_env.current_service.center_frequency = (
    #                     qrmsa_env.frequency_start +
    #                     (qrmsa_env.frequency_slot_bandwidth * candidate) +
    #                     (qrmsa_env.frequency_slot_bandwidth * (number_slots / 2))
    #                 )
    #                 qrmsa_env.current_service.bandwidth = qrmsa_env.frequency_slot_bandwidth * number_slots
    #                 qrmsa_env.current_service.launch_power = qrmsa_env.launch_power
    #                 # print(f"Atualizando serviço com path: {path}, initial_slot: {candidate}, number_slots: {number_slots}")

    #                 # Calcula o OSNR para o serviço com o candidato atual
    #                 osnr, ase, nli = calculate_osnr(qrmsa_env, qrmsa_env.current_service)
    #                 # print(f"OSNR calculado: {osnr}, ASE: {ase}, NLI: {nli}")

    #                 if osnr >= modulation.minimum_osnr + qrmsa_env.margin:
    #                     bl_osnr = False
    #                     # print("OSNR aceitável, atualizando bl_osnr para False")
    #                     # Converte para o índice de ação com base no caminho, modulação e initial slot selecionados
    #                     action = get_action_index(qrmsa_env, idp, idm, candidate)
    #                     # print(f"Ação selecionada: {action}")
    #                     return action, bl_osnr, bl_resource
    #                 else:
    #                     bl_osnr = True
    #                     # print("OSNR não aceitável para este candidato, continuando para o próximo candidato")

    # # Se nenhum bloco candidato resultar num OSNR aceitável, retorna a ação de rejeição
    # # print("Nenhum candidato válido encontrado, retornando ação de rejeição")
    # return qrmsa_env.reject_action, bl_osnr, bl_resource


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
    qrmsa_env: QRMSAEnv = get_qrmsa_env(env)  # Descompactar o ambiente

    for idp, path in enumerate(qrmsa_env.k_shortest_paths[
        qrmsa_env.current_service.source,
        qrmsa_env.current_service.destination,
    ]):
        available_slots = qrmsa_env.get_available_slots(path)
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
                if osnr >= modulation.minimum_osnr + qrmsa_env.margin:
                    # Converter para índice de ação
                    action = get_action_index(qrmsa_env, idp, idm, initial_slot)
                    return action

    # Se nenhuma ação válida encontrada, retornar a ação de rejeição se permitido
    if qrmsa_env.allow_rejection:
        return qrmsa_env.reject_action
    else:
        return None  # ou uma ação padrão específica

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
                    return action

    # Se nenhuma ação válida encontrada, retornar a ação de rejeição se permitido
    return qrmsa_env.reject_action

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
        return solution

    # Se nenhuma ação válida encontrada, retornar a ação de rejeição se permitido
    if qrmsa_env.allow_rejection:
        return qrmsa_env.reject_action
    else:
        return None  # ou uma ação padrão específica
