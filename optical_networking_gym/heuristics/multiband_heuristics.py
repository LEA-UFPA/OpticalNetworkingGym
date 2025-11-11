"""
Multiband-specific Heuristics for Optical Network Gym

This module contains heuristics designed specifically for multiband optical networks.
These heuristics handle multiple spectral bands (C, L, S, etc.) with different characteristics.

Note: Single-band heuristics are located in heuristics.py
"""

import numpy as np
from gymnasium import Env
from optical_networking_gym.envs.qrmsa import QRMSAEnv
from optical_networking_gym.core.osnr import calculate_osnr


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


def try_allocate_in_band(sim_env, k_paths, band_idx):
    """
    Helper function to attempt allocation in a specific band.
    
    Args:
        sim_env: The QRMSA environment
        k_paths: List of k-shortest paths
        band_idx: Index of the band to try allocation in
        
    Returns:
        Tuple of (action_index, blocked_due_to_resources, blocked_due_to_osnr) if successful,
        None otherwise
    """
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


# ==============================================================================
# Multiband Heuristics
# ==============================================================================


def heuristic_priority_band_C_then_L(env: Env) -> tuple[int, bool, bool]:
    """
    Tenta alocar o serviço na banda C enquanto a ocupação for < 90%.
    Se >= 90%, tenta na banda L.
    
    NOTE: This heuristic does not work correctly yet - kept for reference.
    
    Args:
        env: The gymnasium environment
        
    Returns:
        Tuple of (action_index, blocked_due_to_resources, blocked_due_to_osnr)
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


def shortest_available_path_first_fit_best_modulation_best_band(env: Env) -> tuple[int, bool, bool]:
    """
    Multiband heuristic that tries to find the best allocation considering:
    - Shortest available path first
    - Best modulation (highest spectral efficiency that meets OSNR requirements)
    - Best band (tries all bands for each path-modulation combination)
    - First-fit slot selection
    
    This is the recommended heuristic for multiband optical networks.
    
    Args:
        env: The gymnasium environment
        
    Returns:
        Tuple of (action_index, blocked_due_to_resources, blocked_due_to_osnr)
    """
    blocked_due_to_resources, blocked_due_to_osnr = False, False
    sim_env = get_qrmsa_env(env)
    source = sim_env.current_service.source
    destination = sim_env.current_service.destination
    k_paths = sim_env.k_shortest_paths[source, destination]
    
    # Verificar se o ambiente tem bandas
    if not hasattr(sim_env, 'bands') or not sim_env.bands:
        print("AVISO: Ambiente não tem bandas configuradas")
        return env.action_space.n - 1, True, False
    
    bands = sim_env.bands
    
    for path_idx, path in enumerate(k_paths):
        for modulation_idx in range(sim_env.max_modulation_idx, -1, -1):
            modulation = sim_env.modulations[modulation_idx]
            required_slots = sim_env.get_number_slots(sim_env.current_service, modulation)
            
            if required_slots <= 0:
                continue 
            
            for band_idx, band in enumerate(bands):
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
                
                # Verificar se o slot relativo está dentro dos limites reais da banda
                if slot_in_band < 0 or slot_in_band >= band.num_slots:
                    continue

                if not band.contains_slot_range(global_candidate_start, required_slots):
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

                        # Validar combinação decodificando com a mesma lógica do ambiente
                        decoded = sim_env.decimal_to_array(
                            action_index,
                            [sim_env.k_paths, sim_env.num_bands, sim_env.modulations_to_consider, sim_env.bands[0].num_slots]
                        )
                        decoded_path_idx = decoded[0]
                        decoded_band_idx = decoded[1]
                        decoded_slot_rel = decoded[3]
                        
                        paths_for_pair = sim_env.k_shortest_paths[
                            sim_env.current_service.source,
                            sim_env.current_service.destination
                        ]
                        if decoded_path_idx >= len(paths_for_pair):
                            continue

                        decoded_band = sim_env.bands[decoded_band_idx]
                        decoded_global_slot = decoded_band.slot_start + decoded_slot_rel

                        if not decoded_band.contains_slot_range(decoded_global_slot, required_slots):
                            continue

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

