import numpy as np
from optical_networking_gym.core.osnr import calculate_osnr
from optical_networking_gym.envs.qrmsa import QRMSAEnv
from gymnasium import Env

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
    # Ambiente decodifica ações multibanda usando slots_per_band (primeira banda) e
    # modulação relativa (allowed_mods). Precisamos gerar o mesmo índice.
    slots_per_band = env.bands[0].num_slots if env.bands else env.num_spectrum_resources
    if slot_in_band < 0 or slot_in_band >= slots_per_band:
        raise ValueError(
            f"slot_in_band {slot_in_band} fora do intervalo 0..{slots_per_band-1} para banda {band_index}"
        )

    # Mapeia modulação absoluta para índice relativo usado no espaço de ações
    if env.max_modulation_idx > 1:
        allowed_mods = list(range(env.max_modulation_idx, env.max_modulation_idx - env.modulations_to_consider, -1))
    else:
        allowed_mods = list(range(0, env.modulations_to_consider))

    try:
        modulation_relative = allowed_mods.index(modulation_index)
    except ValueError:
        raise ValueError(f"Modulação {modulation_index} não está em allowed_mods {allowed_mods}")

    # action_space_size = (k_paths * num_bands * modulations_to_consider * slots_per_band) + 1
    action_index = (
        ((path_index * env.num_bands + band_index) * env.modulations_to_consider + modulation_relative)
        * slots_per_band
        + slot_in_band
    )
    return action_index


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
