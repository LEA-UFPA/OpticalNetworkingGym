import numpy as np
import math
from typing import Optional
from optical_networking_gym.utils import rle, link_shannon_entropy_, fragmentation_route_cuts, fragmentation_route_rss
from optical_networking_gym.core.osnr import calculate_osnr
from optical_networking_gym.envs.qrmsa import QRMSAEnv
from gymnasium import Env

try:
    import pyswarms as ps
    HAS_PYSWARMS = True
except ImportError:
    HAS_PYSWARMS = False

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
    # IMPORTANTE: O ambiente decodifica usando bands[0].num_slots como referência
    # Isso significa que bandas maiores podem ter slots inacessíveis via ações
    slots_per_band = env.bands[0].num_slots if env.bands else env.num_spectrum_resources
    
    # Validar se o slot está dentro dos limites da banda alvo
    target_band = env.bands[band_index]
    if slot_in_band < 0 or slot_in_band >= target_band.num_slots:
        raise ValueError(
            f"slot_in_band {slot_in_band} fora do intervalo 0..{target_band.num_slots-1} "
            f"para banda {band_index} ({target_band.name})"
        )
    
    # CRÍTICO: Validar se o slot está dentro do limite usado para decodificação
    # Se a banda alvo for maior que bands[0], alguns slots ficam inacessíveis
    if slot_in_band >= slots_per_band:
        raise ValueError(
            f"slot_in_band {slot_in_band} excede o limite de decodificação {slots_per_band} "
            f"(tamanho da primeira banda). Banda {target_band.name} tem {target_band.num_slots} slots, "
            f"mas apenas os primeiros {slots_per_band} são acessíveis."
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


def heuristic_shortest_available_path_first_fit_best_modulation(env: Env) -> tuple[int, bool, bool, dict]:
    """
    Heurística com informações detalhadas sobre bloqueios.
    
    Returns:
        tuple: (action_index, blocked_resources, blocked_osnr, blocking_info)
        onde blocking_info contém: {'ase': float, 'nli': float, 'osnr': float, 
                                     'osnr_req': float, 'band': str or None}
    """
    blocked_due_to_resources, blocked_due_to_osnr = False, False
    blocking_info = {'ase': 0.0, 'nli': 0.0, 'osnr': 0.0, 'osnr_req': 0.0, 'band': None}
    
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
                        blocking_info['band'] = band.name  # Registrar banda
                        break
                else:
                    # Se nenhuma banda contém o slot, usar a primeira banda como fallback
                    service.current_band = sim_env.bands[0]
                    service.center_frequency = (
                        sim_env.frequency_start +
                        (sim_env.frequency_slot_bandwidth * candidate_start) +
                        (sim_env.frequency_slot_bandwidth * (required_slots / 2)))
                    blocking_info['band'] = sim_env.bands[0].name
            else:
                # Ambiente single-band tradicional
                service.current_band = None
                service.center_frequency = (
                        sim_env.frequency_start +
                        (sim_env.frequency_slot_bandwidth * candidate_start) +
                        (sim_env.frequency_slot_bandwidth * (required_slots / 2)))
            
            service.bandwidth = sim_env.frequency_slot_bandwidth * required_slots
            service.launch_power = sim_env.launch_power
            
            # Calcular OSNR, ASE e NLI
            osnr, ase, nli = calculate_osnr(sim_env, service)
            
            threshold = modulation.minimum_osnr + sim_env.margin
            if osnr >= threshold:
                action_index = get_action_index(sim_env, path_idx, modulation_idx, candidate_start)
                return action_index, False, False, blocking_info
            else:
                blocked_due_to_osnr = True
                # Salvar informações detalhadas do bloqueio
                blocking_info['osnr'] = osnr
                blocking_info['ase'] = ase
                blocking_info['nli'] = nli
                blocking_info['osnr_req'] = threshold
                if blocked_due_to_resources:
                    blocked_due_to_resources = False
    
    return env.action_space.n - 1, blocked_due_to_resources, blocked_due_to_osnr, blocking_info 


def shortest_available_path_first_fit_best_modulation_best_band(env: Env) -> tuple[int, bool, bool, dict]:
    """
    Heurística multibanda com informações detalhadas sobre bloqueios.
    
    Returns:
        tuple: (action_index, blocked_resources, blocked_osnr, blocking_info)
    """
    blocked_due_to_resources, blocked_due_to_osnr = False, False
    blocking_info = {'ase': 0.0, 'nli': 0.0, 'osnr': 0.0, 'osnr_req': 0.0, 'band': None}
    
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
                
                # Verificar se o slot relativo está dentro dos limites da banda atual
                # CORREÇÃO: Usar band.num_slots em vez de sim_env.bands[0].num_slots
                if slot_in_band < 0 or slot_in_band >= band.num_slots:
                    continue
                
                # Configurar o serviço para teste de OSNR
                service = sim_env.current_service
                service.path = path
                service.initial_slot = global_candidate_start
                service.number_slots = required_slots
                service.current_modulation = modulation
                service.current_band = band
                blocking_info['band'] = band.name  # Registrar banda
                
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
                    osnr, ase, nli = calculate_osnr(sim_env, service)
                    
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
                        
                        return action_index, False, False, blocking_info
                    else:
                        blocked_due_to_osnr = True
                        # Salvar informações detalhadas do bloqueio
                        blocking_info['osnr'] = osnr
                        blocking_info['ase'] = ase
                        blocking_info['nli'] = nli
                        blocking_info['osnr_req'] = threshold
                        # Reset blocked_due_to_resources se há problema de OSNR
                        if blocked_due_to_resources:
                            blocked_due_to_resources = False
                except Exception as e:
                    print(f"Erro no cálculo de OSNR: {e}")
                    blocked_due_to_osnr = True
    
    return env.action_space.n - 1, blocked_due_to_resources, blocked_due_to_osnr, blocking_info 


def get_best_band_path_pso(env: Env) -> tuple[int, bool, bool, dict]:
    """
    Heurística PSO para seleção de banda e caminho em ambiente multiband.
    
    Usa Particle Swarm Optimization para encontrar a melhor combinação (rota, banda),
    considerando atenuação, utilização de slots e fragmentação.
    Após encontrar a melhor (rota, banda), usa first-fit para escolher modulação e slots.
    
    Args:
        env: Ambiente Gym (potencialmente wrapped).
        
    Returns:
        tuple[int, bool, bool, dict]: (action_index, blocked_resources, blocked_osnr, blocking_info)
    """
    if not HAS_PYSWARMS:
        # Fallback para heurística padrão se PySwarms não estiver instalado
        print("AVISO: PySwarms não está instalado, usando heurística padrão")
        return shortest_available_path_first_fit_best_modulation_best_band(env)
    
    # Resolve o QRMSA base e garanta fallback se o ambiente não tiver bandas configuradas
    sim_env = get_qrmsa_env(env)
    if not hasattr(sim_env, "bands") or not sim_env.bands:
        return heuristic_shortest_available_path_first_fit_best_modulation(env)
    
    service = sim_env.current_service
    source = service.source
    destination = service.destination
    k_paths = sim_env.k_shortest_paths[source, destination]
    num_paths = min(len(k_paths), sim_env.k_paths)
    num_bands = len(sim_env.bands)

    if num_paths == 0 or num_bands == 0:
        return env.action_space.n - 1, True, False

    # Tamanho da demanda de referência vem da maior modulação permitida no instante
    ref_mod_idx = sim_env.max_modulation_idx if sim_env.max_modulation_idx >= 0 else 0
    ref_mod_idx = min(ref_mod_idx, len(sim_env.modulations) - 1)
    ref_modulation = sim_env.modulations[ref_mod_idx]
    ref_slots = max(1, sim_env.get_number_slots(service, ref_modulation))
    penalty = 1e6

    def clamp_indices(p_idx: float, b_idx: float) -> tuple[int, int]:
        path_idx = int(np.clip(round(p_idx), 0, num_paths - 1))
        band_idx = int(np.clip(round(b_idx), 0, num_bands - 1))
        return path_idx, band_idx

    def allocation_metric(band_available: np.ndarray) -> float:
        # Combina taxa de utilização e maior bloco contíguo para medir pressão de alocação
        if band_available.size == 0:
            return 1.0
        free_ratio = np.sum(band_available) / band_available.size
        largest_block = 0
        initial_indices, values, lengths = rle(band_available)
        if values is not None:
            free_blocks = lengths[values == 1]
            if free_blocks.size > 0:
                largest_block = int(np.max(free_blocks))
        term1 = 1.0 - free_ratio
        term2 = min(1.0, ref_slots / max(largest_block, 1))
        return 0.6 * term1 + 0.4 * term2

    def fragmentation_metric(path_idx: int, band) -> float:
        # Avalia fragmentação da banda recortando o espectro do caminho e usando as métricas do QRMSA
        raw = sim_env._get_spectrum_slots(path_idx)
        if isinstance(raw, list):
            spectrum = [np.array(row) for row in raw]
        else:
            arr = np.array(raw)
            if arr.ndim == 1:
                spectrum = [arr]
            elif arr.ndim == 2:
                spectrum = [arr[i, :] for i in range(arr.shape[0])]
            else:
                return penalty
        band_spectrum = [row[band.slot_start:band.slot_end] for row in spectrum if row.size > 0]
        if not band_spectrum:
            return penalty
        entropies = [link_shannon_entropy_(row.tolist()) for row in band_spectrum]
        shannon = sum(entropies) / len(entropies) if entropies else 0.0
        cuts = fragmentation_route_cuts([row.tolist() for row in band_spectrum])
        rss = fragmentation_route_rss([row.tolist() for row in band_spectrum])
        return 0.4 * shannon + 0.3 * cuts + 0.3 * rss

    def evaluate_pair(path_idx: int, band_idx: int) -> float:
        try:
            path = k_paths[path_idx]
            band = sim_env.bands[band_idx]
            available_slots = sim_env.get_available_slots(path)
            if available_slots is None:
                return penalty
            band_slice = np.array(available_slots[band.slot_start:band.slot_end])
            try:
                candidates = sim_env._get_candidates_in_band(available_slots, ref_slots, band)
            except Exception:
                return penalty
            if not candidates:
                return penalty
            attenuation = getattr(band, "attenuation_normalized", 1.0)
            # Normaliza atenuação pela pior (maior) banda disponível para manter 0..1
            max_att = max(getattr(b, "attenuation_normalized", 1.0) for b in sim_env.bands) or 1.0
            attenuation_norm = min(1.0, attenuation / max_att) if attenuation >= 0 else 1.0

            allocation = allocation_metric(band_slice)  # já 0..1 pela construção

            # Fragmentação pode crescer; usa transformação saturante para 0..1
            fragmentation_raw = max(0.0, fragmentation_metric(path_idx, band))
            fragmentation_norm = fragmentation_raw / (1.0 + fragmentation_raw)

            result = 0.1560 * attenuation_norm + 0.0581 * allocation + 0.8662 * fragmentation_norm
            
            # Garantir que o resultado é finito
            if not np.isfinite(result):
                return penalty
            
            return result
        except Exception:
            return penalty

    def pso_objective(x: np.ndarray) -> np.ndarray:
        # Pontua cada partícula convertendo (path, band) em score composto
        scores = np.zeros(x.shape[0], dtype=float)
        for idx, particle in enumerate(x):
            path_idx, band_idx = clamp_indices(particle[0], particle[1])
            score = evaluate_pair(path_idx, band_idx)
            # Garantir que score é finito
            if not np.isfinite(score):
                score = penalty
            scores[idx] = score
        return scores

    if num_paths * num_bands == 1:
        best_path_idx, best_band_idx = 0, 0
    else:
        # Executa o PSO apenas sobre (rota, banda); o restante fica com o first-fit padrão
        options = {'c1': 0.8116, 'c2': 1.9064, 'w': 0.9856}
        particles = max(5, min(32, num_paths * num_bands))
        bounds = (np.zeros(2), np.array([num_paths - 1, num_bands - 1], dtype=float))
        optimizer = ps.single.GlobalBestPSO(
            n_particles=particles,
            dimensions=2,
            options=options,
            bounds=bounds,
        )
        try:
            _, pos = optimizer.optimize(pso_objective, iters=10, verbose=False)
            # Garantir que pos não contém NaN
            if not np.all(np.isfinite(pos)):
                best_path_idx, best_band_idx = 0, 0
            else:
                best_path_idx, best_band_idx = clamp_indices(pos[0], pos[1])
        except Exception:
            # Se PSO falhar, usar valores padrão
            best_path_idx, best_band_idx = 0, 0

    def allocate_with_first_fit(path_idx: int, band_idx: int) -> tuple[Optional[int], bool, bool, dict]:
        # Depois da escolha via PSO, caia no fluxo first-fit existente para modularização e slots
        blocked_resources = False
        blocked_osnr = False
        blocking_info = {'ase': 0.0, 'nli': 0.0, 'osnr': 0.0, 'osnr_req': 0.0, 'band': None}
        
        path = k_paths[path_idx]
        band = sim_env.bands[band_idx]
        blocking_info['band'] = band.name
        available_slots = sim_env.get_available_slots(path)

        for modulation_idx in range(sim_env.max_modulation_idx, -1, -1):
            modulation = sim_env.modulations[modulation_idx]
            required_slots = sim_env.get_number_slots(service, modulation)
            if required_slots <= 0:
                continue

            try:
                valid_starts = sim_env._get_candidates_in_band(available_slots, required_slots, band)
            except Exception:
                blocked_resources = True
                continue

            if not valid_starts:
                blocked_resources = True
                continue

            global_candidate_start = valid_starts[0]
            slot_in_band = global_candidate_start - band.slot_start
            # CORREÇÃO: Usar o tamanho da banda atual (band.num_slots) em vez de assumir 
            # que todas as bandas têm o mesmo tamanho (sim_env.bands[0].num_slots)
            if slot_in_band < 0 or slot_in_band >= band.num_slots:
                continue

            candidate_service = service
            candidate_service.path = path
            candidate_service.initial_slot = global_candidate_start
            candidate_service.number_slots = required_slots
            candidate_service.current_modulation = modulation
            candidate_service.current_band = band

            try:
                candidate_service.center_frequency = band.center_frequency_hz_from_global(
                    global_candidate_start, required_slots
                )
            except Exception:
                candidate_service.center_frequency = (
                    sim_env.frequency_start +
                    (sim_env.frequency_slot_bandwidth * global_candidate_start) +
                    (sim_env.frequency_slot_bandwidth * (required_slots / 2))
                )

            candidate_service.bandwidth = sim_env.frequency_slot_bandwidth * required_slots
            candidate_service.launch_power = sim_env.launch_power

            try:
                osnr, ase, nli = calculate_osnr(sim_env, candidate_service)
            except Exception:
                blocked_osnr = True
                continue

            threshold = modulation.minimum_osnr + sim_env.margin
            if osnr >= threshold:
                action_index = get_multiband_action_index(
                    sim_env, path_idx, band_idx, modulation_idx, slot_in_band
                )
                if action_index < sim_env.action_space.n:
                    return action_index, False, False, blocking_info
            else:
                blocked_osnr = True
                # Salvar informações detalhadas do bloqueio
                blocking_info['osnr'] = osnr
                blocking_info['ase'] = ase
                blocking_info['nli'] = nli
                blocking_info['osnr_req'] = threshold
                if blocked_resources:
                    blocked_resources = False

        return None, blocked_resources, blocked_osnr, blocking_info

    action_index, blocked_resources, blocked_osnr, blocking_info = allocate_with_first_fit(best_path_idx, best_band_idx)
    if action_index is not None:
        return action_index, blocked_resources, blocked_osnr, blocking_info

    return shortest_available_path_first_fit_best_modulation_best_band(env)
