from math import pi
from libc.math cimport asinh, pi, exp, log10

import cython
cimport numpy as np
import numpy as np

import typing

if typing.TYPE_CHECKING:
    from optical_networking_gym.envs.qrmsa import QRMSAEnv

cpdef calculate_osnr(env: QRMSAEnv, current_service: object, qot_constraint: str = "ASE+NLI"):
    cdef double beta_2 = -21.3e-27
    cdef double gamma = 1.3e-3
    cdef double h_plank = 6.626e-34
    cdef double acc_gsnr = 0.0
    cdef double acc_ase = 0.0
    cdef double acc_nli = 0.0
    cdef double gsnr = 0.0
    cdef double ase = 0.0
    cdef double nli = 0.0
    cdef double l_eff_a = 0.0
    cdef double l_eff = 0.0
    cdef double phi = 0.0
    cdef double sum_phi = 0.0
    cdef double power_ase = 0.0
    cdef double power_nli_span = 0.0

    cdef np.ndarray[np.double_t, ndim=1] phi_modulation_format = np.array(
        [1.0, 1.0, 2.0/3.0, 17.0/25.0, 69.0/100.0, 13.0/21.0],
        dtype=np.float64
    )

    cdef object link
    cdef object span
    cdef object running_service

    for link in current_service.path.links:
        if hasattr(env, 'topo_cache') and env.topo_cache is not None:
            node1_id = env.topo_cache.get_node_id(link.node1)
            node2_id = env.topo_cache.get_node_id(link.node2)
            edge_idx = env.topo_cache.get_edge_index(node1_id, node2_id)
            link_data = env.topo_cache.edge_data[edge_idx]
        else:
            link_data = env.topology[link.node1][link.node2]

        for span in link_data["link"].spans:
            if qot_constraint == "ASE+NLI":
                l_eff_a = 1.0 / (2.0 * span.attenuation_normalized)
                l_eff = (
                    1.0 - np.exp(-2.0 * span.attenuation_normalized * span.length * 1e3)
                ) / (2.0 * span.attenuation_normalized)
                sum_phi = asinh(
                    pi**2 * abs(beta_2) * (current_service.bandwidth**2) /
                    (4.0 * span.attenuation_normalized)
                )
                if hasattr(env, 'topo_cache') and env.topo_cache is not None:
                    running_services = env.topo_cache.get_running_services(link.node1, link.node2)
                else:
                    running_services = []
                
                for running_service in running_services:
                    if running_service.service_id != current_service.service_id:
                        try:
                            phi = (
                                asinh(
                                    pi**2 * abs(beta_2) * l_eff_a * running_service.bandwidth *
                                    (
                                        running_service.center_frequency
                                        - current_service.center_frequency
                                        + (running_service.bandwidth / 2.0)
                                    )
                                )
                                - asinh(
                                    pi**2 * abs(beta_2) * l_eff_a * running_service.bandwidth *
                                    (
                                        running_service.center_frequency
                                        - current_service.center_frequency
                                        - (running_service.bandwidth / 2.0)
                                    )
                                )
                            ) - (
                                phi_modulation_format[running_service.current_modulation.spectral_efficiency - 1]
                                * (
                                    running_service.bandwidth
                                    / abs(running_service.center_frequency - current_service.center_frequency)
                                )
                                * (5.0 / 3.0)
                                * (l_eff / (span.length * 1e3))
                            )
                            sum_phi += phi
                        except Exception as e:
                            print(f"Error: {e}")
                            print("================= error =================")

                power_nli_span = (
                    (current_service.launch_power / current_service.bandwidth)**3
                    * (8.0 / (27.0 * pi * abs(beta_2)))
                    * (gamma**2)
                    * l_eff
                    * sum_phi
                    * current_service.bandwidth
                )
            else:
                power_nli_span = 0.0

            power_ase = (
                current_service.bandwidth
                * h_plank
                * current_service.center_frequency
                * (exp(2.0 * span.attenuation_normalized * span.length * 1e3) - 1.0)
                * span.noise_figure_normalized
            )
            if qot_constraint == "ASE+NLI":
                acc_gsnr += 1.0 / (current_service.launch_power / (power_ase + power_nli_span))
                acc_nli  += 1.0 / (current_service.launch_power / power_nli_span)
            else:
                acc_gsnr += 1.0 / (current_service.launch_power / power_ase)
                acc_nli  += 0.0
            acc_ase  += 1.0 / (current_service.launch_power / power_ase)

    gsnr = 10.0 * np.log10(1.0 / acc_gsnr)
    ase =  10.0 * np.log10(1.0 / acc_ase)
    if acc_nli > 0:
        nli =  10.0 * np.log10(1.0 / acc_nli)
    else:
        nli = 0.0
    return gsnr, ase, nli


cpdef calculate_osnr_default_attenuation(
    env: QRMSAEnv,
    current_service: object,
    attenuation_normalized: cython.double,
    noise_figure_normalized: cython.double
):
    cdef double beta_2 = -21.3e-27
    cdef double gamma = 1.3e-3
    cdef double h_plank = 6.626e-34
    cdef double acc_gsnr = 0.0
    cdef double acc_ase = 0.0
    cdef double acc_nli = 0.0
    cdef double gsnr = 0.0
    cdef double ase = 0.0
    cdef double nli = 0.0
    cdef double l_eff_a = 0.0
    cdef double l_eff = 0.0
    cdef double phi = 0.0
    cdef double sum_phi = 0.0
    cdef double power_ase = 0.0
    cdef double power_nli_span = 0.0

    # Formato de modulação
    cdef np.ndarray[np.double_t, ndim=1] phi_modulation_format = np.array(
        [1.0, 1.0, 2.0/3.0, 17.0/25.0, 69.0/100.0, 13.0/21.0],
        dtype=np.float64
    )

    cdef object link
    cdef object span
    cdef object running_service

    # Percorre cada link do caminho do serviço
    for link in current_service.path.links:
        for span in link.spans:
            # Cálculo da L_eff e L_eff_a
            l_eff_a = 1.0 / (2.0 * attenuation_normalized)
            l_eff = (
                1.0 - np.exp(-2.0 * attenuation_normalized * span.length * 1e3)
            ) / (2.0 * attenuation_normalized)

            # Inicia sum_phi para este span
            sum_phi = asinh(
                pi**2 * abs(beta_2) * (current_service.bandwidth**2)
                / (4.0 * attenuation_normalized)
            )

            # Contribuição NLI de outros serviços - usar cache com fallback
            if hasattr(env, 'topo_cache') and env.topo_cache is not None:
                running_services = env.topo_cache.get_running_services(link.node1, link.node2)
            else:
                running_services = []
            
            for running_service in running_services:
                if running_service.service_id != current_service.service_id:
                    phi = (
                        asinh(
                            pi**2 * abs(beta_2) * l_eff_a * running_service.bandwidth
                            * (
                                running_service.center_frequency
                                - current_service.center_frequency
                                + (running_service.bandwidth / 2.0)
                            )
                        )
                        - asinh(
                            pi**2 * abs(beta_2) * l_eff_a * running_service.bandwidth
                            * (
                                running_service.center_frequency
                                - current_service.center_frequency
                                - (running_service.bandwidth / 2.0)
                            )
                        )
                    ) - (
                        phi_modulation_format[running_service.current_modulation.spectral_efficiency - 1]
                        * (
                            running_service.bandwidth
                            / abs(running_service.center_frequency - current_service.center_frequency)
                        )
                        * (5.0 / 3.0)
                        * (l_eff / (span.length * 1e3))
                    )
                    sum_phi += phi

            # Potência de NLI e ASE
            power_nli_span = (
                (current_service.launch_power / current_service.bandwidth)**3
                * (8.0 / (27.0 * pi * abs(beta_2)))
                * (gamma**2)
                * l_eff
                * sum_phi
                * current_service.bandwidth
            )
            power_ase = (
                current_service.bandwidth
                * h_plank
                * current_service.center_frequency
                * (exp(2.0 * attenuation_normalized * span.length * 1e3) - 1.0)
                * noise_figure_normalized
            )

            # Somatórios
            acc_gsnr += 1.0 / (current_service.launch_power / (power_ase + power_nli_span))
            acc_ase  += 1.0 / (current_service.launch_power / power_ase)
            acc_nli  += 1.0 / (current_service.launch_power / power_nli_span)

    # Converte para dB
    gsnr = 10.0 * np.log10(1.0 / acc_gsnr)
    ase =  10.0 * np.log10(1.0 / acc_ase)
    nli =  10.0 * np.log10(1.0 / acc_nli)

    return gsnr, ase, nli


#############################
# 3) calculate_osnr_observation
#############################
cpdef double calculate_osnr_observation(
    object env,  # QRMSAEnv 
    tuple path_links,  # Lista de  Link
    double service_bandwidth,
    double service_center_frequency,
    int service_id,
    double service_launch_power,
    double gsnr_th
):
    cdef double beta_2 = -21.3e-27
    cdef double gamma = 1.3e-3
    cdef double h_plank = 6.626e-34
    cdef double acc_gsnr = 0.0
    cdef double acc_ase = 0.0   # se quiser usar
    cdef double acc_nli = 0.0   # se quiser usar
    cdef double gsnr = 0.0
    cdef double l_eff_a = 0.0
    cdef double l_eff = 0.0
    cdef double phi = 0.0
    cdef double sum_phi = 0.0
    cdef double power_ase = 0.0
    cdef double power_nli_span = 0.0

    # Modulation array
    cdef np.ndarray[np.double_t, ndim=1] phi_modulation_format = np.array(
        [1.0, 1.0, 2.0/3.0, 17.0/25.0, 69.0/100.0, 13.0/21.0],
        dtype=np.float64
    )

    cdef object link
    cdef object span
    cdef object running_service

    # Percorre cada link do path
    for link in path_links:
        # Otimização: Usar cache para acessar dados do link
        if hasattr(env, 'topo_cache') and env.topo_cache is not None:
            node1_id = env.topo_cache.get_node_id(link.node1)
            node2_id = env.topo_cache.get_node_id(link.node2)
            edge_idx = env.topo_cache.get_edge_index(node1_id, node2_id)
            link_data = env.topo_cache.edge_data[edge_idx]
        else:
            link_data = env.topology[link.node1][link.node2]
            
        for span in link_data["link"].spans:
            l_eff_a = 1.0 / (2.0 * span.attenuation_normalized)
            l_eff = (
                1.0 - exp(-2.0 * span.attenuation_normalized * span.length * 1e3)
            ) / (2.0 * span.attenuation_normalized)

            # sum_phi para este span
            sum_phi = asinh(
                pi**2
                * abs(beta_2)
                * (service_bandwidth**2)
                / (4.0 * span.attenuation_normalized)
            )

            # Contribuição dos serviços em execução - usar cache com fallback
            if hasattr(env, 'topo_cache') and env.topo_cache is not None:
                running_services = env.topo_cache.get_running_services(link.node1, link.node2)
            else:
                running_services = []
            
            for running_service in running_services:
                if running_service.service_id != service_id:
                    phi = (
                        asinh(
                            pi**2
                            * abs(beta_2)
                            * l_eff_a
                            * running_service.bandwidth
                            * (
                                running_service.center_frequency
                                - service_center_frequency
                                + (running_service.bandwidth / 2.0)
                            )
                        )
                        - asinh(
                            pi**2
                            * abs(beta_2)
                            * l_eff_a
                            * running_service.bandwidth
                            * (
                                running_service.center_frequency
                                - service_center_frequency
                                - (running_service.bandwidth / 2.0)
                            )
                        )
                    ) - (
                        phi_modulation_format[running_service.current_modulation.spectral_efficiency - 1]
                        * (
                            running_service.bandwidth
                            / abs(running_service.center_frequency - service_center_frequency)
                        )
                        * (5.0 / 3.0)
                        * (l_eff / (span.length * 1e3))
                    )
                    sum_phi += phi

            # Potência NLI e ASE
            power_nli_span = (
                (service_launch_power / service_bandwidth)**3
                * (8.0 / (27.0 * pi * abs(beta_2)))
                * (gamma**2)
                * l_eff
                * sum_phi
                * service_bandwidth
            )
            power_ase = (
                service_bandwidth
                * h_plank
                * service_center_frequency
                * (exp(2.0 * span.attenuation_normalized * span.length * 1e3) - 1.0)
                * span.noise_figure_normalized
            )

            # Somatório
            acc_gsnr += 1.0 / (service_launch_power / (power_ase + power_nli_span))

    # GSNR final
    gsnr = 10.0 * log10(1.0 / acc_gsnr)
    # Normalização com verificação para evitar divisão por zero
    cdef double normalized_gsnr
    if abs(gsnr_th) < 1e-10:  # Se gsnr_th é praticamente zero
        normalized_gsnr = gsnr  # Retornar valor não normalizado
    else:
        normalized_gsnr = np.round((gsnr - gsnr_th) / abs(gsnr_th), 10)
    return normalized_gsnr


#############################
# 4) compute_slot_osnr_vectorized
#############################
cpdef np.ndarray[np.float64_t, ndim=1] compute_slot_osnr_vectorized(
    object env,  # QRMSAEnv
    object path,  # Path object 
    np.ndarray[np.int32_t, ndim=1] available_slots,
    str qot_constraint = "ASE+NLI"
):
    """
    Calcula OSNR para todos os slots de uma vez usando vetorização.
    Movida do qrmsa.pyx para melhor organização.
    
    Args:
        env: QRMSAEnv environment
        path: Path object
        available_slots: Array de slots disponíveis
        qot_constraint: Tipo de constraint ("ASE+NLI", "DIST", etc.)
    """
    cdef int num_slots = available_slots.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] osnr_values = np.zeros(num_slots, dtype=np.float64)
    cdef double frequency_slot_bandwidth = env.channel_width * 1e9
    cdef double launch_power = 10 ** ((env.launch_power_dbm - 30) / 10)
    cdef int slot
    cdef double service_center_frequency
    cdef double gsnr_db, ase_db, nli_db
    
    # Criar um serviço temporário para simular cada slot
    cdef object temp_service = type('TempService', (), {
        'path': path,
        'initial_slot': 0,
        'number_slots': 1,
        'center_frequency': 0.0,
        'bandwidth': frequency_slot_bandwidth,
        'launch_power': launch_power,
        'service_id': getattr(env.current_service, 'service_id', -999)  # ID único para evitar conflitos
    })()
    
    # Calcular OSNR para cada slot individualmente usando a função original
    for slot in range(num_slots):
        if available_slots[slot] == 1:  # Slot disponível
            service_center_frequency = (
                env.frequency_start + 
                (frequency_slot_bandwidth * slot) + 
                (frequency_slot_bandwidth / 2.0)
            )
            
            # Configurar serviço temporário para este slot
            temp_service.initial_slot = slot
            temp_service.center_frequency = service_center_frequency
            
            # Usar função original calculate_osnr que retorna valores em dB
            gsnr_db, ase_db, nli_db = calculate_osnr(env, temp_service, qot_constraint)
            osnr_values[slot] = gsnr_db  # Retornar valor em dB
        else:
            osnr_values[slot] = -1.0  # Slot ocupado
    
    return osnr_values


#############################
# 5) validate_osnr_vectorized
#############################
cpdef bint validate_osnr_vectorized(
    object env,
    object path,
    np.ndarray[np.int32_t, ndim=1] available_slots,
    double tolerance=1e-6,
    str qot_constraint="ASE+NLI"
):
    """
    Valida a função OSNR vetorizada comparando com a implementação original.
    Retorna True se as diferenças estão dentro da tolerância especificada.
    """
    cdef int num_slots = available_slots.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] vectorized_values
    cdef np.ndarray[np.float64_t, ndim=1] individual_values = np.zeros(num_slots, dtype=np.float64)
    cdef double frequency_slot_bandwidth = env.channel_width * 1e9
    cdef double launch_power = 10 ** ((env.launch_power_dbm - 30) / 10)
    cdef double gsnr_th = 10.0
    cdef int slot
    cdef double service_center_frequency
    cdef double max_diff = 0.0
    cdef double diff
    cdef int errors = 0
    
    print(f"[DEBUG] Validating OSNR vectorized function...")
    
    # Calcular usando função vetorizada
    vectorized_values = compute_slot_osnr_vectorized(env, path, available_slots, qot_constraint)
    
    # Calcular individualmente para comparação
    for slot in range(num_slots):
        if available_slots[slot] == 1:
            service_center_frequency = (
                env.frequency_start + 
                (frequency_slot_bandwidth * slot) + 
                (frequency_slot_bandwidth / 2.0)
            )
            
            individual_values[slot] = calculate_osnr_observation(
                env,
                path.links,
                frequency_slot_bandwidth,
                service_center_frequency,
                env.current_service.service_id,
                launch_power,
                gsnr_th
            )
        else:
            individual_values[slot] = -1.0
    
    # Comparar resultados
    for slot in range(num_slots):
        diff = abs(vectorized_values[slot] - individual_values[slot])
        if diff > tolerance:
            print(f"[VALIDATION ERROR] Slot {slot}: vectorized={vectorized_values[slot]:.6f}, individual={individual_values[slot]:.6f}, diff={diff:.6f}")
            errors += 1
        if diff > max_diff:
            max_diff = diff
    
    print(f"[DEBUG] Validation complete: {errors} errors, max_diff={max_diff:.6f}")
    return errors == 0