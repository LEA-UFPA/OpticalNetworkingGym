
#####################
# Imports e Typing
#####################
from math import pi
from libc.math cimport asinh, pi, exp, log10

import cython
cimport numpy as np
import numpy as np

import typing

if typing.TYPE_CHECKING:
    from optical_networking_gym.envs.qrmsa import QRMSAEnv


#############################
# 1) calculate_osnr
#############################
cpdef calculate_osnr(env: QRMSAEnv, current_service: object):
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

    # Percorre cada link do caminho do serviço
    for link in current_service.path.links:
        # Otimização: Usar cache para acessar dados do link
        if hasattr(env, 'topo_cache') and env.topo_cache is not None:
            node1_id = env.topo_cache.get_node_id(link.node1)
            node2_id = env.topo_cache.get_node_id(link.node2)
            edge_idx = env.topo_cache.get_edge_index(node1_id, node2_id)
            link_data = env.topo_cache.edge_data[edge_idx]
        else:
            link_data = env.topology[link.node1][link.node2]
        
        # Para cada span do link
        for span in link_data["link"].spans:
            # Usar parâmetros da banda do serviço se disponível
            if hasattr(current_service, 'current_band') and current_service.current_band is not None:
                # Converter dB para linear
                attenuation_normalized = (current_service.current_band.attenuation_db_km / 1000.0) / (10.0 * log10(exp(1)))
                noise_figure_normalized = 10.0 ** (current_service.current_band.noise_figure_db / 10.0)
            else:
                # Usar parâmetros originais do span
                attenuation_normalized = span.attenuation_normalized
                noise_figure_normalized = span.noise_figure_normalized
            
            # Cálculo da L_eff e L_eff_a
            l_eff_a = 1.0 / (2.0 * attenuation_normalized)
            l_eff = (
                1.0 - np.exp(-2.0 * attenuation_normalized * span.length * 1e3)
            ) / (2.0 * attenuation_normalized)

            # Inicia sum_phi para este span
            sum_phi = asinh(
                pi**2 * abs(beta_2) * (current_service.bandwidth**2) /
                (4.0 * attenuation_normalized)
            )

            # Soma das contribuições NLI de outros serviços rodando nesse link
            # Usar cache do environment (com fallback durante inicialização)
            if hasattr(env, 'topo_cache') and env.topo_cache is not None:
                running_services = env.topo_cache.get_running_services(link.node1, link.node2)
            else:
                # Fallback durante inicialização - retorna lista vazia
                running_services = []
            
            for running_service in running_services:
                if running_service.service_id != current_service.service_id:
                    try:
                        # Cálculo do phi
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
                        print("current_time: ", env.current_time)
                        print(f'current_service: {current_service}\n')
                        print(f'running_service: {running_service}\n')
                        # Usar cache para obter índice do link (se disponível)
                        # Usar cache para acessar dados do link (com fallback)
                        if hasattr(env, 'topo_cache') and env.topo_cache is not None:
                            node1_id = env.topo_cache.get_node_id(link.node1)
                            node2_id = env.topo_cache.get_node_id(link.node2)
                            index = env.topo_cache.get_edge_index(node1_id, node2_id)
                        else:
                            # Fallback durante inicialização
                            index = 0
                        # Usar cache para debug (opcional)
                        if hasattr(env, 'topo_cache') and env.topo_cache is not None and env.topo_cache.available_slots is not None:
                            print(f'Link {link.node1,link.node2}: {env.topo_cache.available_slots[index,:]}\n\n')
                        else:
                            print(f'Link {link.node1,link.node2}: {env.topology.graph["available_slots"][index,:]}\n\n')
                        print(f"running services:" )
                        # Usar cache para debug de serviços (com fallback)
                        if hasattr(env, 'topo_cache') and env.topo_cache is not None:
                            services_to_print = env.topo_cache.get_running_services(link.node1, link.node2)
                        else:
                            services_to_print = []
                        for service in services_to_print:
                            print(f"ID: {service.service_id}, src: {service.source}, tgt: {service.destination}, Path: {service.path}, init_slot: {service.initial_slot}, numb_slots: {service.number_slots}, BW: {service.bandwidth}, center_freq: {service.center_frequency}, mod: {service.current_modulation}, OSNR: {service.OSNR}, ASE: {service.ASE}, NLI: {service.NLI}\n")
                        raise ValueError("Error in calculate_osnr")

            # Potência de NLI no span
            power_nli_span = (
                (current_service.launch_power / current_service.bandwidth)**3
                * (8.0 / (27.0 * pi * abs(beta_2)))
                * (gamma**2)
                * l_eff
                * sum_phi
                * current_service.bandwidth
            )

            # Potência de ASE no span
            power_ase = (
                current_service.bandwidth
                * h_plank
                * current_service.center_frequency
                * (exp(2.0 * attenuation_normalized * span.length * 1e3) - 1.0)
                * noise_figure_normalized
            )

            # Somatório para GSNR, ASE e NLI
            #   --> 1 / (SNR) = 1 / (P_signal / P_ruído) = P_ruído / P_signal
            #       mas P_signal = current_service.launch_power
            #   --> SNR_total = launch_power / (power_ase + power_nli_span)
            #   --> SNR_ase   = launch_power / power_ase
            #   --> SNR_nli   = launch_power / power_nli_span
            acc_gsnr += 1.0 / (current_service.launch_power / (power_ase + power_nli_span))
            acc_ase  += 1.0 / (current_service.launch_power / power_ase)
            acc_nli  += 1.0 / (current_service.launch_power / power_nli_span)

    # Converte cada acúmulo para dB
    gsnr = 10.0 * np.log10(1.0 / acc_gsnr)
    ase =  10.0 * np.log10(1.0 / acc_ase)
    nli =  10.0 * np.log10(1.0 / acc_nli)

    return gsnr, ase, nli


#############################
# 2) calculate_osnr_default_attenuation
#############################
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
    double gsnr_th,
    object band = None  # Novo parâmetro para banda
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
            # Usar parâmetros da banda se fornecida, senão usar span original
            if band is not None:
                # Converter dB para linear
                attenuation_normalized = (band.attenuation_db_km / 1000.0) / (10.0 * log10(exp(1)))
                noise_figure_normalized = 10.0 ** (band.noise_figure_db / 10.0)
            else:
                # Usar parâmetros originais do span
                attenuation_normalized = span.attenuation_normalized
                noise_figure_normalized = span.noise_figure_normalized
            
            l_eff_a = 1.0 / (2.0 * attenuation_normalized)
            l_eff = (
                1.0 - exp(-2.0 * attenuation_normalized * span.length * 1e3)
            ) / (2.0 * attenuation_normalized)

            # sum_phi para este span
            sum_phi = asinh(
                pi**2
                * abs(beta_2)
                * (service_bandwidth**2)
                / (4.0 * attenuation_normalized)
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
                * (exp(2.0 * attenuation_normalized * span.length * 1e3) - 1.0)
                * noise_figure_normalized
            )

            # Somatório
            acc_gsnr += 1.0 / (service_launch_power / (power_ase + power_nli_span))

    # GSNR final
    gsnr = 10.0 * log10(1.0 / acc_gsnr)
    # Normalização
    cdef double normalized_gsnr = np.round((gsnr - gsnr_th) / abs(gsnr_th), 10)
    return normalized_gsnr