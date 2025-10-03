
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
    print(f"[DEBUG OSNR] Calculating OSNR for Service ID: {current_service.service_id}")
    print(f"[DEBUG OSNR] Service : {current_service}")
    print(f"[DEBUG OSNR] QOT Constraint: {qot_constraint}")
    print(f"band values: att={current_service.current_band.attenuation_normalized}, nf={current_service.current_band.noise_figure_normalized}")
    print(f"launch power (W): {current_service.launch_power}")

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
                l_eff_a = 1.0 / (2.0 * current_service.current_band.attenuation_normalized)
                l_eff = (
                    1.0 - np.exp(-2.0 * current_service.current_band.attenuation_normalized * span.length * 1e3)
                ) / (2.0 * current_service.current_band.attenuation_normalized)
                sum_phi = asinh(
                    pi**2 * abs(beta_2) * (current_service.bandwidth**2) /
                    (4.0 * current_service.current_band.attenuation_normalized)
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
                * (exp(2.0 * current_service.current_band.attenuation_normalized * span.length * 1e3) - 1.0)
                * current_service.current_band.noise_figure_normalized
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
    # DEBUG: Print parâmetros de entrada para observation
    print(f"[DEBUG OSNR_OBS] Service ID: {service_id}, Launch power: {service_launch_power:.2f} dBm")
    print(f"[DEBUG OSNR_OBS] Bandwidth: {service_bandwidth:.2e} Hz, Center freq: {service_center_frequency:.2e} Hz")
    print(f"[DEBUG OSNR_OBS] GSNR threshold: {gsnr_th:.2f} dB, Band: {band.name if band else 'None'}")
    
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
                noise_figure_normalized = 10.0 ** (band.noise_figure_normalized / 10.0)
         
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
    
    # DEBUG: Print resultados da observation
    print(f"[DEBUG OSNR_OBS] GSNR calculado: {gsnr:.2f} dB, Normalizado: {normalized_gsnr:.6f}")
    print(f"[DEBUG OSNR_OBS] Acc_gsnr: {acc_gsnr:.2e}")
    
    return normalized_gsnr