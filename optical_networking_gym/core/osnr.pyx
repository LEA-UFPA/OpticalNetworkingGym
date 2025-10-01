
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
    # DEBUG: Print parâmetros de entrada
    print(f"[DEBUG OSNR] Service ID: {current_service.service_id}, Path: {current_service.path.node_list if hasattr(current_service.path, 'node_list') else 'N/A'}")
    print(f"[DEBUG OSNR] Launch power: {current_service.launch_power:.2f} dBm, Bandwidth: {current_service.bandwidth:.2e} Hz, Center freq: {current_service.center_frequency:.2e} Hz")
    print(f"[DEBUG OSNR] Modulation: {current_service.current_modulation.name if current_service.current_modulation else 'None'}")
    
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
    
    # DEBUG: Variáveis para capturar valores durante o cálculo
    cdef double total_span_length = 0.0
    cdef double total_phi_sum = 0.0
    cdef int total_spans = 0
    
    # DEBUG: Variáveis para operações matemáticas
    cdef double snr_gsnr = 0.0
    cdef double snr_ase = 0.0
    cdef double snr_nli = 0.0
    cdef double inv_gsnr = 0.0
    cdef double inv_ase = 0.0
    cdef double inv_nli = 0.0

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
            
            # DEBUG: Capturar valores para o print detalhado
            total_span_length += span.length
            total_spans += 1

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
            
            # DEBUG: Print operações matemáticas do NLI
            print(f"[DEBUG OSNR MATEMÁTICA] === CÁLCULO NLI - Span {total_spans} ===")
            print(f"[DEBUG OSNR MATEMÁTICA] Passo 1: (launch_power / bandwidth)^3")
            print(f"[DEBUG OSNR MATEMÁTICA]   = ({current_service.launch_power:.6f} / {current_service.bandwidth:.6e})^3")
            print(f"[DEBUG OSNR MATEMÁTICA]   = ({current_service.launch_power/current_service.bandwidth:.6e})^3")
            print(f"[DEBUG OSNR MATEMÁTICA]   = {(current_service.launch_power/current_service.bandwidth)**3:.6e}")
            print(f"[DEBUG OSNR MATEMÁTICA] Passo 2: 8/(27*pi*|beta_2|)")
            print(f"[DEBUG OSNR MATEMÁTICA]   = 8/(27*{pi:.6f}*{abs(beta_2):.6e})")
            print(f"[DEBUG OSNR MATEMÁTICA]   = 8/{27.0*pi*abs(beta_2):.6e}")
            print(f"[DEBUG OSNR MATEMÁTICA]   = {8.0/(27.0*pi*abs(beta_2)):.6e}")
            print(f"[DEBUG OSNR MATEMÁTICA] Passo 3: gamma^2 = {gamma:.6e}^2 = {gamma**2:.6e}")
            print(f"[DEBUG OSNR MATEMÁTICA] Passo 4: l_eff = {l_eff:.6e}")
            print(f"[DEBUG OSNR MATEMÁTICA] Passo 5: sum_phi = {sum_phi:.6e}")
            print(f"[DEBUG OSNR MATEMÁTICA] Passo 6: bandwidth = {current_service.bandwidth:.6e}")
            print(f"[DEBUG OSNR MATEMÁTICA] RESULTADO NLI: power_nli_span = {power_nli_span:.6e}")
            print(f"[DEBUG OSNR MATEMÁTICA] ======================================")

            # Potência de ASE no span
            power_ase = (
                current_service.bandwidth
                * h_plank
                * current_service.center_frequency
                * (exp(2.0 * attenuation_normalized * span.length * 1e3) - 1.0)
                * noise_figure_normalized
            )
            
            # DEBUG: Print operações matemáticas do ASE
            print(f"[DEBUG OSNR MATEMÁTICA] === CÁLCULO ASE - Span {total_spans} ===")
            print(f"[DEBUG OSNR MATEMÁTICA] Passo 1: bandwidth = {current_service.bandwidth:.6e}")
            print(f"[DEBUG OSNR MATEMÁTICA] Passo 2: h_plank = {h_plank:.6e}")
            print(f"[DEBUG OSNR MATEMÁTICA] Passo 3: center_frequency = {current_service.center_frequency:.6e}")
            print(f"[DEBUG OSNR MATEMÁTICA] Passo 4: attenuation_normalized = {attenuation_normalized:.6e}")
            print(f"[DEBUG OSNR MATEMÁTICA] Passo 5: span.length = {span.length:.2f} km")
            print(f"[DEBUG OSNR MATEMÁTICA] Passo 6: 2*attenuation*length*1000 = 2*{attenuation_normalized:.6e}*{span.length:.2f}*1000")
            print(f"[DEBUG OSNR MATEMÁTICA]   = {2.0 * attenuation_normalized * span.length * 1e3:.6e}")
            print(f"[DEBUG OSNR MATEMÁTICA] Passo 7: exp(...) - 1 = exp({2.0 * attenuation_normalized * span.length * 1e3:.6e}) - 1")
            print(f"[DEBUG OSNR MATEMÁTICA]   = {exp(2.0 * attenuation_normalized * span.length * 1e3):.6e} - 1")
            print(f"[DEBUG OSNR MATEMÁTICA]   = {exp(2.0 * attenuation_normalized * span.length * 1e3) - 1.0:.6e}")
            print(f"[DEBUG OSNR MATEMÁTICA] Passo 8: noise_figure_normalized = {noise_figure_normalized:.6e}")
            print(f"[DEBUG OSNR MATEMÁTICA] RESULTADO ASE: power_ase = {power_ase:.6e}")
            print(f"[DEBUG OSNR MATEMÁTICA] ======================================")

            # Somatório para GSNR, ASE e NLI
            #   --> 1 / (SNR) = 1 / (P_signal / P_ruído) = P_ruído / P_signal
            #       mas P_signal = current_service.launch_power
            #   --> SNR_total = launch_power / (power_ase + power_nli_span)
            #   --> SNR_ase   = launch_power / power_ase
            #   --> SNR_nli   = launch_power / power_nli_span
            
            # DEBUG: Print operações de somatório
            print(f"[DEBUG OSNR MATEMÁTICA] === SOMATÓRIOS - Span {total_spans} ===")
            print(f"[DEBUG OSNR MATEMÁTICA] launch_power = {current_service.launch_power:.6f} dBm")
            print(f"[DEBUG OSNR MATEMÁTICA] power_ase = {power_ase:.6e}")
            print(f"[DEBUG OSNR MATEMÁTICA] power_nli_span = {power_nli_span:.6e}")
            print(f"[DEBUG OSNR MATEMÁTICA] power_ase + power_nli_span = {power_ase + power_nli_span:.6e}")
            
            # Cálculo dos inversos de SNR
            snr_gsnr = current_service.launch_power / (power_ase + power_nli_span)
            snr_ase = current_service.launch_power / power_ase
            snr_nli = current_service.launch_power / power_nli_span
            
            print(f"[DEBUG OSNR MATEMÁTICA] SNR_GSNR = launch_power / (power_ase + power_nli)")
            print(f"[DEBUG OSNR MATEMÁTICA]   = {current_service.launch_power:.6f} / {power_ase + power_nli_span:.6e}")
            print(f"[DEBUG OSNR MATEMÁTICA]   = {snr_gsnr:.6e}")
            print(f"[DEBUG OSNR MATEMÁTICA] SNR_ASE = launch_power / power_ase")
            print(f"[DEBUG OSNR MATEMÁTICA]   = {current_service.launch_power:.6f} / {power_ase:.6e}")
            print(f"[DEBUG OSNR MATEMÁTICA]   = {snr_ase:.6e}")
            print(f"[DEBUG OSNR MATEMÁTICA] SNR_NLI = launch_power / power_nli")
            print(f"[DEBUG OSNR MATEMÁTICA]   = {current_service.launch_power:.6f} / {power_nli_span:.6e}")
            print(f"[DEBUG OSNR MATEMÁTICA]   = {snr_nli:.6e}")
            
            # Inversos (para o acúmulo)
            inv_gsnr = 1.0 / snr_gsnr
            inv_ase = 1.0 / snr_ase
            inv_nli = 1.0 / snr_nli
            
            print(f"[DEBUG OSNR MATEMÁTICA] 1/SNR_GSNR = 1/{snr_gsnr:.6e} = {inv_gsnr:.6e}")
            print(f"[DEBUG OSNR MATEMÁTICA] 1/SNR_ASE = 1/{snr_ase:.6e} = {inv_ase:.6e}")
            print(f"[DEBUG OSNR MATEMÁTICA] 1/SNR_NLI = 1/{snr_nli:.6e} = {inv_nli:.6e}")
            
            acc_gsnr += inv_gsnr
            acc_ase  += inv_ase
            acc_nli  += inv_nli
            
            print(f"[DEBUG OSNR MATEMÁTICA] acc_gsnr += {inv_gsnr:.6e} → acc_gsnr = {acc_gsnr:.6e}")
            print(f"[DEBUG OSNR MATEMÁTICA] acc_ase += {inv_ase:.6e} → acc_ase = {acc_ase:.6e}")
            print(f"[DEBUG OSNR MATEMÁTICA] acc_nli += {inv_nli:.6e} → acc_nli = {acc_nli:.6e}")
            print(f"[DEBUG OSNR MATEMÁTICA] ======================================")
            
            # DEBUG: Capturar sum_phi total
            total_phi_sum += sum_phi

    # Converte cada acúmulo para dB
    print(f"\n[DEBUG OSNR MATEMÁTICA] === CONVERSÃO FINAL PARA dB ===")
    print(f"[DEBUG OSNR MATEMÁTICA] acc_gsnr final = {acc_gsnr:.6e}")
    print(f"[DEBUG OSNR MATEMÁTICA] acc_ase final = {acc_ase:.6e}")
    print(f"[DEBUG OSNR MATEMÁTICA] acc_nli final = {acc_nli:.6e}")
    
    print(f"[DEBUG OSNR MATEMÁTICA] GSNR = 10 * log10(1 / acc_gsnr)")
    print(f"[DEBUG OSNR MATEMÁTICA]   = 10 * log10(1 / {acc_gsnr:.6e})")
    print(f"[DEBUG OSNR MATEMÁTICA]   = 10 * log10({1.0/acc_gsnr:.6e})")
    gsnr = 10.0 * np.log10(1.0 / acc_gsnr)
    print(f"[DEBUG OSNR MATEMÁTICA]   = 10 * {np.log10(1.0 / acc_gsnr):.6f}")
    print(f"[DEBUG OSNR MATEMÁTICA]   = {gsnr:.6f} dB")
    
    print(f"[DEBUG OSNR MATEMÁTICA] ASE = 10 * log10(1 / acc_ase)")
    print(f"[DEBUG OSNR MATEMÁTICA]   = 10 * log10(1 / {acc_ase:.6e})")
    print(f"[DEBUG OSNR MATEMÁTICA]   = 10 * log10({1.0/acc_ase:.6e})")
    ase =  10.0 * np.log10(1.0 / acc_ase)
    print(f"[DEBUG OSNR MATEMÁTICA]   = 10 * {np.log10(1.0 / acc_ase):.6f}")
    print(f"[DEBUG OSNR MATEMÁTICA]   = {ase:.6f} dB")
    
    print(f"[DEBUG OSNR MATEMÁTICA] NLI = 10 * log10(1 / acc_nli)")
    print(f"[DEBUG OSNR MATEMÁTICA]   = 10 * log10(1 / {acc_nli:.6e})")
    print(f"[DEBUG OSNR MATEMÁTICA]   = 10 * log10({1.0/acc_nli:.6e})")
    nli =  10.0 * np.log10(1.0 / acc_nli)
    print(f"[DEBUG OSNR MATEMÁTICA]   = 10 * {np.log10(1.0 / acc_nli):.6f}")
    print(f"[DEBUG OSNR MATEMÁTICA]   = {nli:.6f} dB")
    
    print(f"[DEBUG OSNR MATEMÁTICA] ===== RESULTADO FINAL DA OSNR =====")
    print(f"[DEBUG OSNR MATEMÁTICA] GSNR = {gsnr:.6f} dB")
    print(f"[DEBUG OSNR MATEMÁTICA] ASE = {ase:.6f} dB") 
    print(f"[DEBUG OSNR MATEMÁTICA] NLI = {nli:.6f} dB")
    print(f"[DEBUG OSNR MATEMÁTICA] ======================================")

    # DEBUG: Print resultados calculados
    print(f"[DEBUG OSNR] Resultados calculados - GSNR: {gsnr:.2f} dB, ASE: {ase:.2f} dB, NLI: {nli:.2f} dB")
    print(f"[DEBUG OSNR] Acc_gsnr: {acc_gsnr:.2e}, Acc_ase: {acc_ase:.2e}, Acc_nli: {acc_nli:.2e}")
    
    # DEBUG: Print detalhado com todos os valores utilizados no cálculo
    print(f"\n[DEBUG OSNR DETALHADO] ===== VALORES UTILIZADOS NO CÁLCULO DE OSNR =====")
    print(f"[DEBUG OSNR DETALHADO] Service ID: {current_service.service_id}")
    print(f"[DEBUG OSNR DETALHADO] Launch Power: {current_service.launch_power:.6f} dBm = {10**(current_service.launch_power/10):.6e} W")
    print(f"[DEBUG OSNR DETALHADO] Bandwidth: {current_service.bandwidth:.6e} Hz")
    print(f"[DEBUG OSNR DETALHADO] Center Frequency: {current_service.center_frequency:.6e} Hz")
    print(f"[DEBUG OSNR DETALHADO] Modulation: {current_service.current_modulation.name if current_service.current_modulation else 'None'}")
    print(f"[DEBUG OSNR DETALHADO] Total Span Length: {total_span_length:.2f} km")
    print(f"[DEBUG OSNR DETALHADO] Total Spans: {total_spans}")
    print(f"[DEBUG OSNR DETALHADO] Total Phi Sum: {total_phi_sum:.6e}")
    print(f"[DEBUG OSNR DETALHADO] Beta_2: {beta_2:.6e} s²/m")
    print(f"[DEBUG OSNR DETALHADO] Gamma: {gamma:.6e} /(W·m)")
    print(f"[DEBUG OSNR DETALHADO] h_plank: {h_plank:.6e} J·s")
    print(f"[DEBUG OSNR DETALHADO] Acc_GSNR: {acc_gsnr:.6e}")
    print(f"[DEBUG OSNR DETALHADO] Acc_ASE: {acc_ase:.6e}")
    print(f"[DEBUG OSNR DETALHADO] Acc_NLI: {acc_nli:.6e}")
    print(f"[DEBUG OSNR DETALHADO] ===== RESULTADO FINAL =====")
    print(f"[DEBUG OSNR DETALHADO] GSNR: {gsnr:.6f} dB")
    print(f"[DEBUG OSNR DETALHADO] ASE: {ase:.6f} dB")
    print(f"[DEBUG OSNR DETALHADO] NLI: {nli:.6f} dB")
    print(f"[DEBUG OSNR DETALHADO] ===========================================\n")
    
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
    
    # DEBUG: Print resultados da observation
    print(f"[DEBUG OSNR_OBS] GSNR calculado: {gsnr:.2f} dB, Normalizado: {normalized_gsnr:.6f}")
    print(f"[DEBUG OSNR_OBS] Acc_gsnr: {acc_gsnr:.2e}")
    
    return normalized_gsnr