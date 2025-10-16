import os
import re
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Acessa o valor de episode_length do script de simulação
try:
    import multiband_graph
    EPISODE_LENGTH = multiband_graph.parse_arguments().episode_length
except (ImportError, AttributeError):
    print("Aviso: Não foi possível importar 'multiband_graph'. Usando EPISODE_LENGTH padrão de 1000.")
    EPISODE_LENGTH = 100000

def get_y_lim_min(episode_length):
    """
    Calcula o limite inferior do eixo Y com base no número de requisições por episódio.
    """
    return 1 / episode_length

def gerar_grafico_probabilidade_bloqueio():
    """
    Este script lê os resultados da simulação a partir de pastas de banda
    e gera um gráfico da Probabilidade de Bloqueio de Serviço vs. Carga na Rede.
    """
    print("A iniciar a geração do gráfico de probabilidade de bloqueio...")

    # CONFIGURAÇÃO

    topology_name = "nobel-eu"

    base_results_dir = "results"
    band_folders = ["BandaC+S", "BandaL", "BandaS"]
    markers = ("o", ">", "s")

    plt.figure(figsize=(5, 4))

    load_pattern = re.compile(r"load_results_[\w\-]+_(\d+)\.csv")
    min_load = float('inf')
    max_load = 0
    
    min_y_lim = get_y_lim_min(EPISODE_LENGTH)
    # Define o limite superior para a filtragem.
    max_y_lim_filter = 1

    # Loop principal para processar cada pasta de banda.
    for idx, band_folder in enumerate(band_folders):
        all_data = []
        
        full_band_path = os.path.join(base_results_dir, band_folder)
        print(f"\nColetando dados para a pasta: '{full_band_path}'...")
        
        if not os.path.exists(full_band_path):
            print(f"Aviso: Pasta '{full_band_path}' não encontrada. Pulando.")
            continue
            
        files = [f for f in os.listdir(full_band_path) if f.endswith(".csv")]

        if not files:
            print(f"ERRO: Nenhum ficheiro CSV encontrado na pasta '{band_folder}'.")
            continue

        print(f"Arquivos encontrados em {band_folder}: {files}")

        for f in files:
            match = load_pattern.match(f)
            if match:
                load = int(match.group(1))
                full_file_path = os.path.join(full_band_path, f)
                try:
                    data_load = pd.read_csv(full_file_path, skiprows=1)

                    data_load = data_load[
                        (data_load['episode_service_blocking_rate'] > 0) & 
                        (data_load['episode_service_blocking_rate'] <= max_y_lim_filter)
                    ]
                    
                    if not data_load.empty:
                        data_load["load"] = load
                        all_data.append(data_load)
                        
                        if load > max_load:
                            max_load = load
                        if load < min_load:
                            min_load = load
                    else:
                        print(f"Aviso: O ficheiro {f} contém apenas valores de bloqueio fora do intervalo desejado e será ignorado.")

                except Exception as e:
                    print(f"Aviso: Não foi possível ler ou processar o ficheiro {full_file_path}. Erro: {e}")
            else:
                print(f"Aviso: Nome de arquivo não corresponde ao padrão esperado: {f}")

        if not all_data:
            print(f"ERRO: Nenhum dado válido foi carregado em '{band_folder}'.")
            continue

        data_loads = pd.concat(all_data, axis=0, ignore_index=True)
        print(f"Dados carregados para '{band_folder}': cargas {sorted(data_loads['load'].unique())}")

        mean_blocking_rate = data_loads.groupby("load").mean()["episode_service_blocking_rate"]

        plt.plot(
            mean_blocking_rate.index,
            mean_blocking_rate.values,
            label=f"{band_folder}",
            marker=markers[idx],
            markersize=6,
            linewidth=1,
            mec="black",
        )

    # CONFIGURAÇAO DO GRAFICO

    plt.xlabel("Offered Traffic Load [Erlang]")
    plt.ylabel("Blocking probability")
    plt.gca().set_yscale("log")

    # Define os limites do eixo Y 
    plt.ylim(min_y_lim, 10e-1)

    if max_load > 0 and min_load != float('inf'):
        plt.xlim(min_load - 50, max_load + 50)
    else:
        plt.xlim(0, 1000)

    plt.grid(visible=True, which="major", axis="both", ls="--")
    plt.grid(visible=True, which="minor", axis="both", ls=":")

    plt.legend(frameon=False, loc='lower right')

    plt.tight_layout()

    figures_path = "figures"
    os.makedirs(figures_path, exist_ok=True)
    output_filename = f"{topology_name}_load_blocking_rate_multi_band_graph"
    plt.savefig(os.path.join(figures_path, f"{output_filename}.png"))
    plt.savefig(os.path.join(figures_path, f"{output_filename}.pdf"))

    print(f"\nGráfico gerado com sucesso!")
    print(f"Ficheiros guardados em '{figures_path}/{output_filename}.png' e '.pdf'")

if __name__ == "__main__":
    gerar_grafico_probabilidade_bloqueio()