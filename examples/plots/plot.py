import os
import re
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import argparse
import glob

def get_y_lim_min(episode_length):
    """
    Calcula o limite inferior do eixo Y com base no número de requisições por episódio.
    """
    return 1 / episode_length

def generate_plot(
    results_dir: str = "results",
    topology_name: str = "nobel-eu",
    y_metric: str = "episode_service_blocking_rate",
    output_dir: str = "figures",
    episode_length: int = 100000
):
    """
    Gera um gráfico genérico a partir dos resultados CSV.
    """
    print(f"\nIniciando geração de gráfico para métrica: {y_metric}")
    
    # Configuração
    markers = ("o", ">", "s", "D", "^", "v", "<", "p", "*")
    
    plt.figure(figsize=(6, 5)) # Slightly larger for better readability

    # Regex para capturar a carga do nome do arquivo
    # Novo formato: load_results_{topo}_{heuristic}_{band}_{load}.csv
    # Ex: load_results_nobel-eu_best_band_BandaC+L+S_100.0.csv
    # Vamos tentar ser flexíveis para suportar o formato antigo também se possível, mas focando no novo.
    
    # Regex mais robusto:
    # Captura tudo entre 'load_results_{topo}_' e '_{load}.csv' como "label" (heuristic + band)
    # E captura o load no final.
    
    # Pattern: load_results_{topo}_(.*)_(\d+(?:\.\d+)?)\.csv
    # Onde (.*) é o identificador da série
    
    escaped_topo = re.escape(topology_name)
    load_pattern = re.compile(rf"load_results_{escaped_topo}_(.+)_(\d+(?:\.\d+)?)\.csv")
    
    # Fallback para formato antigo: load_results_{topo}_{load}.csv
    old_load_pattern = re.compile(rf"load_results_{escaped_topo}_(\d+(?:\.\d+)?)\.csv")
    
    min_load = float('inf')
    max_load = 0
    
    min_y_lim = get_y_lim_min(episode_length)
    max_y_lim_filter = 1.0

    # Dicionário para agrupar dados por série (heuristic + band)
    # Key: label, Value: list of dicts {load, value}
    series_data = {}
    # Use glob to find files recursively in subdirectories
    search_pattern = os.path.join(results_dir, "**", f"load_results_{topology_name}*.csv")
    files = glob.glob(search_pattern, recursive=True)
    
    if not files:
        print(f"Nenhum arquivo encontrado em {results_dir} para a topologia {topology_name}")
        return

    print(f"Processando {len(files)} arquivos...")

    data = []
    
    # Regex for new format with heuristic and band
    # Format: load_results_{topology}_{heuristic}_{band}_{load}.csv
    # Example: load_results_nobel-eu_shortest_path_BandaC_100.0.csv
    # We need to be flexible because heuristic names can contain underscores
    # Strategy: Split by underscore.
    # The prefix is always "load_results_{topology_name}_"
    # The suffix is always "_{load}.csv"
    # Everything in between is "{heuristic}_{band}"
    # We know the band starts with "Banda"
    
    for file_path in files:
        filename = os.path.basename(file_path)
        try:
            # Extract load
            load_str = filename.split('_')[-1].replace('.csv', '')
            load = float(load_str)
            
            # Extract Heuristic and Band
            # Remove prefix and suffix
            prefix = f"load_results_{topology_name}_"
            suffix = f"_{load_str}.csv"
            
            if filename.startswith(prefix) and filename.endswith(suffix):
                middle = filename[len(prefix):-len(suffix)]
                
                # Find where "Banda" starts
                band_idx = middle.rfind("Banda")
                if band_idx != -1:
                    heuristic = middle[:band_idx].strip('_')
                    band = middle[band_idx:]
                else:
                    # Fallback for old format or unknown format
                    heuristic = "Unknown"
                    band = "Default"
            else:
                 # Skip files that don't match the topology prefix
                 continue

            df = pd.read_csv(file_path, comment='#')
            
            # Check if metric exists
            if y_metric not in df.columns:
                print(f"Aviso: Métrica '{y_metric}' não encontrada em {filename}. Pulando.")
                continue
                
            # Calculate mean of the last 10% episodes to get steady state or just mean of all
            # Using mean of all episodes as per original script logic (which plotted per load)
            # The original script plotted "Blocking Probability" vs "Load".
            # So we need one value per file (load).
            
            # If the metric is "episode_service_blocking_rate", it's the blocking probability.
            y_value = df[y_metric].mean()
            
            data.append({
                "Load": load,
                "Value": y_value,
                "Heuristic": heuristic,
                "Band": band,
                "Group": f"{heuristic} - {band}"
            })
            
        except ValueError:
            continue

    if not data:
        print("Nenhum dado válido extraído.")
        return

    df_plot = pd.DataFrame(data)
    
    # Sort by Load
    df_plot = df_plot.sort_values("Load")

    plt.figure(figsize=(10, 6))
    
    # Plot each group
    groups = df_plot["Group"].unique()
    for group in groups:
        subset = df_plot[df_plot["Group"] == group]
        plt.plot(subset["Load"], subset["Value"], marker='o', label=group)
    
    plt.xlabel("Load (Erlang)")
    
    # Custom Labeling for Blocking Probability
    if y_metric == "episode_service_blocking_rate":
        plt.ylabel("Blocking Probability")
        plt.title(f"Blocking Probability vs Load - {topology_name}")
        plt.yscale('log') # Blocking prob is usually log scale
    else:
        plt.ylabel(y_metric)
        plt.title(f"{y_metric} vs Load - {topology_name}")
        
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.legend()
    
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{topology_name}_{y_metric}_plot.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Gráfico salvo em: {output_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Gerador de Gráficos para OpticalNetworkingGym")
    parser.add_argument("-d", "--results-dir", default="results", help="Diretório dos resultados")
    parser.add_argument("-t", "--topology", default="nobel-eu", help="Nome da topologia")
    parser.add_argument("-m", "--metric", default="episode_service_blocking_rate", help="Métrica para o eixo Y")
    parser.add_argument("-o", "--output-dir", default="figures", help="Diretório de saída")
    parser.add_argument("-l", "--episode-length", type=int, default=100000, help="Tamanho do episódio (para limite Y)")
    
    args = parser.parse_args()
    
    generate_plot(
        results_dir=args.results_dir,
        topology_name=args.topology,
        y_metric=args.metric,
        output_dir=args.output_dir,
        episode_length=args.episode_length
    )