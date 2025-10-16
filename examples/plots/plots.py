import os
import re
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def read_simulation_parameters(file_path):
    params = {}
    band_spec_lines = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            header_lines = [line for line in f if line.startswith('#')]

        # Process key-value pairs
        for line in header_lines:
            cleaned_line = line.strip('# ').strip()
            match = re.match(r'([^:]+):\s*(.+)', cleaned_line)
            if match:
                key = match.group(1).strip()
                value = match.group(2).strip()
                params[key] = value

        # Process band specifications table
        table_started = False
        for line in header_lines:
            cleaned_line = line.strip('# ').strip()
            if 'Band Specifications' in cleaned_line:
                table_started = True
                band_spec_lines.append(cleaned_line)
                continue
            if table_started and cleaned_line.startswith('|'):
                band_spec_lines.append(cleaned_line)

        params['band_spec_lines'] = band_spec_lines

    except FileNotFoundError:
        print(f"Aviso: Arquivo de parâmetros não encontrado em {file_path}")
    except Exception as e:
        print(f"Erro ao ler parâmetros de {file_path}: {e}")
    
    return params

def format_band_specs(band_spec_lines):
    if not band_spec_lines:
        return "Band specifications not available."
    return '\n'.join(band_spec_lines)

def gerar_grafico_probabilidade_bloqueio():
    """
    Este script lê os resultados da simulação e gera um gráfico 
    da Probabilidade de Bloqueio de Serviço vs. Carga na Rede,
    incluindo os parâmetros da simulação no gráfico.
    """
    print("A iniciar a geração do gráfico de probabilidade de bloqueio...")

    # CONFIGURAÇÃO
    base_results_dir = "results"
    
    fig, ax = plt.subplots(figsize=(10, 6))

    load_pattern = re.compile(r"load_results_([\w\-]+)_(\d+)\.csv")
    
    all_data = []
    simulation_params = {}
    
    if not os.path.exists(base_results_dir):
        print(f"ERRO: Pasta de resultados '{base_results_dir}' não encontrada.")
        return

    files = [f for f in os.listdir(base_results_dir) if f.endswith(".csv")]
    files.sort() # Sort to get a predictable file for parameter reading

    if not files:
        print(f"ERRO: Nenhum ficheiro CSV encontrado na pasta '{base_results_dir}'.")
        return

    print(f"Arquivos encontrados: {files}")

    # Ler os parâmetros do primeiro ficheiro encontrado
    simulation_params = read_simulation_parameters(os.path.join(base_results_dir, files[0]))
    
    min_load = float('inf')
    max_load = 0

    for f in files:
        match = load_pattern.match(f)
        if match:
            topology_name_from_file = match.group(1)
            load = int(match.group(2))
            full_file_path = os.path.join(base_results_dir, f)
            try:
                data_load = pd.read_csv(full_file_path, comment='#')
                
                if not data_load.empty:
                    data_load["load"] = load
                    all_data.append(data_load)
                    
                    if load > max_load:
                        max_load = load
                    if load < min_load:
                        min_load = load
                else:
                    print(f"Aviso: O ficheiro {f} está vazio ou não contém dados válidos.")

            except Exception as e:
                print(f"Aviso: Não foi possível ler ou processar o ficheiro {full_file_path}. Erro: {e}")
        else:
            print(f"Aviso: Nome de arquivo não corresponde ao padrão esperado: {f}")

    if not all_data:
        print("ERRO: Nenhum dado válido foi carregado.")
        return

    data_loads = pd.concat(all_data, axis=0, ignore_index=True)
    mean_blocking_rate = data_loads.groupby("load").mean()["episode_service_blocking_rate"]

    ax.plot(
        mean_blocking_rate.index,
        mean_blocking_rate.values,
        label="Probabilidade de Bloqueio",
        marker='o',
        markersize=6,
        linewidth=1,
        mec="black",
    )

    # CONFIGURAÇAO DO GRAFICO
    ax.set_xlabel("Offered Traffic Load [Erlang]")
    ax.set_ylabel("Blocking probability")
    ax.set_yscale("log")

    episode_length_str = simulation_params.get('Episode Length', '100000')
    min_y_lim = 1 / int(episode_length_str) if int(episode_length_str) > 0 else 0.001
    ax.set_ylim(bottom=min_y_lim, top=1)

    if max_load > 0 and min_load != float('inf'):
        ax.set_xlim(min_load - 50, max_load + 50)
    else:
        ax.set_xlim(0, 1000)

    ax.grid(visible=True, which="major", axis="both", ls="--")
    ax.grid(visible=True, which="minor", axis="both", ls=":")
    ax.legend(frameon=False, loc='lower right')

    # Adicionar parâmetros da simulação ao gráfico
    if simulation_params:
        
        other_params = (
            f"Topology: {simulation_params.get('Topology', 'N/A')}\n"
            f"Heuristic: {simulation_params.get('Heuristic', 'N/A')}\n"
            f"Episodes: {simulation_params.get('Episodes', 'N/A')}\n"
            f"Episode Length: {simulation_params.get('Episode Length', 'N/A')}\n"
            f"Bit Rates: {simulation_params.get('Bit Rates', 'N/A')}\n"
            f"Modulations: {simulation_params.get('Modulations', 'N/A')}\n"
            f"Span Length: {simulation_params.get('Span Length', 'N/A')}"
        )
        
        band_specs_str = format_band_specs(simulation_params.get('band_spec_lines', []))
        
        full_param_text = f"Simulation Parameters\n{'-'*30}\n{other_params}\n\n{band_specs_str}"

        # Use ax.text to position relative to the axes, outside the plot area
        fig.text(0.5, -0.4, full_param_text, ha='center', va='bottom', fontsize=8,
                 bbox=dict(boxstyle="round,pad=0.5", fc='whitesmoke', alpha=0.7),
                 fontfamily='monospace')

    fig.tight_layout()
    
    topology_name = simulation_params.get('Topology', 'unknown_topology')
    figures_path = "figures"
    os.makedirs(figures_path, exist_ok=True)
    output_filename = f"{topology_name}_load_blocking_rate_graph"
    plt.savefig(os.path.join(figures_path, f"{output_filename}.png"), bbox_inches='tight')

    print(f"\nGráfico gerado com sucesso!")
    print(f"Ficheiros guardados em '{figures_path}/{output_filename}.png'")

if __name__ == "__main__":
    gerar_grafico_probabilidade_bloqueio()
