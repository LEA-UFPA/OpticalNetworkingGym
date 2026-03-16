import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import re
import numpy as np

def plot_margin_results(results_path, load_to_plot):
    # Procura os ficheiros gerados pelo seu graph_margin.py
    pattern = os.path.join(results_path, f"m_episodes_m_*_load_{load_to_plot}.csv")
    files = glob.glob(pattern)
    
    if not files:
        print(f"Erro: Nenhum ficheiro encontrado para a carga {load_to_plot}")
        return

    data = []
    for f in files:
        match = re.search(r"m_([\d.]+)_load", f)
        if match:
            margin = float(match.group(1))
            df = pd.read_csv(f, comment='#')
            # Extrai a média da taxa de bloqueio
            avg_blocking = df['episode_service_blocking_rate'].mean()
            data.append({'margin': margin, 'blocking': avg_blocking})

    plot_df = pd.DataFrame(data).sort_values(by='margin')

    # --- CONFIGURAÇÃO ESTÉTICA (SEMELHANTE À IMAGEM) ---
    plt.figure(figsize=(7, 6)) # Formato mais quadrado como na imagem
    
    # Marcadores: 'o' (círculo), cor azul, com borda preta
    plt.plot(plot_df['margin'], plot_df['blocking'], 
             marker='o', markersize=8, markerfacecolor='#2554C7', markeredgecolor='black',
             linestyle='-', color='#2554C7', label=f'Carga {load_to_plot} Erlang')
    
    # Escala Logarítmica e Limites do Eixo Y
    plt.yscale('log')
    plt.ylim(1e-5, 1) # Range de 10^-5 a 10^0 como na imagem
    
    # Grelha (Grid) pontilhada para major e minor ticks
    plt.grid(True, which="both", ls="--", color='gray', alpha=0.5)
    
    # Nomes dos Eixos
    plt.xlabel('Margem de $OSNR$ [$dB$]', fontsize=12) # Mantendo Margem como variável
    plt.ylabel('Blocking probability', fontsize=12) # Nome igual à imagem
    
    # Legenda no canto inferior direito
    plt.legend(loc='lower right', frameon=True)
    
    # Ajuste dos ticks do Eixo X
    plt.xticks(np.arange(0, 3.1, 0.5)) 
    
    plt.tight_layout()
    # -------------------------------------------------

    output_image = os.path.join("examples/SbrT_2026", f"grafico_SbrT_style_load_{load_to_plot}.png")
    plt.savefig(output_image, dpi=300) # Alta resolução para o artigo
    print(f"\nSucesso! Gráfico estilo SbrT gerado em: {output_image}")

if __name__ == "__main__":
    path_resultados = "examples/SbrT_2026/results/margin_study/"
    plot_margin_results(path_resultados, 200.0)