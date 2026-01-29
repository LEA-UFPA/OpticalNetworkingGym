import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import re
import numpy as np

def plot_blocking_reasons():
    # 1. Localizar todos os arquivos de resultados
    result_files = glob.glob('results/load_results_*.csv')
    
    if not result_files:
        print("Nenhum arquivo de resultados encontrado na pasta 'results/'")
        return

    data_list = []

    # 2. Ler os dados de cada arquivo (cada carga)
    for file in result_files:
        try:
            # Extrair a carga (load) do nome do arquivo usando regex
            # Exemplo: load_results_nobel-eu_BandaC+L+S_900.csv -> 900
            match = re.search(r'_(\d+\.?\d*)\.csv$', file)
            if not match:
                continue
            
            load = float(match.group(1))
            
            # Ler o CSV (pulando a primeira linha de comentário se existir)
            df = pd.read_csv(file, comment='#')
            
            # Calcular as médias se houver mais de um episódio por arquivo
            # Vamos focar nos contadores de bloqueio
            # blocked_due_to_resources, blocked_due_to_osnr, blocked_due_to_ase, blocked_due_to_nli
            
            blocking_res = df['blocked_due_to_resources'].mean()
            blocking_osnr = df['blocked_due_to_osnr'].mean()
            
            # Tentativa de pegar bloqueios específicos se existirem nas colunas
            # bl_ase_dominant e bl_nli_dominant (que aparecem no header como blocked_due_to_ase/nli)
            blocking_ase = df['blocked_due_to_ase'].mean() if 'blocked_due_to_ase' in df.columns else 0
            blocking_nli = df['blocked_due_to_nli'].mean() if 'blocked_due_to_nli' in df.columns else 0
            
            total_processed = df['episode_service_blocking_rate'].mean() * 10000 # Aproximação se não tiver total_processed direto
            
            data_list.append({
                'load': load,
                'Recursos': blocking_res,
                'OSNR (Total)': blocking_osnr,
                'ASE (Ruído)': blocking_ase,
                'NLI (Não-Linear)': blocking_nli
            })
        except Exception as e:
            print(f"Erro ao processar arquivo {file}: {e}")

    if not data_list:
        print("Não foi possível extrair dados válidos dos arquivos.")
        return

    # 3. Criar DataFrame consolidado e ordenar por carga
    summary_df = pd.DataFrame(data_list).sort_values('load')

    # 4. Configurar o gráfico
    plt.figure(figsize=(12, 7))
    plt.style.use('bmh') # Estilo bonito

    # Plotar as linhas
    plt.plot(summary_df['load'], summary_df['Recursos'], marker='o', label='Falta de Recursos (Espectro)', linewidth=2, markersize=8)
    # Se OSNR for relevante, plotamos o total ou as quebras
    plt.plot(summary_df['load'], summary_df['ASE (Ruído)'], marker='s', label='Bloqueio por ASE (Ruído)', linestyle='--')
    plt.plot(summary_df['load'], summary_df['NLI (Não-Linear)'], marker='^', label='Bloqueio por NLI (Interferência)', linestyle='--')
    
    # Se houver valores no OSNR Total que não estão quebrados em ASE/NLI (caso de versões antigas)
    if summary_df['OSNR (Total)'].sum() > (summary_df['ASE (Ruído)'].sum() + summary_df['NLI (Não-Linear)'].sum()):
         plt.plot(summary_df['load'], summary_df['OSNR (Total)'], marker='x', label='OSNR Total', alpha=0.5)

    # Customização
    plt.title('Distribuição de Tipos de Erros/Bloqueios por Carga da Rede', fontsize=14, pad=20)
    plt.xlabel('Carga da Rede (Erlang)', fontsize=12)
    plt.ylabel('Número Médio de Bloqueios por Episódio', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    
    # Escala logarítmica no eixo Y se houver muita variação
    if summary_df[['Recursos', 'ASE (Ruído)', 'NLI (Não-Linear)']].max().max() > 100:
        plt.yscale('symlog', linthresh=10)
        plt.ylabel('Número Médio de Bloqueios (Escala SimLog)', fontsize=12)

    # Salvar o gráfico
    output_plot = 'blocking_reasons_plot.png'
    plt.tight_layout()
    plt.savefig(output_plot, dpi=300)
    
    print(f"\nGráfico gerado com sucesso: {output_plot}")
    print("\nResumo dos dados por carga:")
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    plot_blocking_reasons()
