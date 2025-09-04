#!/usr/bin/env python3
"""
Script para monitorar o progresso dos testes de QoT constraints.
"""

import os
import glob
from pathlib import Path

def check_progress():
    """Verifica o progresso dos arquivos de resultado."""
    results_dir = Path("results")
    
    if not results_dir.exists():
        print("Diretório 'results' não encontrado.")
        return
    
    # Padrão dos arquivos esperados
    pattern = "load_episodes_1_*_def_*_0_nobel-eu_1.0_*.0_nw_cnr_nobel-eu.csv"
    files = list(results_dir.glob(pattern))
    
    print(f"=== Progresso dos Testes de QoT Constraints ===")
    print(f"Arquivos encontrados: {len(files)}")
    print()
    
    # Configurações esperadas
    qot_constraints = ["asenli", "ase", "dist"]  # nomes dos arquivos
    loads = [100, 200, 300]
    defrag_options = [False, True]
    
    total_expected = len(qot_constraints) * len(loads) * len(defrag_options)
    print(f"Total esperado: {total_expected} arquivos")
    print()
    
    # Verificar quais arquivos existem
    for qot in qot_constraints:
        for load in loads:
            for defrag in defrag_options:
                filename = f"load_episodes_1_{qot}_def_{defrag}_0_nobel-eu_1.0_{load}.0_nw_cnr_nobel-eu.csv"
                filepath = results_dir / filename
                
                status = "✓" if filepath.exists() else "✗"
                size = ""
                lines = ""
                
                if filepath.exists():
                    try:
                        stat = filepath.stat()
                        size = f" ({stat.st_size} bytes)"
                        
                        # Contar linhas (episódios)
                        with open(filepath, 'r') as f:
                            line_count = sum(1 for line in f) - 2  # -2 para remover header
                        lines = f" - {line_count}/50 episódios"
                    except Exception as e:
                        size = f" (erro: {e})"
                
                print(f"{status} {qot.upper():8s} | Load={load:3d} | Defrag={str(defrag):5s}{size}{lines}")
    
    print()
    print(f"Progresso: {len(files)}/{total_expected} arquivos ({len(files)/total_expected*100:.1f}%)")

if __name__ == "__main__":
    check_progress()
