# Correção da Heurística PSO (get_best_band_path_pso)

## Problema Identificado

A heurística PSO (índice 3) estava gerando **100% de bloqueio** mesmo com recursos disponíveis. A análise dos resultados mostrou:

- ✅ 0% de ocupação em todas as bandas
- ✅ OSNR adequado (~19.66 dB)
- ❌ Nenhum serviço sendo aceito
- ❌ ~1294 bloqueios por "recursos" por episódio

## Causa Raiz

O código assumia que **todas as bandas têm o mesmo número de slots** (`bands[0].num_slots`), mas as bandas configuradas têm tamanhos diferentes:

- **Banda C**: 344 slots
- **Banda L**: 406 slots  
- **Banda S**: 647 slots

### Locais Afetados

1. **`get_multiband_action_index()`** (linha 84):
   - Usava `env.bands[0].num_slots` para validação
   - Bloqueava slots válidos em bandas maiores que a primeira

2. **`shortest_available_path_first_fit_best_modulation_best_band()`** (linha 237):
   - Verificava `slot_in_band >= sim_env.bands[0].num_slots`
   - Rejeitava alocações válidas em Banda L e S

3. **`get_best_band_path_pso()` - allocate_with_first_fit** (linha 475):
   - Mesma verificação incorreta que #2

## Correções Aplicadas

### 1. Correção em `shortest_available_path_first_fit_best_modulation_best_band`

```python
# ANTES (linha 237)
if slot_in_band < 0 or slot_in_band >= sim_env.bands[0].num_slots:
    continue

# DEPOIS
if slot_in_band < 0 or slot_in_band >= band.num_slots:
    continue
```

### 2. Correção em `get_best_band_path_pso` - allocate_with_first_fit

```python
# ANTES (linha 475)
if slot_in_band < 0 or slot_in_band >= sim_env.bands[0].num_slots:
    continue

# DEPOIS  
if slot_in_band < 0 or slot_in_band >= band.num_slots:
    continue
```

### 3. Melhoria em `get_multiband_action_index`

Adicionada validação dupla para:
1. Verificar se o slot está dentro da banda alvo
2. Verificar se o slot está dentro do limite de decodificação do ambiente

```python
# Validar se o slot está dentro dos limites da banda alvo
target_band = env.bands[band_index]
if slot_in_band < 0 or slot_in_band >= target_band.num_slots:
    raise ValueError(...)

# CRÍTICO: Validar se o slot está dentro do limite de decodificação
# Se a banda alvo for maior que bands[0], alguns slots ficam inacessíveis
if slot_in_band >= slots_per_band:
    raise ValueError(
        f"slot_in_band {slot_in_band} excede o limite de decodificação {slots_per_band} "
        f"(tamanho da primeira banda). Banda {target_band.name} tem {target_band.num_slots} slots, "
        f"mas apenas os primeiros {slots_per_band} são acessíveis."
    )
```

## Limitação Conhecida

⚠️ **IMPORTANTE**: Devido à forma como o ambiente decodifica ações (usando `bands[0].num_slots`), **bandas maiores que a primeira têm slots inacessíveis**.

No seu caso:
- Banda C: 344 slots ✅ (todos acessíveis)
- Banda L: 406 slots ⚠️ (apenas primeiros 344 acessíveis)
- Banda S: 647 slots ⚠️ (apenas primeiros 344 acessíveis)

### Solução Futura

Para usar completamente todas as bandas, seria necessário:
1. Modificar `qrmsa.pyx` linha 1378 para usar `total_slots` ou `band.num_slots`
2. Atualizar o `action_space` para refletir isso
3. Recompilar o código Cython

## Resultados Após Correção

Teste com carga 700-900 Erlangs (100 requisições):
- ✅ 99% de aceitação (apenas 1% de bloqueio)
- ✅ Ocupação ~5% na Banda L
- ✅ Modulações sendo usadas corretamente:
  - 64QAM: 55-57 serviços
  - 32QAM: 36-40 serviços
  - 16QAM: 2-7 serviços
- ✅ 0 bloqueios por OSNR ou recursos

## Arquivos Modificados

- `/home/eclipse/Documentos/Artigo/ONG/OpticalNetworkingGym/optical_networking_gym/heuristics/heuristics.py`
  - Linhas modificadas: 84-101 (get_multiband_action_index)
  - Linhas modificadas: 237 (shortest_available_path_first_fit_best_modulation_best_band)
  - Linhas modificadas: 475 (get_best_band_path_pso)
