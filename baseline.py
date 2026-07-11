import pandas as pd
import numpy as np
from pathlib import Path
import json

# Caminhos do Projeto
CAMINHO_DADOS = Path("data/processed/petroleo_brent_tratado.csv")
CAMINHO_RELATORIOS = Path("reports")

CAMINHO_RELATORIOS.mkdir(parents=True, exist_ok=True)

CAMINHO_COMPARACAO = CAMINHO_RELATORIOS / "baseline_comparacao.csv"
CAMINHO_METRICAS = CAMINHO_RELATORIOS / "baseline_metricas.json"

# 1. Carregar a base tratada
df = pd.read_csv(CAMINHO_DADOS)

# 2. Converter a coluna de data para o tipo datetime
df['data'] = pd.to_datetime(df['data'])

# 3. Ordenar por data, por segurança
df = df.sort_values('data').reset_index(drop=True)

# 4. Criar a previsão baseline
# A previsão de hoje será igual ao preço de ontem
df['previsao_baseline'] = df['preco_usd'].shift(1)

# 5. Remover a primeira linha
# Ela não tem preço anterior para servir como previsão
df_baseline = df.dropna(subset=['previsao_baseline']).copy()

# 6. Calcular os erros
df_baseline['erro'] = df_baseline['preco_usd'] - df_baseline['previsao_baseline']
df_baseline['erro_absoluto'] = df_baseline['erro'].abs()
df_baseline['erro_percentual_absoluto'] = (df_baseline['erro_absoluto'] / df_baseline['preco_usd']) * 100

# 7. Calcular métricas do baseline
mae = df_baseline['erro_absoluto'].mean()
rmse = np.sqrt((df_baseline['erro'] ** 2).mean())
mape = df_baseline['erro_percentual_absoluto'].mean()

metricas = {
    'modelo': 'Baseline - preço previsto de hoje igual ao preço de ontem',
    'mae': round(mae, 4),
    'rmse': round(rmse, 4),
    'mape': round(mape, 4),
    'quantidade_linhas_avaliadas': int(len(df_baseline)),
    'primeira_linha_avaliada': str(df_baseline['data'].min().date()),
    'ultima_linha_avaliada': str(df_baseline['data'].max().date())
}

# 8. Salvar comparação entre o valor real e o valor previsto
df_baseline.to_csv(CAMINHO_COMPARACAO, index=False)

# 9. Salvar métrica em um arquivo JSON
with open(CAMINHO_METRICAS, 'w', encoding='utf-8') as arquivo:
    json.dump(metricas, arquivo, ensure_ascii=False, indent=4)

# 10. Mostrar resultado no terminal
print('Baseline criado com sucesso!')
print()
print('Métricas do Baseline:')
print(f'MAE: {mae:.4f}')
print(f'RMSE: {rmse:.4f}')
print(f'MAPE: {mape:.4f}%')
print()
print(f'Arquivo de comparação salvo em: {CAMINHO_COMPARACAO}')
print(f'Arquivo de métricas salvo em: {CAMINHO_METRICAS}')

print()
print('Últimas 10 linhas de comparação:')
print(df_baseline[['data', 'preco_usd', 'previsao_baseline', 'erro_absoluto']].tail(10))

"""
Criei uma previsão baseline usando a estratégia de repetir o preço do dia anterior 
como previsão para o dia atual. Essa abordagem simples serve como referência mínima 
para comparar modelos mais avançados.

O baseline obteve MAE de 0.9375, RMSE de 1.5177 e MAPE de 1.7901%. 
Isso significa que, em média, a previsão ingênua errou cerca de US$ 0.94 por dia, 
ou aproximadamente 1.79% em termos percentuais.

O modelo preditivo final precisa ser comparado com esse baseline para verificar 
se realmente entrega ganho em relação a uma previsão simples.
"""