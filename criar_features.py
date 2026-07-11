import pandas as pd
from pathlib import Path
import json

# Caminhos do Projeto
CAMINHO_DADOS = Path("data/processed/petroleo_brent_tratado.csv")
CAMINHO_SAIDA = Path("data/processed/petroleo_brent_features.csv")
CAMINHOS_RELATORIOS = Path("reports")
CAMINHOS_RELATORIOS.mkdir(parents=True, exist_ok=True)

CAMINHO_RESUMO = CAMINHOS_RELATORIOS / "features_resumo.json"

# 1. Carregar a base tratada
df = pd.read_csv(CAMINHO_DADOS)

# 2. Converter a coluna de data para o tipo datetime
df['data'] = pd.to_datetime(df['data'])

# 3. Ordenar por data, por segurança
df = df.sort_values('data').reset_index(drop=True)

# 4. Criar features de defasagem
# Essas colunas usam preços passados para prever o preço atual
df['lag_1'] = df['preco_usd'].shift(1)
df['lag_7'] = df['preco_usd'].shift(7)
df['lag_30'] = df['preco_usd'].shift(30)

# 5. Criar médias móveis usando apenas dados anteriores
# O shift(1) evita usar o preço do próprio dia na média
df['media_7'] = df['preco_usd'].shift(1).rolling(window=7).mean()
df['media_30'] = df['preco_usd'].shift(1).rolling(window=30).mean()

# 6. Criar variações recentes
df['variacao_1'] = df['preco_usd'].shift(1) - df['preco_usd'].shift(2)
df['variacao_7'] = df['preco_usd'].shift(1) - df['preco_usd'].shift(7)

# 7. Criar features de calendário
df['ano'] = df['data'].dt.year
df['mes'] = df['data'].dt.month
df['dia_da_semana'] = df['data'].dt.dayofweek  # Segunda-feira=0, Domingo=6

# 8. Remover linhas com valores vazios criados pelas lags e médias
df_features = df.dropna().copy()


# 9. Salvar a base com as features
df_features.to_csv(CAMINHO_SAIDA, index=False)

# 10. Criar resumo para documentação
resumo = {
    'arquivo_entrada': str(CAMINHO_DADOS),
    'arquivo_saida': str(CAMINHO_SAIDA),
    'linhas_base_original': int(len(df)),
    'linhas_base_com_features': int(len(df_features)),
    'linhas_removidas_por_lags_e_medias': int(len(df)) - int(len(df_features)),
    'primeira_data': str(df['data'].min().date()),
    'ultima_data': str(df['data'].max().date()),
    'features_criadas': [
        'lag_1', 'lag_7', 'lag_30',
        'media_7', 'media_30',
        'variacao_1', 'variacao_7',
        'ano', 'mes', 'dia_da_semana',
    ]
}

with open(CAMINHO_RESUMO, 'w', encoding='utf-8') as arquivo:
    json.dump(resumo, arquivo, ensure_ascii=False, indent=4)

# 11. Mostrar resultado no terminal
print('Features temporais criadas com sucesso!')
print()
print(f'Arquivo de entrada: {CAMINHO_DADOS}')
print(f'Arquivo de saída: {CAMINHO_SAIDA}')
print(f'Resumo salvo em: {CAMINHO_RESUMO}')
print()
print(f'Linhas na base original: {len(df)}')
print(f'Linhas na base com features: {len(df_features)}')
print(f'Linhas removidas: {len(df) - len(df_features)}')
print()
print('Colunas da base com features:')
print(df_features.columns.tolist())
print()
print('Primeiras 5 linhas:')
print(df_features.head(5))
print('Últimas 5 linhas:')
print(df_features.tail(5))


"""
Nesta etapa, criei features temporais a partir do histórico do preço do Brent. 
Foram criadas variáveis com preços defasados, médias móveis, variações recentes e 
informações de calendário, como ano, mês e dia da semana.

Essas variáveis permitem que o modelo use informações do passado para 
tentar prever o preço atual, sem acessar dados futuros.

As primeiras 30 linhas foram removidas porque não havia histórico suficiente 
para calcular as variáveis de 30 períodos. As médias móveis foram calculadas 
com deslocamento para evitar uso de informação do próprio dia na previsão.
"""