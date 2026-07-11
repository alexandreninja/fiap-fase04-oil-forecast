import pandas as pd
import numpy as np
from pathlib import Path
import json
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Caminho do projeto
CAMINHOS_DADOS = Path("data/processed/petroleo_brent_features.csv")
CAMINHOS_RELATORIOS = Path("reports")
CAMINHOS_RELATORIOS.mkdir(parents=True, exist_ok=True)

CAMINHOS_MODELOS = Path("models")
CAMINHOS_MODELOS.mkdir(parents=True, exist_ok=True)

CAMINHOS_PREVISOES = CAMINHOS_RELATORIOS / "previsoes_modelo.csv"
CAMINHOS_METRICAS_MODELO = CAMINHOS_RELATORIOS / "metricas_modelo.json"
CAMINHO_MODELO = CAMINHOS_MODELOS / "modelo_petroleo.joblib"

# 1. Carregar base com features
df = pd.read_csv(CAMINHOS_DADOS)

# 2. Converter data usando datetime
df['data'] = pd.to_datetime(df['data'])

# 3. Ordenar por data, por segurança
df = df.sort_values('data').reset_index(drop=True)

# 4. Definir features e alvo
features = [
    'lag_1', 'lag_7', 'lag_30',
    'media_7', 'media_30',
    'variacao_1', 'variacao_7',
    'ano', 'mes', 'dia_da_semana',
]

alvo = 'preco_usd'

X = df[features]
y = df[alvo]

# 5. Separar treino e teste respeitando a ordem temporal
# Vamos usar 80% inicial para treino e 20% final para teste
tamanho_treino = int(len(df) * 0.8)

X_treino = X.iloc[:tamanho_treino]
X_teste = X.iloc[tamanho_treino:]

y_treino = y.iloc[:tamanho_treino]
y_teste = y.iloc[tamanho_treino:]

datas_teste = df['data'].iloc[tamanho_treino:]


# 6. Criar e treinar o modelo
modelo = RandomForestRegressor(
    n_estimators=50,
    max_depth=12,
    random_state=42,
    n_jobs=1
)

modelo.fit(X_treino, y_treino)

# 7. Salvar modelo treinoado
joblib.dump(modelo, CAMINHO_MODELO)

# 8. Fazer previsões
previsoes = modelo.predict(X_teste)

# 9. Calcular métricas do modelo
mae = mean_absolute_error(y_teste, previsoes)
rmse = np.sqrt(mean_squared_error(y_teste, previsoes))
mape = np.mean(np.abs((y_teste - previsoes) / y_teste)) * 100

# 10. Calcular baseline no mesmo período de teste
# Baseline: preço previsto de hoje = preço de ontem
previsoes_baseline_teste = X_teste['lag_1']

mae_baseline_teste = mean_absolute_error(y_teste, previsoes_baseline_teste)
rmse_baseline_teste = np.sqrt(mean_squared_error(y_teste, previsoes_baseline_teste))
mape_baseline_teste = np.mean(
    np.abs((y_teste - previsoes_baseline_teste) / y_teste)
) * 100


# 11. Montar comparação com baseline no mesmo período de teste
comparacao_baseline = {
    'observacao': 'Baseline calculado no mesmo período de teste do modelo',
    'mae_baseline_teste': round(mae_baseline_teste, 4),
    'rmse_baseline_teste': round(rmse_baseline_teste, 4),
    'mape_baseline_teste': round(mape_baseline_teste, 4),
    'modelo_melhor_que_baseline_mae': bool(mae < mae_baseline_teste),
    'modelo_melhor_que_baseline_rmse': bool(rmse < rmse_baseline_teste),
    'modelo_melhor_que_baseline_mape': bool(mape < mape_baseline_teste),
}

# 12. Salvar métricas do modelo
metricas_modelo = {
    'modelo': 'RandomForestRegressor',
    'quantidade_linhas_total': int(len(df)),
    'quantidade_linhas_treino': int(len(X_treino)),
    'quantidade_linhas_teste': int(len(X_teste)),
    'primeira_data_treino': str(df['data'].iloc[:tamanho_treino].min().date()),
    'ultima_data_treino': str(df['data'].iloc[:tamanho_treino].max().date()),
    'primeira_data_teste': str(datas_teste.min().date()),
    'ultima_data_teste': str(datas_teste.max().date()),
    'feature_usadas': features,
    'mae': round(mae, 4),
    'rmse': round(rmse, 4),
    'mape': round(mape, 4),
    'comparacao_baseline': comparacao_baseline,
    }

with open(CAMINHOS_METRICAS_MODELO, 'w', encoding='utf-8') as arquivo:
    json.dump(metricas_modelo, arquivo, ensure_ascii=False, indent=4)

# 14. Salvar previsões
df_previsoes = pd.DataFrame({
    'data': datas_teste.values,
    'preco_real': y_teste.values,
    'preco_previsto': previsoes,
})

df_previsoes['erro'] = df_previsoes['preco_real'] - df_previsoes['preco_previsto']
df_previsoes['erro_absoluto'] = df_previsoes['erro'].abs()
df_previsoes['erro_percentual_absoluto'] = (df_previsoes['erro_absoluto'] / df_previsoes['preco_real']) * 100

df_previsoes.to_csv(CAMINHOS_PREVISOES, index=False)

# 13. Mostrar resultados no terminal
print('Modelo treinado com sucesso!')
print()
print('Período de treino:')
print(f'{df['data'].iloc[:tamanho_treino].min().date()} até {df['data'].iloc[:tamanho_treino].max().date()}')
print()
print('Período de teste:')
print(f'{datas_teste.min().date()} até {datas_teste.max().date()}')
print()
print('Métricas do modelo:')
print(f'MAE: {mae:.4f}')
print(f'RMSE: {rmse:.4f}')
print(f'MAPE: {mape:.4f}%')

print()
print('Comparação com baseline no mesmo período de teste:')
print(f'MAE baseline teste:  {mae_baseline_teste:.4f}')
print(f'RMSE baseline teste: {rmse_baseline_teste:.4f}')
print(f'MAPE baseline teste: {mape_baseline_teste:.4f}%')
print()
print(f'Melhor que baseline em MAE?  {mae < mae_baseline_teste}')
print(f'Melhor que baseline em RMSE? {rmse < rmse_baseline_teste}')
print(f'Melhor que baseline em MAPE? {mape < mape_baseline_teste}')

print()
print(f"Modelo salvo em:     {CAMINHO_MODELO}")
print(f"Previsões salvas em: {CAMINHOS_PREVISOES}")
print(f"Métricas salvas em:  {CAMINHOS_METRICAS_MODELO}")

print()
print('Últimas 10 previsões:')
print(df_previsoes.tail(10))


"""
Treinei um modelo Random Forest usando variáveis temporais como preços defasados, 
médias móveis, variações recentes e variáveis de calendário. A separação entre treino e 
teste respeitou a ordem temporal da série, usando os primeiros 80% dos dados para treino e 
os últimos 20% para teste.

Ao comparar o modelo com o baseline no mesmo período de teste, o baseline apresentou desempenho 
melhor. Isso indica que, para previsão diária, repetir o preço do dia anterior é uma referência 
forte e que o modelo inicial ainda precisa de melhorias para agregar valor preditivo.

O resultado reforça a importância de usar baseline em projetos de machine learning, 
depois nem sempre um modelo mais complexo supera uma regra simples.
"""