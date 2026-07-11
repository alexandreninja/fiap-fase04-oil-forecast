from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st
from pandas.tseries.offsets import BDay


# =========================
# Configuração da página
# =========================

st.set_page_config(
    page_title="Previsão Brent — FIAP Fase 04",
    page_icon="🛢️",
    layout="wide",
)


# =========================
# Caminhos do projeto
# =========================

ROOT = Path(__file__).parent

DATA_PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"
MODELS = ROOT / "models"

CAMINHO_DADOS_TRATADOS = DATA_PROCESSED / "petroleo_brent_tratado.csv"
CAMINHO_DADOS_FEATURES = DATA_PROCESSED / "petroleo_brent_features.csv"
CAMINHO_PREVISOES = REPORTS / "previsoes_modelo.csv"
CAMINHO_METRICAS_MODELO = REPORTS / "metricas_modelo.json"
CAMINHO_MODELO = MODELS / "modelo_petroleo.joblib"


# =========================
# Funções auxiliares
# =========================

@st.cache_data
def carregar_dados():
    df_tratado = pd.read_csv(CAMINHO_DADOS_TRATADOS, parse_dates=["data"])
    df_features = pd.read_csv(CAMINHO_DADOS_FEATURES, parse_dates=["data"])
    df_previsoes = pd.read_csv(CAMINHO_PREVISOES, parse_dates=["data"])

    with open(CAMINHO_METRICAS_MODELO, "r", encoding="utf-8") as arquivo:
        metricas_modelo = json.load(arquivo)

    return df_tratado, df_features, df_previsoes, metricas_modelo


@st.cache_resource
def carregar_modelo():
    return joblib.load(CAMINHO_MODELO)


def criar_features_proximo_dia(df_tratado):
    df = df_tratado.sort_values("data").reset_index(drop=True).copy()

    ultima_data = df["data"].iloc[-1]
    proxima_data = ultima_data + BDay(1)

    ultimo_preco = df["preco_usd"].iloc[-1]
    preco_anterior = df["preco_usd"].iloc[-2]
    preco_7_periodos_atras = df["preco_usd"].iloc[-7]
    preco_30_periodos_atras = df["preco_usd"].iloc[-30]

    media_7 = df["preco_usd"].tail(7).mean()
    media_30 = df["preco_usd"].tail(30).mean()

    variacao_1 = ultimo_preco - preco_anterior
    variacao_7 = ultimo_preco - preco_7_periodos_atras

    features = pd.DataFrame([{
        "lag_1": ultimo_preco,
        "lag_7": preco_7_periodos_atras,
        "lag_30": preco_30_periodos_atras,
        "media_7": media_7,
        "media_30": media_30,
        "variacao_1": variacao_1,
        "variacao_7": variacao_7,
        "ano": proxima_data.year,
        "mes": proxima_data.month,
        "dia_da_semana": proxima_data.dayofweek,
    }])

    return proxima_data, ultimo_preco, features


# =========================
# Carregamento
# =========================

df_tratado, df_features, df_previsoes, metricas_modelo = carregar_dados()
modelo = carregar_modelo()


# =========================
# Título
# =========================

st.title("🛢️ Previsão do Preço do Petróleo Brent")
st.subheader("FIAP — Tech Challenge — Fase 04")

st.markdown(
    """
    Este aplicativo apresenta um MVP de análise e previsão do preço do petróleo Brent.

    O projeto utiliza dados históricos, features temporais e um modelo de machine learning
    para estimar o preço do Brent. O desempenho do modelo é comparado com um baseline simples:
    
    **preço previsto de hoje = preço de ontem**.
    """
)


# =========================
# Visão geral
# =========================

st.header("1. Visão geral dos dados")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Quantidade de registros", f"{len(df_tratado):,}".replace(",", "."))
col2.metric("Data inicial", df_tratado["data"].min().strftime("%d/%m/%Y"))
col3.metric("Data final", df_tratado["data"].max().strftime("%d/%m/%Y"))
col4.metric("Último preço", f"US$ {df_tratado['preco_usd'].iloc[-1]:.2f}")

st.line_chart(
    data=df_tratado,
    x="data",
    y="preco_usd",
    height=400,
)


# =========================
# Métricas do modelo
# =========================

st.header("2. Desempenho do modelo")

comparacao = metricas_modelo["comparacao_baseline"]

col1, col2, col3 = st.columns(3)

col1.metric("MAE do modelo", metricas_modelo["mae"])
col2.metric("RMSE do modelo", metricas_modelo["rmse"])
col3.metric("MAPE do modelo", f"{metricas_modelo['mape']}%")

st.markdown("### Comparação com baseline no mesmo período de teste")

df_comparacao = pd.DataFrame({
    "Métrica": ["MAE", "RMSE", "MAPE (%)"],
    "Modelo Random Forest": [
        metricas_modelo["mae"],
        metricas_modelo["rmse"],
        metricas_modelo["mape"],
    ],
    "Baseline": [
        comparacao["mae_baseline_teste"],
        comparacao["rmse_baseline_teste"],
        comparacao["mape_baseline_teste"],
    ],
    "Modelo melhor que baseline?": [
        "✅ Sim" if comparacao["modelo_melhor_que_baseline_mae"] else "❌ Não",
        "✅ Sim" if comparacao["modelo_melhor_que_baseline_rmse"] else "❌ Não",
        "✅ Sim" if comparacao["modelo_melhor_que_baseline_mape"] else "❌ Não",
    ],
})

st.dataframe(df_comparacao, use_container_width=True)

st.warning(
    """
    O modelo Random Forest não superou o baseline no período de teste.
    Isso mostra que, para previsão diária do Brent, repetir o preço do dia anterior
    é uma referência forte.
    """
)


# =========================
# Previsões no período de teste
# =========================

st.header("3. Previsões no período de teste")

st.markdown(
    """
    O gráfico abaixo compara o preço real com o preço previsto pelo modelo
    no final do período de teste.
    """
)

amostra = df_previsoes.tail(300).copy()

df_grafico_previsoes = amostra[["data", "preco_real", "preco_previsto"]].set_index("data")

st.line_chart(df_grafico_previsoes, height=400)

st.markdown("### Últimas previsões registradas")

df_ultimas_previsoes = df_previsoes.tail(10).copy()

df_ultimas_previsoes["data"] = df_ultimas_previsoes["data"].dt.strftime("%d/%m/%Y")

df_ultimas_previsoes = df_ultimas_previsoes.rename(columns={
    "data": "Data",
    "preco_real": "Preço real",
    "preco_previsto": "Preço previsto",
    "erro": "Erro",
    "erro_absoluto": "Erro absoluto",
    "erro_percentual_absoluto": "Erro percentual absoluto (%)",
})

st.dataframe(
    df_ultimas_previsoes,
    use_container_width=True,
)


# =========================
# Previsão para próximo dia útil
# =========================

st.header("4. Previsão para o próximo dia útil")

proxima_data, ultimo_preco, features_proximo_dia = criar_features_proximo_dia(df_tratado)

previsao_modelo = modelo.predict(features_proximo_dia)[0]
previsao_baseline = ultimo_preco

col1, col2, col3 = st.columns(3)

col1.metric("Próximo dia útil estimado", proxima_data.strftime("%d/%m/%Y"))
col2.metric("Previsão do modelo", f"US$ {previsao_modelo:.2f}")
col3.metric("Baseline", f"US$ {previsao_baseline:.2f}")

st.markdown(
    """
    **Interpretação:**  
    A previsão do modelo é calculada a partir das features temporais mais recentes.
    Já o baseline assume que o próximo preço será igual ao último preço conhecido.
    """
)


# =========================
# Conclusão
# =========================

st.header("5. Conclusão")

st.markdown(
    """
    O MVP foi construído com sucesso, contemplando coleta, tratamento, análise,
    criação de features, treinamento de modelo, salvamento do modelo e disponibilização
    em uma aplicação Streamlit.

    Apesar de o modelo Random Forest ter sido treinado corretamente, o baseline apresentou
    desempenho melhor no período de teste. Esse resultado é relevante porque mostra que
    modelos mais complexos nem sempre superam regras simples, especialmente em séries temporais
    diárias com forte dependência do valor mais recente.
    """
)