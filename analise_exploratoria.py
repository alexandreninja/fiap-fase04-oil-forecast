import pandas as pd
import matplotlib.pyplot as plt

# 1. Caminho do arquivo tratado
arquivo_tratado = "data/processed/petroleo_brent_tratado.csv"

# 2. Ler a base tratada
tabela = pd.read_csv(arquivo_tratado)

print('Base tratada carregada com sucesso!')
print('Colunas:', tabela.columns.tolist())
print('Quantidade de linhas:', len(tabela))

# 3. Converter a coluna de data para o formato de data
tabela['data'] = pd.to_datetime(tabela['data'])

# 4. Verificar as informações principais da base
primeira_data = tabela['data'].min()
ultima_data = tabela['data'].max()

menor_preco = tabela['preco_usd'].min()
maior_preco = tabela['preco_usd'].max()

preco_medio = tabela['preco_usd'].mean()
preco_mediano = tabela['preco_usd'].median()

valores_vazios = tabela.isnull().sum()

print('\nPeríodo da série:')
print('Primeira data:', primeira_data)
print('Última data:', ultima_data)

print('\nResumo dos preços:')
print('Menor preço:', round(menor_preco, 2))
print('Maior preço:', round(maior_preco, 2))
print('Preço médio:', round(preco_medio, 2))
print('Preço mediano:', round(preco_mediano, 2))

print('\nValores vazios por coluna:')
print(valores_vazios)

print('\nPrimeiras linhas da base:')
print(tabela.head())

print('\nÚltimas linhas da base:')
print(tabela.tail())

# 5. Criar um gráfico simples do histórico
plt.figure(figsize=(12, 6))
plt.plot(tabela['data'], tabela['preco_usd'])
plt.title('Histórico do preço do petróleo Brent em dólar (US$)')
plt.xlabel('Data')
plt.ylabel('Preço (US$)')
plt.grid(True)


# 6. Salvar o gráfico em um arquivo
caminho_grafico = "reports/grafico_historico_petroleo_brent.png"
plt.savefig(caminho_grafico, bbox_inches='tight', dpi=300)

print('\nGráfico salvo em:', caminho_grafico)

"""
Após limpar a base, fiz uma análise exploratória simples. 
Verifiquei o período disponível, a quantidade de registros, os preços mínimo, 
máximo, médio e mediano, confirmei que não havia valores vazios e 
gerei um gráfico histórico para visualizar o comportamento do preço 
do petróleo Brent ao longo do tempo.

O gráfico mostra que a série tem bastante variação, 
com picos e quedas importantes. Por isso, antes de criar um modelo mais elaborado, 
faz sentido construir uma previsão simples de referência.
"""