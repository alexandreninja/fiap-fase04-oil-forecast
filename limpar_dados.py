import pandas as pd

# 1. Caminhos dos arquivos
arquivo_bruto = 'data/raw/petroleo_brent_bruto.csv'
arquivo_tratado = 'data/processed/petroleo_brent_tratado.csv'

# 2. Ler o CSV bruto
tabela = pd.read_csv(arquivo_bruto)

print('Arquivo bruto carregado com sucesso!')
print('Linhas no arquivo bruto:', len(tabela))
print('Colunos no arquivo bruto:', tabela.columns.tolist())

# 3. Manter somente as colunas que interessam
tabela = tabela[['VALDATA', 'VALVALOR']]

# 4. Renomear as colunas para nomes mais fáceis de trabalhar
tabela = tabela.rename(
    columns={
        'VALDATA': 'data',
        'VALVALOR': 'preco_usd'
    }
)

# 5. Converter data e preço para formatos corretos
tabela['data'] = pd.to_datetime(tabela['data'], errors='coerce', utc=True).dt.date
tabela['preco_usd'] = pd.to_numeric(tabela['preco_usd'], errors='coerce')


# 6. Remover linhas sem data ou sem preço
linhas_antes = len(tabela)
tabela = tabela.dropna(subset=['data', 'preco_usd'])

linhas_depois = len(tabela)
linhas_removidas = linhas_antes - linhas_depois

# 7. Ordenar pela data
tabela = tabela.sort_values('data').reset_index(drop=True)

# 8. Salvar o CSV tratado
tabela.to_csv(arquivo_tratado, index=False, encoding='utf-8-sig')

print('\nArquivo tratado salvo com sucesso!')
print('Arquivo:', arquivo_tratado)
print('Linhas antes da limpeza:', linhas_antes)
print('Linhas depois da limpeza:', linhas_depois)
print('Linhas removidas:', linhas_removidas)

print('\nPrimeiras linhas:')
print(tabela.head())

print('\nÚltimas linhas:')
print(tabela.tail())

print('\nResumo do preço:')
print(tabela['preco_usd'].describe())


"""
Nesta etapa, eu li o CSV bruto baixado do IPEADATA, 
mantive apenas as colunas de data e valor, 
renomeei essas colunas para facilitar o entendimento, 
removi registros sem preço e salvei uma nova base tratada para usar nas análises e no modelo.
"""

"""
“Depois de baixar o CSV bruto da API do IPEADATA, 
eu fiz uma etapa de limpeza. 
Mantive apenas as colunas de data e valor, 
renomeei para data e preco_usd, removi registros sem preço e 
salvei uma nova base tratada para usar na análise e no modelo.”
"""