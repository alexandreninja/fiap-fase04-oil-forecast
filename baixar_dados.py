import requests
import pandas as pd

url = "https://www.ipeadata.gov.br/api/odata4/Metadados('EIA366_PBRENT366')/Valores"

resposta = requests.get(url)

dados_json = resposta.json()

dados = dados_json["value"]

tabela = pd.DataFrame(dados)

tabela.to_csv("data/raw/petroleo_brent_bruto.csv", index=False, encoding="utf-8-sig")

print("Arquivo salvo com sucesso!")
print("Quantidade de linhas:", len(tabela))
print("Colunas:", tabela.columns.tolist())
print(tabela.head())


"""
Eu baixei os dados originais do IPEADATA. 
Agora vou criar uma versão limpa, deixando só as colunas que preciso para trabalhar: 
data e preço do petróleo em dólar.”
"""