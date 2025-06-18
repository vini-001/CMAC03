# Importa as bibliotecas necessárias
import pandas as pd

# Passo 1: Carregar os dados (simulados ou lidos do arquivo Excel)

# Tentativa de leitura do arquivo real
dados = pd.read_excel("planilha_editada.xlsx")
print("Arquivo carregado com sucesso.")

atributos_utilizados = [
    "Renda per capita (classes)",
    "Despesas per capita (classes)",
    "Valor total dos bens familiares (classes)",
    "Número de filhos",
    "Indivíduos com doenças graves  na família",
    "Superior completo ou pós",
    "Procedência escolar",
    "Moradia do grupo familiar",
    "Moradia do aluno",
    "Meio de Transporte"
]
# Extraímos os dados dos atributos em uma lista de listas.
# Cada linha representa um solicitante, e cada coluna representa um atributo.
# Usamos o método tolist() para converter o DataFrame pandas em uma lista de listas do Python.
# Isso facilita a comparação direta entre os valores dos atributos.
lista_de_vetores = dados[atributos_utilizados].values.tolist()

# Descobrimos quantos solicitantes temos
quantidade_de_solicitantes = len(lista_de_vetores)

# Inicializamos a matriz de similaridade com zeros.
# A matriz será quadrada (m x m), onde m é o número de solicitantes.
# Não abreviamos: usamos nomes descritivos.
matriz_de_similaridade = []

for linha in range(quantidade_de_solicitantes):
    nova_linha = []
    for coluna in range(quantidade_de_solicitantes):
        nova_linha.append(0)
    matriz_de_similaridade.append(nova_linha)

limite_percentual = 0.5 
total_de_atributos = len(atributos_utilizados)
arestas = 0
# Preenchemos a matriz de similaridade comparando os atributos de cada par de alunos
for i in range(quantidade_de_solicitantes):
    for j in range(i + 1, quantidade_de_solicitantes):
        contador_de_atributos_iguais = 0
        for k in range(len(atributos_utilizados)):
            if lista_de_vetores[i][k] == lista_de_vetores[j][k]:
                contador_de_atributos_iguais += 1
        # Armazenamos o grau de similaridade em posições simétricas da matriz (grafo não direcionado)
        if contador_de_atributos_iguais > limite_percentual * total_de_atributos:
            matriz_de_similaridade[i][j] = contador_de_atributos_iguais
            matriz_de_similaridade[j][i] = contador_de_atributos_iguais
            arestas += 1


# Cada aluno será representado pelo seu identificador (id_discente)
lista_de_ids = dados['SOLICITANTE'].tolist()

print(f"Total de arestas no grafo: {arestas}")


# Exporta a matriz de similaridade para um arquivo CSV legível
nome_do_arquivo = "matriz_similaridade_model1.csv"

# Usamos o pandas para facilitar a criação da tabela com cabeçalhos (ids)
df_matriz = pd.DataFrame(matriz_de_similaridade, index=lista_de_ids, columns=lista_de_ids)

# Salvamos o DataFrame como CSV
df_matriz.to_csv(nome_do_arquivo, sep=";", encoding="utf-8")

print(f"Matriz de similaridade salva como '{nome_do_arquivo}' com sucesso.")