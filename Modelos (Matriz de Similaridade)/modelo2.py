import pandas as pd
import numpy as np

# --- Configurações do Modelo ---
# Caminho para o arquivo Excel da planilha.
caminho = 'planilha_editada.xlsx'

# Pesos para cada conjunto de atributos (alpha, beta, gamma)
ALPHA = 1  # Peso para Moradia e Mobilidade (MM)
BETA = 1   # Peso para Condição Familiar (CF)
GAMMA = 1  # Peso para Renda e Despesas (RD)

# Limite de similaridade (w%) para a construção das arestas do grafo.
# Arestas serão consideradas apenas se Rxy >= limite.
limite = 0.5

# Definição dos atributos para cada grupo
# Estes são os nomes das colunas exatos no seu arquivo Excel.
atrMM = [
    'Moradia do aluno',
    'Moradia do grupo familiar',
    'Meio de Transporte'
]

atrCF = [
    'Procedência escolar',
    'Número de filhos',
    'Indivíduos com doenças graves  na família',
    'Superior completo ou pós'
]

atrRD = [
    'Renda per capita (classes)',
    'Despesas per capita (classes)',
    'Valor total dos bens familiares (classes)'
]

# Número total de atributos em cada grupo
tot_MM = len(atrMM)
tot_CF = len(atrCF)
tot_RD = len(atrRD)

# --- Funções para o Modelo ---

def calcula_rxy(estX, estY):
    """
    Calcula o grau de relacionamento Rxy entre dois solicitantes.

    Args:
        estX (pd.Series): Linha de dados do primeiro solicitante.
        estY (pd.Series): Linha de dados do segundo solicitante.

    Returns:
        float: O valor Rxy, representando a similaridade entre os dois solicitantes.
    """

    # --- Cálculo de relacionamentos comuns para o grupo MM ---
    mmComuns = 0
    for attr in atrMM:
        if estX[attr] == estY[attr]:
            mmComuns += 1

    # Termo MM: (Soma MM_i) / (2 * tot_MM)
    termoMM = (mmComuns / tot_MM) if tot_MM > 0 else 0

    # --- Cálculo de relacionamentos comuns para o grupo CF ---
    cfComuns = 0
    for attr in atrCF:
        if estX[attr] == estY[attr]:
            cfComuns += 1

    # Termo CF: (Soma CF_i) / (2 * tot_CF)
    termoCF = (cfComuns / tot_CF) if tot_CF > 0 else 0

    # --- Cálculo de relacionamentos comuns para o grupo RD ---
    rdComuns = 0
    for attr in atrRD:
        if estX[attr] == estY[attr]:
            rdComuns += 1

    # Termo RD: (Soma RD_i) / (2 * tot_RD)
    termoRD = (rdComuns / tot_RD) if tot_RD > 0 else 0

    # --- Cálculo final de Rxy ---
    rxy = (ALPHA * termoMM) + (BETA * termoCF) + (GAMMA * termoRD)
    return rxy

# --- Carregamento e Pré-processamento dos Dados ---
df = pd.read_excel(caminho)
print(f"Dados carregados com sucesso de '{caminho}'.")
print(f"Número de solicitantes: {len(df)}")
print(f"Colunas do DataFrame: {df.columns.tolist()}")

# Garante que a coluna 'SOLICITANTE' seja o índice para fácil acesso
df = df.set_index('SOLICITANTE')

# --- Construção da Matriz de Similaridade (MS) ---
numEstudantes = len(df)
idsEstudantes = df.index.tolist()
matriz = np.zeros((numEstudantes, numEstudantes))

print("\nCalculando a matriz de similaridade (isso pode levar alguns minutos para 961 solicitantes)...")

arestas = 0

# Itera sobre todos os pares únicos de solicitantes (para evitar redundância e Rxy = Ryx)
for i in range(numEstudantes):
    for j in range(i, numEstudantes): # Começa de 'i' para preencher a diagonal e a parte superior
        estX = df.iloc[i]
        estY = df.iloc[j]

        rxy = calcula_rxy(estX, estY)
        if rxy > limite and i != j:
            arestas += 1
            matriz[i, j] = rxy
            matriz[j, i] = rxy # Garante que a matriz seja simétrica

        

# --- Análise Baseada na Matriz de Adjacência (similaridade_matrix) ---
print(f"\nConsiderando a matriz de similaridade como matriz de adjacência com limiar w% = {limite * 100:.1f}%...")

# Número de nós (solicitantes)
vertices = numEstudantes
print(f"Número de vértices: {vertices}")

# Calcula o número de arestas (conexões) com base no limiar
# Iteramos apenas a parte superior da matriz para contar cada aresta uma vez
print(f"Número de arestas: {arestas}")

# Densidade do "grafo" (proporção de arestas existentes em relação ao máximo possível)
max_possible_edges = (vertices * (vertices - 1)) / 2
density = arestas / max_possible_edges
print(f"Densidade da matriz de adjacência (considerando o limiar): {density:.4f}")

# Crie um DataFrame a partir da matriz numpy, usando os IDs dos solicitantes como índices e colunas
similarity_df = pd.DataFrame(matriz, index=idsEstudantes, columns=idsEstudantes)

# Salve o DataFrame em um arquivo CSV
# index=True e header=True garantem que os IDs dos solicitantes e os nomes das colunas sejam incluídos.
# O delimitador (sep) é definido como ';' para corresponder ao formato desejado.
similarity_df.to_csv('matriz.csv', index=True, header=True, sep=';')

print("Matriz de Similaridade salva com sucesso em 'matriz.csv'.")