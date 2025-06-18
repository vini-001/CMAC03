from fpdf import FPDF
import pandas as pd
import networkx as nx
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.neighbors import kneighbors_graph
import numpy as np
from collections import Counter

# Carregar os dados
df = pd.read_excel('Solicitantes de Auxílio Estudantil - 2018.xlsx', sheet_name='2018')

# Atributos utilizados
features = [
    'Qual sua PROCEDÊNCIA ESCOLAR?',
    'Qual a situação da MORADIA DO ALUNO?',
    'Qual a situação da MORADIA DO GRUPO FAMILIAR?',
    'Quantos filhos o solicitante possui?',
    'Renda per capita',
    'Despesas per capita',
    'Quantidade de individuos com doença grave no grupo familiar',
    'Familiares com Superior Completo ou Pós',
    'Valor Total dos bens familiares'
]

# Pesos dos atributos (0 a 10) – ajuste conforme desejado
pesos = {
    'Qual sua PROCEDÊNCIA ESCOLAR?': 4,
    'Qual a situação da MORADIA DO ALUNO?': 9,
    'Qual a situação da MORADIA DO GRUPO FAMILIAR?': 7,
    'Quantos filhos o solicitante possui?': 5,
    'Renda per capita': 10,
    'Despesas per capita': 8,
    'Quantidade de individuos com doença grave no grupo familiar': 7,
    'Familiares com Superior Completo ou Pós': 3,
    'Valor Total dos bens familiares': 6
}

#
# Codificar variáveis categóricas
encoders = {}
for col in features:
    if df[col].dtype == 'object':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

 # Normalizar os dados originais (sem peso)
scaler = MinMaxScaler()
df_normalized = df.copy()
df_normalized[features] = scaler.fit_transform(df[features])

# Inverter atributos específicos após normalização
atributos_inverter = [
    'Quantos filhos o solicitante possui?',
    'Despesas per capita',
    'Quantidade de individuos com doença grave no grupo familiar'
]

df_normalized_inv = df_normalized.copy()
for col in atributos_inverter:
    if col in df_normalized_inv.columns:
        df_normalized_inv[col] = 1.0 - df_normalized_inv[col]

# Ajustar pesos para escala 0 a 1 dividindo por 10
pesos_array = np.array([pesos[f] for f in features]) / 10.0

# Aplicar pesos ajustados após a normalização (max peso vira 1) usando df_normalized_inv
df_weighted = df_normalized_inv.copy()
df_weighted[features] = df_normalized_inv[features] * pesos_array

# Criar matriz de similaridade usando k-vizinhos mais próximos com dados ponderados normalizados
X_ponderado_normalizado = df_weighted[features].values
adj_matrix = kneighbors_graph(X_ponderado_normalizado, n_neighbors=75, mode='connectivity', include_self=False)
adj_matrix = adj_matrix + adj_matrix.T  # Tornar a matriz simétrica
adj_matrix.data = np.ones_like(adj_matrix.data)  # Tornar binária

# Criar grafo
G = nx.from_scipy_sparse_array(adj_matrix)

# Adicionar IDs como atributos
id_map = {idx: id_val for idx, id_val in enumerate(df['id_discente'])}
nx.set_node_attributes(G, id_map, 'id_discente')

# Mapear índice para renda per capita original (não normalizada)
renda_original = df['Renda per capita'].to_dict()

# Mapeamentos de classe
classe_labels = {
    1: 'E', 2: 'D', 3: 'C2', 4: 'C1', 5: 'B2',
    6: 'B1', 7: 'A', 8: 'A', 9: 'A', 10: 'A'
}

ordered_classes = ['A', 'B1', 'B2', 'C1', 'C2', 'D', 'E']
classe_to_int = {classe: i for i, classe in enumerate(ordered_classes)}
int_to_classe = {i: classe for i, classe in enumerate(ordered_classes)}

# Calcular médias de referência ponderadas para cada classe usando dados ponderados normalizados
referencias_classes = {}
for classe_valor, classe_nome in classe_labels.items():
    grupo = df[df['classes (Renda per capita)'] == classe_valor]
    grupo_weighted = df_weighted.loc[grupo.index][features]
    grupo_valores = grupo_weighted.values

    if grupo_valores.shape[0] > 0 and grupo_valores.shape[1] == len(features):
        if grupo_valores.shape[0] == 1:
            media_ponderada = grupo_valores.flatten()
        else:
            media_ponderada = np.mean(grupo_valores, axis=0)
        referencias_classes[classe_to_int[classe_nome]] = media_ponderada
    else:
        print(f"[AVISO] Grupo da classe {classe_valor} ({classe_nome}) inválido ou vazio. Pulando...")

# Executar o algoritmo de propagação de rótulos
communities = list(nx.community.label_propagation_communities(G))

# Atribuir rótulo de comunidade e classe social com base nas médias ponderadas
ordered_community_labels = ['A', 'B1', 'B2', 'C1', 'C2', 'D', 'E']

for community_id, nodes in enumerate(communities):
    indices = list(nodes)
    grupo_comunidade_weighted = df_weighted.iloc[indices][features]
    media_comunidade = np.mean(grupo_comunidade_weighted.values, axis=0)

    classe_id = min(referencias_classes.items(), key=lambda item: np.linalg.norm(media_comunidade - item[1]))[0]
    melhor_classe = int_to_classe[classe_id]

    community_label = ordered_community_labels[community_id] if community_id < len(ordered_community_labels) else f"Com{community_id}"

    for node in nodes:
        G.nodes[node]['community'] = community_label

# Salvar o grafo
nx.write_graphml(G, 'classificacao_social.graphml')
print("Classificação concluída! Resultados salvos em 'classificacao_social.graphml'")

# Gerar PDF com os alunos por comunidade
pdf = FPDF()
pdf.add_page()
pdf.set_font("Arial", size=12)

# Cabeçalho
pdf.image("logo.png", x=10, y=8, w=30)
pdf.image("logoimc.png", x=pdf.w - 40, y=8, w=30)

pdf.set_font("Arial", "B", 16)
pdf.cell(0, 10, "Classificação Social por Comunidade", ln=True, align="C")
pdf.cell(0, 10, "Universidade Federal de Itajubá", ln=True, align="C")
pdf.cell(0, 10, "CMAC03 - Algoritmos em Grafos", ln=True, align="C")
pdf.ln(5)

# Legenda
pdf.set_font("Arial", "B", 10)
pdf.cell(0,10,"Todos os dados foram transformados em valores entre 0 e 1", align="C")
pdf.ln(10)
pdf.set_font("Arial", "I", 10)
legenda = ", ".join(features)
pdf.multi_cell(0, 8, f"Atributos (em ordem): {legenda}")
pdf.ln(3)

# Alunos por comunidade
comunidades_alunos = {}
for node, data in G.nodes(data=True):
    comunidade = data['community']
    aluno_id = data['id_discente']
    if comunidade not in comunidades_alunos:
        comunidades_alunos[comunidade] = []
    comunidades_alunos[comunidade].append(aluno_id)

pdf.set_font("Arial", size=12)
def comunidade_key(x):
    try:
        return ordered_community_labels.index(x)
    except ValueError:
        return len(ordered_community_labels) + 1

for comunidade in sorted(comunidades_alunos.keys(), key=comunidade_key):
    pdf.set_font("Arial", "B", 14)
    pdf.cell(0, 10, f"Comunidade {comunidade}", ln=True)
    pdf.set_font("Arial", size=12)
    for aluno in sorted(comunidades_alunos[comunidade]):
        atributos = df_weighted[df_weighted['id_discente'] == aluno][features].values.flatten()
        atributos_texto = ', '.join(f"{v:.2f}" for v in atributos)
        pdf.cell(0, 8, f"ID {aluno} - {atributos_texto}", ln=True)
    pdf.ln(5)

pdf.output("relatorio_comunidades.pdf")
print("Relatório PDF gerado: relatorio_comunidades.pdf")

# Salvar DataFrame com comunidade
comunidade_map = {data['id_discente']: data['community'] for node, data in G.nodes(data=True)}
df['community'] = df['id_discente'].map(comunidade_map)

if 'classe_final' in df.columns:
    df.drop(columns=['classe_final'], inplace=True)

df.to_excel("comunidades_completas.xlsx", index=False)
print("Arquivo comunidades_completas.xlsx gerado com sucesso.")