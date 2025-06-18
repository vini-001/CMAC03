import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from networkx.algorithms.community import girvan_newman
from networkx.algorithms.community.quality import modularity
import itertools

df = pd.read_excel("Solicitantes de Auxílio Estudantil - 2018.xlsx")

# Pré-processamento
categorical_cols = [
    'Qual sua PROCEDÊNCIA ESCOLAR?',
    'Qual a situação da MORADIA DO ALUNO?',
    'Qual a situação da MORADIA DO GRUPO FAMILIAR?',
    'Qual o principal meio de transporte que você utiliza para vir até a Universidade?'
]
numerical_cols = [
    'Quantos filhos o solicitante possui?',
    'Renda per capita',
    'Despesas per capita',
    'Quantidade de individuos com doença grave no grupo familiar',
    'Familiares com Superior Completo ou Pós',
    'Valor Total dos bens familiares'
]

encoder = OneHotEncoder(sparse_output=False)
encoded_categorical = encoder.fit_transform(df[categorical_cols])
feature_names = encoder.get_feature_names_out(categorical_cols)

scaler = StandardScaler()
scaled_numerical = scaler.fit_transform(df[numerical_cols])

# Combinação de características
features = np.concatenate([scaled_numerical, encoded_categorical], axis=1)

# Construir grafo único
G = nx.Graph()
for i, id in enumerate(df['id_discente']):
    G.add_node(id, classe_economica=df.iloc[i]['classes (Renda per capita)'])

similarity_matrix = cosine_similarity(features)

# Otimização: encontrar pares com similaridade > 0.7 sem loop duplo
threshold = 0.7
pairs = np.argwhere(np.triu(similarity_matrix, k=1) > threshold)
for i, j in pairs:
    G.add_edge(df.iloc[i]['id_discente'], df.iloc[j]['id_discente'])

# Girvan-Newman
if G.number_of_edges() > 0:
    comp = girvan_newman(G)
    partitions = []
    modularities = []
    for partition in itertools.islice(comp, 50):
        partitions.append(partition)
        mod = modularity(G, partition)
        modularities.append(mod)
        if len(modularities) > 2 and modularities[-1] < modularities[-2]:
            break
    best_partition = partitions[np.argmax(modularities)]

    # Atribuir comunidades como atributo do nó
    results = []
    for comm_id, community in enumerate(best_partition):
        for student_id in community:
            G.nodes[student_id]['comunidade'] = comm_id
            results.append({
                'id_discente': student_id,
                'classe_economica': G.nodes[student_id]['classe_economica'],
                'comunidade': comm_id
            })

    # Resultados
    results_df = pd.DataFrame(results)
    print("Comunidades detectadas:")
    for comm_id in results_df['comunidade'].unique():
        students = results_df[results_df['comunidade'] == comm_id]['id_discente'].tolist()
        print(f"  Comunidade {comm_id}: Alunos {students}")

    # Exportar grafo para Gephi (GML)
    nx.write_gml(G, "grafo_unico_comunidades.gml")
else:
    print("O grafo não possui arestas suficientes para detectar comunidades.")