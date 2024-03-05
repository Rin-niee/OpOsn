
import networkx as nx
import matplotlib.pyplot as plt
from scipy.optimize import linprog
import numpy as np

num_vertices = 5
num_edges = 10

# Создание матрицы инцидентности
incidence_matrix = np.zeros((num_edges, num_vertices))
# Логические индексы для выбора минимального числа ребер из отправных пунктов I в пункты J
indices = np.random.choice(range(num_edges), 5, replace=False)
for i, j in enumerate(indices):
    incidence_matrix[j, i] = 1

G = nx.Graph()
for i in range(num_vertices):
    G.add_node(i)
    for j in range(num_vertices):
        if incidence_matrix[i,j] != 0:
            G.add_edge(i, j,  weight=incidence_matrix[i,j])
pos = nx.circular_layout(G)
plt.figure(figsize=(8, 8))
nx.draw(G, pos, with_labels=True, node_size=2000, font_size=10, node_color='skyblue', edge_color='gray')
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
#plt.title("Graph Visualization")
#plt.axis('off')
plt.show()

# Вектор пропускных способностей
capacities = np.random.randint(1, 10, num_edges)

# Создание коэффициентов целевой функции
c = np.zeros(num_edges)
# Максимизация суммарного потока из отправных пунктов I в пункты J
c[indices] = -1

# Создание ограничений равенства
# A_eq = np.concatenate((incidence_matrix, -incidence_matrix), axis=0)
# b_eq = np.zeros(2 * num_edges)

# Создание ограничений неравенства
A_ub = np.identity(num_edges)
b_ub = capacities

# Решение линейной программы
result = linprog(c, A_ub=A_ub, b_ub=b_ub)
flows = result.x[indices]
print(flows)
