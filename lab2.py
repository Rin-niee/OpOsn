import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog
import random

num_input_vertices = int(input("Введите количество входных вершин"))
num_output_vertices = int(input("Введите количество выходных вершин"))
min_num_intermediate_vertices = int(input("Введите количество вершин переходных"))
matrix_size = num_input_vertices + num_output_vertices + min_num_intermediate_vertices

def generate_graph(num_input_vertices, num_output_vertices, min_num_intermediate_vertices):
    matrix_size = num_input_vertices + num_output_vertices + min_num_intermediate_vertices
    D = [[0 for _ in range(matrix_size)] for _ in range(matrix_size)]
    for i in range(num_input_vertices):
        j = random.randint(num_input_vertices, matrix_size - 1)
        D[i][j] = random.randint(1, 100)
    for i in range(num_input_vertices, num_input_vertices + num_output_vertices):
        j = random.randint(0, num_input_vertices - 1)
        D[j][i] = random.randint(1, 100)

    for i in range(num_input_vertices + num_output_vertices, matrix_size):
        j = random.randint(0, num_input_vertices - 1)
        k = random.randint(num_input_vertices, matrix_size - 1)
        D[j][i] = random.randint(1, 100)
        D[i][k] = random.randint(1, 100)
    for i in range(matrix_size):
        D[i][i] = 0
    return D

D = generate_graph(num_input_vertices, num_output_vertices, min_num_intermediate_vertices)

G = nx.Graph()
for i in range(matrix_size):
    G.add_node(i)
    for j in range(matrix_size):
        if D[i][j] != 0:
            G.add_edge(i, j, weight=D[i][j])
pos = nx.circular_layout(G)
plt.figure(figsize=(8, 8))
nx.draw(G, pos, with_labels=True, node_size=2000, font_size=10, node_color='skyblue', edge_color='gray')
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
#plt.title("Graph Visualization")
#plt.axis('off')
plt.show()

A_ub = np.zeros((matrix_size, matrix_size)) #создает матрицу смежности, если есть из в i -> j, то 1, наоборот -1 и аналогичично в обратную сторону
for i in range(matrix_size):
    for j in range(matrix_size):
        if D[i][j] > 0:
            A_ub[i][j] = 1
            A_ub[j][i] = -1
print(A_ub)

b_ub = np.zeros(matrix_size)

# Создание целевой функции
c = np.zeros(matrix_size)
c[0] = 1
c[matrix_size - 1] = -1

# Решение задачи линейного программирования
res = linprog(c, A_ub=A_ub, b_ub=b_ub, method='highs')
sv = res.x

# Вывод максимального потока
print(sv)
