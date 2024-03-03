import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog
import random

matrix_size = int(input("Введите количество вершин"))
num_input_vertices = int(input("Введите количество входных вершин"))
num_output_vertices = int(input("Введите количество выходных вершин"))
min_num_intermediate_vertices = 5

def generate_graph(num_input_vertices, num_output_vertices, min_num_intermediate_vertices):
    matrix_size = num_input_vertices + num_output_vertices + min_num_intermediate_vertices
    D = [[0 for _ in range(matrix_size)] for _ in range(matrix_size)]

    # Создание входных вершин
    for i in range(num_input_vertices):
        # Выбор случайной выходной вершины
        j = random.randint(num_input_vertices, matrix_size - 1)

        # Добавление ребра от входной вершины к выходной вершине
        D[i][j] = random.randint(1, 100)

    # Создание выходных вершин
    for i in range(num_input_vertices, num_input_vertices + num_output_vertices):
        # Выбор случайной входной вершины
        j = random.randint(0, num_input_vertices - 1)

        # Добавление ребра от входной вершины к выходной вершине
        D[j][i] = random.randint(1, 100)

    # Создание промежуточных вершин
    for i in range(num_input_vertices + num_output_vertices, matrix_size):
        # Выбор случайной входной вершины
        j = random.randint(0, num_input_vertices - 1)

        # Выбор случайной выходной вершины
        k = random.randint(num_input_vertices, matrix_size - 1)

        # Добавление ребра от входной вершины к промежуточной вершине
        D[j][i] = random.randint(1, 100)

        # Добавление ребра от промежуточной вершины к выходной вершине
        D[i][k] = random.randint(1, 100)

    # Удаление петель
    for i in range(matrix_size):
        D[i][i] = 0

    # Возврат матрицы смежности
    return D


D = generate_graph(num_input_vertices, num_output_vertices, min_num_intermediate_vertices)

G = nx.Graph()

# Добавляем вершины и ребра на основе матрицы смежности D
for i in range(matrix_size):
    G.add_node(i)
    for j in range(matrix_size):
        if D[i][j] != 0:
            G.add_edge(i, j, weight=D[i][j])

# Располагаем вершины графа
pos = nx.circular_layout(G)

# Рисуем граф
plt.figure(figsize=(8, 8))
nx.draw(G, pos, with_labels=True, node_size=2000, font_size=10, node_color='skyblue', edge_color='gray')
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.title("Graph Visualization")
plt.axis('off')
plt.show()

A_ub = np.zeros((matrix_size, matrix_size))
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
print(c)

# Решение задачи линейного программирования
res = linprog(c, A_ub=A_ub, b_ub=b_ub)
# Вывод максимального потока
print(res)
