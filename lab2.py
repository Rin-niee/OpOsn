import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog

#создание исходной матрицы +
               #1   #2  #3  #4  #5  #6   #7  #8    #9  #10 #11 #12  #13  #14  #15 #16  #17  #18
D = np.array([[0,   0,  0,   5,  0,  0,   0,  0,   0,  0,  0,  0,    0,   0,   0,   0,    0,   0], #1
              [0,   0,  0,   0, 25,  0,   0,  0,   0,  0,  0,  0,    0,   0,   0,   0,    0,   0],  #2
              [0,   0,  0,   0,  0, 70,   0,  0,   0,  0,  0,  0,    0,   0,   0,   0,    0,   0], #3
              [0,   0,  0,   0,  0,  0,   5,  0,   0,  0,  0,  0,    0,   0,   0,   0,    0,   0], #4
              [0,   0,  0,   0,  0,  0,   0, 25,   0,  0,  0,  0,    0,   0,   0,   0,    0,   0], #5
              [0,   0,  0,   0,  0,  0,   0,  0,  70,  0,  0,  0,    0,   0,   0,   0,    0,   0], #6
              [0,   0,  0,   0,  0,  0,   0,  0,   0,  5,  0,  0,    0,   0,   0,   0,    0,   0], #7
              [0,   0,  0,   0,  0,  0,   0,  0,   0,  0, 25,  0,    0,   0,   0,   0,    0,   0], #8
              [0,   0,  0,   0,  0,  0,   0,  0,   0,  0,  0, 70,    0,   0,   0,   0,    0,   0], #9
              [0,   0,  0,   0,  0,  0,   0,  0,   0,  0,  0,  0,    5,   0,   0,   0,    0,   0], #10
              [0,   0,  0,   0,  0,  0,   0,  0,   0,  0,  0,  0,   25,   0,   0,   0,    0,   0], #11
              [0,   0,  0,   0,  0,  0,   0,  0,   0,  0,  0,  0,   70,   0,   0,   0,    0,   0], #12
              [0,   0,  0,   0,  0,  0,   0,  0,   0,  0,  0,  0,    0,  75,  60,   0,    0,   0], #13
              [0,   0,  0,   0,  0,  0,   0,  0,   0,  0,  0,  0,    0,   0,  0,   18,   57,   0], #14
              [0,   0,  0,   0,  0,  0,   0,  0,   0,  0,  0,  0,    0,   0,   0,   0,   40,  20], #15
              [0,   0,  0,   0,  0,  0,   0,  0,   0,  0,  0,  0,    0,   0,   0,   0,    0,   0], #16
              [0,   0,  0,   0,  0,  0,   0,  0,   0,  0,  0,  0,    0,   0,   0,   0,    0,   0], #17
              [0,   0,  0,   0,  0,  0,   0,  0,   0,  0,  0,  0,    0,   0,   0,   0,    0,   0]]) #18
matrix_size = len(D)

#матрица инцедентности +
def matrix_incidence(x):
    D1 = np.zeros((matrix_size, matrix_size))
    for i in range(matrix_size):
        for j in range(matrix_size):
            if D[i, j] != 0:
                D1[i, j] = 1
            elif ((D[i, j] == 0) and (D[j, i]!=0)):
                D1[i, j] = -1
    return D1
D1 = matrix_incidence(D)
print("incidence matrix: \n", D1)

#подсчет количества ребер и вершин +
edges_num = 0
matrix_size = len(D)
for i in range(matrix_size):
    for j in range(matrix_size):
        if D1[i, j] > 0:
         edges_num = edges_num + 1
#рисование графа +
def GraphDraw(x):
    G = nx.DiGraph()
    for i in range(matrix_size):
        G.add_node(i)
        for j in range(matrix_size):
            if (D1[i, j] > 0):
                G.add_edges_from([(i, j)])
                G.add_edge(i, j, weight=D[i][j])
    return G

G = GraphDraw(D1)
pos = nx.circular_layout(G)
plt.figure(figsize=(8, 8))
nx.draw(G, pos, with_labels=True)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.show()

# Создание целевой функции +
def sinks(x3):
    c = np.zeros((edges_num))
    m = -1
    for i in range(matrix_size):
        for j in range(matrix_size):
            if (D1[i, j]== 1):
                m=m+1
                if ((j == 15 or j == 16 or j == 17) & (m <= edges_num)):
                    c[m] = 1
    return c
c = sinks(D1)
print("Celevaya: ", c)
#матрица A_eq
def conservation(x4):
    A_eq = np.zeros((matrix_size, matrix_size)) #создает матрицу смежности, если есть из в i -> j, то 1, наоборот -1
    e = 0
    for j in range(matrix_size):
        for i in range(matrix_size):
            if (D1[i, j] == 1):
                A_eq[i, e] = 1
                A_eq[j, e] = -1
                e+=1
    return A_eq
A_eq = conservation(D1)
A_eq1 = A_eq[3:16]
print("Matrix A_eq :\n", A_eq)

b_eq = np.zeros(A_eq1.shape[0])

b_ub = np.zeros((edges_num))
m = -1
for i in range(matrix_size):
    for j in range(matrix_size):
        if (D1[i, j] > 0  & (m < edges_num)):
            m+=1
            b_ub[m] = D[i, j]
print(b_ub)

A_ub = np.eye(b_ub.size)

# #Решение задачи линейного программирования
res = linprog(-c, A_eq=A_eq1, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, method = "HiGHS")
print(res)
r = np.array(list(res['x']))
# Вывод максимального потока
print(r)

matrix = np.zeros((matrix_size, matrix_size))
e = 0
for i in range(matrix_size):
    for j in range(matrix_size):
        if (D1[i, j]==1):
            matrix[i, j] = r[e]
            e+=1
G1 = nx.DiGraph()
for i in range(matrix_size):
    for j in range(matrix_size):
        if (matrix[i, j]>0 and D1[i, j] > 0):
            G1.add_node(i)
            G1.add_edges_from([(i, j)])
            G1.add_edge(i, j, weight=matrix[i][j])

pos = nx.circular_layout(G1)
plt.figure(figsize=(8, 8))
nx.draw(G1, pos, with_labels=True)
labels = nx.get_edge_attributes(G1, 'weight')
nx.draw_networkx_edge_labels(G1, pos, edge_labels=labels)
plt.show()
