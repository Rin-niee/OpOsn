import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog

# 𝑐=[{−𝑝𝑘},𝑟𝑒𝑠ℎ𝑎𝑝𝑒(𝑐𝑖𝑗,1,𝑁2),𝑧𝑒𝑟𝑜𝑠(1,𝑁2)]1×(𝐾+2𝑁2)
# 𝐴𝑒𝑞=[𝐴𝑒𝑞(3);𝐴𝑒𝑞(4);𝐴𝑒𝑞(11)](1+𝑁+1)×(𝐾+2𝑁2)
#𝑏𝑒𝑞=𝑧𝑒𝑟𝑜𝑠(1,1+𝑁)
# 𝐴=[𝐴(2);𝐴(7)](𝐿+𝑁2)×(𝐾+2𝑁2) 
#𝐴(2)=[𝐴𝑙𝑘,𝑧𝑒𝑟𝑜𝑠(𝐿,2𝑁2)]𝐿×(𝐾+2𝑁2)
# 𝑏=𝑏𝑙
# 𝑏𝑜𝑢𝑛𝑑𝑠=[𝑧𝑒𝑟𝑜𝑠(1,(𝐾+2𝑁2));[{𝑄𝑘},𝑧𝑒𝑟𝑜𝑠(1,)]]

# def reverse(matrix, r1, r2):
#     for row in matrix:
#         row[r1], row[r2] = row[r2], row[r1]
#     for i in range(len(matrix)):
#         matrix[r1][i], matrix[r2][i] = matrix[r2][i], matrix[r1][i]
#     return matrix

K = int(input("введите количество типов товаров"))  # количество типов товаров
L = int(input("введите количество типов сырья")) # количество типов сырья

pk = np.random.randint(100, size=(K)) # цена реализации товара типа k
qk = np.random.randint(30, size=(K)) #спрос на товары типа k
bl = np.random.randint(1000, size=(L))  # запас сырья типа l

Alk = np.random.randint(20, size=(K, L))

# транспортный граф с пропускными спрособностями

D = np.array([[0,  90, 60,   75,  0,  0,   0,  0,   0,  0,  0,  0,    0,   0,   0,   0,    0,   0], #1
              [0,   0,  0,   0, 25,  0,   0,  0,   0,  0,  0,  0,    0,   0,   0,   0,    0,   0],  #2
              [0,   0,  0,   0,  0, 70,   0,  0,   0,  0,  0,  0,    0,   0,   0,   0,    0,   0], #3
              [0,   0,  0,   0,  0,  0,   75,  0,   0,  0,  0,  0,    0,   0,   0,   0,    0,   0], #4
              [0,   0,  0,   0,  0,  0,   0, 25,   0,  0,  0,  0,    0,   0,   0,   0,    0,   0], #5
              [0,   0,  0,   0,  0,  0,   0,  0,  70,  0,  0,  0,    0,   0,   0,   0,    0,   0], #6
              [0,   0,  0,   0,  0,  0,   0,  0,   0,  75,  0,  0,    0,   0,   0,   0,    0,   0], #7
              [0,   0,  0,   0,  0,  0,   0,  0,   0,  0, 25,  0,    0,   0,   0,   0,    0,   0], #8
              [0,   0,  0,   0,  0,  0,   0,  0,   0,  0,  0, 70,    0,   0,   0,   0,    0,   0], #9
              [0,   0,  0,   0,  0,  0,   0,  0,   0,  0,  0,  0,    75,   0,   0,   0,    0,   0], #10
              [0,   0,  0,   0,  0,  0,   0,  0,   0,  0,  0,  0,   25,   0,   0,   0,    0,   0], #11
              [0,   0,  0,   0,  0,  0,   0,  0,   0,  0,  0,  0,   70,   0,   0,   0,    0,   0], #12
              [0,   0,  0,   0,  0,  0,   0,  0,   0,  0,  0,  0,    0,  85,  50,   0,    0,   0], #13
              [0,   0,  0,   0,  0,  0,   0,  0,   0,  0,  0,  0,    0,   0,   0,  35,   50,   0], #14
              [0,   0,  0,   0,  0,  0,   0,  0,   0,  0,  0,  0,    0,   0,   0,   0,   30,  20], #15
              [0,   0,  0,   0,  0,  0,   0,  0,   0,  0,  0,  0,    0,   0,   0,   0,    0,   0], #16
              [0,   0,  0,   0,  0,  0,   0,  0,   0,  0,  0,  0,    0,   0,   0,   0,    0,   0], #17
              [0,   0,  0,   0,  0,  0,   0,  0,   0,  0,  0,  0,    0,   0,   0,   0,    0,   0]]) #18

#цена перевоза товара из одного пункта в другой
cij = np.array([[0,  10, 5,   7,  0,  0,   0,  0,   0,  0,  0,  0,    0,   0,   0,   0,    0,   0], #1
              [0,   0,  0,   0, 25,  0,   0,  0,   0,  0,  0,  0,    0,   0,   0,   0,    0,   0],  #2
              [0,   0,  0,   0,  0, 12,   0,  0,   0,  0,  0,  0,    0,   0,   0,   0,    0,   0], #3
              [0,   0,  0,   0,  0,  0,   7,  0,   0,  0,  0,  0,    0,   0,   0,   0,    0,   0], #4
              [0,   0,  0,   0,  0,  0,   0, 25,   0,  0,  0,  0,    0,   0,   0,   0,    0,   0], #5
              [0,   0,  0,   0,  0,  0,   0,  0,  12,  0,  0,  0,    0,   0,   0,   0,    0,   0], #6
              [0,   0,  0,   0,  0,  0,   0,  0,   0,  7,  0,  0,    0,   0,   0,   0,    0,   0], #7
              [0,   0,  0,   0,  0,  0,   0,  0,   0,  0, 25,  0,    0,   0,   0,   0,    0,   0], #8
              [0,   0,  0,   0,  0,  0,   0,  0,   0,  0,  0, 12,    0,   0,   0,   0,    0,   0], #9
              [0,   0,  0,   0,  0,  0,   0,  0,   0,  0,  0,  0,    7,   0,   0,   0,    0,   0], #10
              [0,   0,  0,   0,  0,  0,   0,  0,   0,  0,  0,  0,   25,   0,   0,   0,    0,   0], #11
              [0,   0,  0,   0,  0,  0,   0,  0,   0,  0,  0,  0,   12,   0,   0,   0,    0,   0], #12
              [0,   0,  0,   0,  0,  0,   0,  0,   0,  0,  0,  0,    0,  23,  50,   0,    0,   0], #13
              [0,   0,  0,   0,  0,  0,   0,  0,   0,  0,  0,  0,    0,   0,   0,  3,   5,   0], #14
              [0,   0,  0,   0,  0,  0,   0,  0,   0,  0,  0,  0,    0,   0,   0,   0,   7,  23], #15
              [0,   0,  0,   0,  0,  0,   0,  0,   0,  0,  0,  0,    0,   0,   0,   0,    0,   0], #16
              [0,   0,  0,   0,  0,  0,   0,  0,   0,  0,  0,  0,    0,   0,   0,   0,    0,   0], #17
              [0,   0,  0,   0,  0,  0,   0,  0,   0,  0,  0,  0,    0,   0,   0,   0,    0,   0]]) #18

N = len(D)
#матрица инцидентности +
def matrix_incidence(x, matrix_size):
    D1 = np.zeros((matrix_size, matrix_size))
    for i in range(matrix_size):
        for j in range(matrix_size):
            if x[i, j] != 0:
                D1[i, j] = 1
            elif ((x[i, j] == 0) and (x[j, i]!=0)):
                D1[i, j] = -1
    return D1
D1 = matrix_incidence(D, N)
#рисование графа
def GraphDraw(x, x1):
    G = nx.DiGraph()
    for i in range(matrix_size):
        G.add_node(i)
        for j in range(matrix_size):
            if (x1[i, j] > 0):
                G.add_edges_from([(i, j)])
                G.add_edge(i, j, weight=x[i][j])
    pos = nx.circular_layout(G)
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, with_labels=True)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()
    return G
D1 = GraphDraw(D, N)

def celevaya(x1, x2, x):
    C = []
    c1 = -np.array(x11)
    c2 = np.reshape(x2, x**2)
    c3 = np.zeros((x**2))
    C.extend(c1)
    C.extend(c2)
    C.extend(c3)
    return C
C = celevaya(pk, cij, N)

A_eq = []

A_eq3 = [1]*K + [0]*(N **2) + [0] + [-1]*(N - 1) + [0]*(N * (N - 1))
A_eq4 = [[0] * N ** 2 for i in range(N)]
    for i in range(N):
        for j in range(N):
            if d[i][j] != 0:
                A_eq4[i][i * N + j] = -1
                A_eq4[j][i * N + j] = 1
A_eq4 = A_eq4[1: N - 1]
A_eq11 = [1] * K + [0] * (N ** 2) + ([0] * (N - 1) + [-1]) * N  # = 0 (11)
A_eq.extend(A_eq3)
A_eq.extend(A_eq11)
A_eq.extend(A_eq4)
b_eq = np.zeros(1, N+1)


A_ub=[]
b_ub=[]
for l in range(L):
  A_ub.append(list(a[l]) + [0] * 2 * (N ** 2))
b_ub += list(b)
for k in range(K):  # (5)
    A_ub.append([0] * (K + 2 * (N ** 2)))
    A_ub[-1][k] = 1

b_ub += list(q)

for j in range(N ** 2):  # (6)
    A_ub.append([0] * (K + 2 * N ** 2))
    A_ub[-1][K + N ** 2 + j] = 1

b_ub += list(d.flatten())

res = linprog(c=C, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, integrality=np.ones(K + 2 * N ** 2))
r = np.array((res['x']))


