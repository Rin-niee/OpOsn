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

K = int(input("введите количество типов товаров"))  # количество типов товаров
L = int(input("введите количество типов сырья"))
N = 12
Pk = np.random.randint(100, size=(K))
Bl = np.random.randint(30, size=(K))
Alk = np.random.randint(20, size=(K, L))
Qk = np.random.randint(30, size=(K))

D = np.array([[0, 90, 15, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 50, 43, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 55,  4, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 54, 70, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 90, 78, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 14, 69, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 60, 59, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 21],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
Cij = np.array([[0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 7, 70, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 3, 4, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 7, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 3, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])


def solution(K, L, N, p, b, a, d, q, c):
    C = list(np.negative(p)) + list(c.flatten()) + [0]*(N ** 2)
    A_eq = []
    b_eq = []
    A_eq3 = [1] * K + [0] * (N ** 2) + [0] + [-1] * (N - 1) + [0] * (N * (N - 1))
    A_eq.append(A_eq3)
    b_eq.append(0)
    A_eq11 = [1] * K + [0] * (N ** 2) + ([0] * (N - 1) + [-1]) * N
    A_eq.append(A_eq11)
    b_eq.append(0)
    A_eq4 = [[0] * N ** 2 for i in range(N)]
    for i in range(N):
        for j in range(N):
            if d[i][j] != 0:
                A_eq4[i][i * N + j] = -1
                A_eq4[j][i * N + j] = 1

    for i in range(N):
        A_eq4[i] = [0] * (K + N ** 2) + A_eq4[i]
    A_eq4 = A_eq4[1: N - 1]

    A_eq += A_eq4
    b_eq += [0] * (N - 2)

    A_ub = []
    b_ub = []
    for l in range(L):
        A_ub.append(list(a[l]) + [0] * 2 * (N ** 2))
    b_ub += list(b)
    for k in range(K):
        A_ub.append([0] * (K + 2 * (N ** 2)))
        A_ub[-1][k] = 1
    b_ub += list(q)
    for j in range(N ** 2):  # (6)
        A_ub.append([0] * (K + 2 * N ** 2))
        A_ub[-1][K + N ** 2 + j] = 1
    b_ub += list(d.flatten())

    res = linprog(c=C, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)
    return res


def vs(matrix1, matrix2, matrix3=[]):
    G = nx.DiGraph(matrix1)
    pos = nx.circular_layout(G)
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, with_labels=True)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()
vs(D, Cij)

# Решение
res = solution(K, L, N, Pk, Bl, Alk, D, Qk, Cij)
a = res.x
a = a[-2 * (len(Cij) ** 2):-(len(Cij) ** 2):1].reshape(len(D), len(D))
b = res.x
b = b[-(len(Cij) ** 2):].reshape(len(Cij), len(Cij))
print(a, "\n======\n", b, "\n======\n", D, "\n======\n", res.x)

vs(b, D, Cij)  # Визуализация доставки товаров,  пропускной способности, стоимости перевозки

