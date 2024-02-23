import numpy as np
from scipy.optimize import linprog
import networkx as nx
import matplotlib.pyplot as plt

#размер матрицы цен
I = int((input("Введите размер спроса:")))
J = int((input("Введите размер предложения:")))
#цены на товары
cena = np.random.randint(100, size=(I, J))
#матрицы в вектор
m, n = cena.shape
C = list(np.reshape(cena, n * m))
#ограничения
b1 = np.random.randint(100, size=I)
b = np.random.randint(100, size=J)
#ограничение предложений
A = np.zeros([m, m * n])
for i in range(n):
    for j in range(n * m):
        if i * n <= j <= n + i * n - 1:
            A[i, j] = 1
#ограничение спроса
A1 = np.zeros([n, m * n])
for i in range(n):
    p = 0
    for j in range(n * m):
        if j == p * n + i:
            A1[i, j] = 1
            p += 1
#проверка баланса
if np.sum(b1) < np.sum(b):
    res = linprog(c=C, A_ub=A1, b_ub=b, A_eq=A, b_eq=b1, method='HiGHS')
elif np.sum(b1) > np.sum(b):
    res = linprog(c=C, A_ub=A, b_ub=b1, A_eq=A1, b_eq=b, method='HiGHS')
elif np.sum(b1) == np.sum(b):
    A_eq = np.concatenate((A, A1), axis=0)
    b_eq = np.concatenate((b1, b), axis=0)
    res = linprog(c=C, A_eq=A_eq, b_eq=b_eq, method='HiGHS')
r = np.array(list(res['x'])).reshape(I, J)
np.reshape(r, (I, J))
print(r)
#граф
G = nx.Graph()
spros = [f'D{i}' for i in range(I)]
predloj = [f'S{i}' for i in range(J)]
G.add_nodes_from(spros, bipartite=0)
G.add_nodes_from(predloj, bipartite=1)
for i in range(I):
    for j in range(J):
        G.add_edge(spros[i], predloj[j], weight=cena[i][j])
res = nx.max_weight_matching(G)
pos = nx.bipartite_layout(G, spros)
nx.draw(G, pos, with_labels=True)
nx.draw_networkx_edge_labels(G, pos, edge_labels={(u, v): w for u, v, w in G.edges.data('weight')})
plt.show()
