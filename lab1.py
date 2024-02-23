import numpy as np
from scipy.optimize import linprog
import networkx as nx
import matplotlib.pyplot as plt


I = int((input("Введите размер спроса:")))
J = int((input("Введите размер предложения:")))
cena = np.random.randint(100, size=(I, J)) #26

m, n = cena.shape
C = list(np.reshape(cena, n * m))
b1 = np.random.randint(100, size=I)
b = np.random.randint(100, size=J)
A = np.zeros([m, m * n])
for i in range(n):
    for j in range(n * m):
        if i * n <= j <= n + i * n - 1:
            A[i, j] = 1
A1 = np.zeros([n, m * n])
for i in range(n):
    p = 0
    for j in range(n * m):
        if j == p * n + i:
            A1[i, j] = 1
            p += 1
if np.sum(b1) < np.sum(b):
    res = linprog(c=C, A_ub=A1, b_ub=b, A_eq=A, b_eq=b1, method='HiGHS')
elif np.sum(b1) > np.sum(b):
    res = linprog(c=C, A_ub=A, b_ub=b1, A_eq=A1, b_eq=b, method='HiGHS')
elif np.sum(b1) == np.sum(b):
    A_eq = np.concatenate((A, A1), axis=0)
    b_eq = np.concatenate((b1, b), axis=0)
    res = linprog(c=C, A_eq=A_eq, b_eq=b_eq, method='HiGHS')
result = np.array(list(res['x'])).reshape(I, J)
np.reshape(result, (21, 42))
print(result)
G = nx.Graph()
variables = [f'x{i+1}' for i in range(I*J)]
constraints = [f'c{i+1}' for i in range(I+J)]
G.add_nodes_from(variables, bipartite=0)
G.add_nodes_from(constraints, bipartite=1)
edges = [(variables[i], constraints[j]) for i in range(I*J) for j in range(I+J) if res['x'][i] > 0]
G.add_edges_from(edges)
pos = nx.bipartite_layout(G, nodes=variables)
nx.draw(G, pos, with_labels=True, node_color=['skyblue' if node in variables else 'salmon' for node in G.nodes])
print(f"x = {result} \n \nfun = {res['fun']}, {res['message']}")
plt.show()

