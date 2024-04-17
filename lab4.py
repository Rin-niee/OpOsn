import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import networkx as nx

K = int(input("введите количество типов товаров"))  # количество типов товаров
L = int(input("введите количество типов сырья"))
M = int(input("введите количество дней"))
N = 12
Pkm = np.random.randint(100, size=(K, M)) #цена реализации типа k в день m
Ylm = np.random.randint(50, size=(L, M)) #объем сырья l в день m
Alk = np.random.randint(20, size=(K, L)) #бъем сырья типа l для производства товара типа k
Qk = np.random.randint(30, size=(K)) #спрос товаров типа k

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
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])  #<-матрица пропускной способности
Cij = np.array([[0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 3, 4, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 2, 3, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 7, 70, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 9, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 3, 4, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 3, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 7, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 3, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 6],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]) #<-матрица стоимости перевозок
# res = [x11,lyam11, z11, b11]
# 𝑥𝑘𝑚 – объем производства товара типа 𝑘 в день 𝑚;
# 𝜆𝑖𝑗 – факт перевозки товаров из пункта 𝑖 в 𝑗;
# 𝑧𝑖𝑗 – объем перевозки товаров из пункта 𝑖 в 𝑗;
# 𝑏𝑙𝑚 – запас сырья типа 𝑙 в день 𝑚;
def celevaya(Pkm, Cij, K, M, N):
# reshape(p), reshape(c), 0(нет z, как и b)
    C1= list(-np.reshape(Pkm, K*M))
    C2 = list(np.reshape(Cij, N*N))
    C34 = [0]*(L*M + N ** 2)
    C = []
    C.extend(C1)
    C.extend(C2)
    C.extend(C34)
    return C
C = celevaya(Pkm, Cij, K, M, N)
A_eq = []
b_eq = []
#3, 4, 11, 12
# ∑𝑥𝑘𝑚𝑘,𝑚=∑𝑧1𝑗𝑗 (3)
# ∑𝑧𝑖𝑗𝑗=∑𝑧𝑗𝑖𝑗 (4)
# ∑𝑧𝑖𝑁𝑖=∑𝑥𝑘𝑚𝑘,𝑚 (11)
# 𝑏𝑙(𝑚+1)=𝑏𝑙𝑚−∑𝐴𝑙𝑘𝑥𝑘𝑚𝑘+𝛾𝑙𝑚 (12)
A_eq3 = [1] * (K*M) + [0] * (N ** 2) + [0] + [-1] * (N - 1) + [0] * (N * (N - 1))+ [0] *(L*M)
#print(A_eq3)
A_eq4 = [[0] * N ** 2 for i in range(N)]
for i in range(N):
    for j in range(N):
        if D[i][j] != 0:
            A_eq4[i][i*N + j] = 1
            A_eq4[j][i*N + j] = -1
for i in range(N):
    A_eq4[i] = [0] * (K + N ** 2) + A_eq4[i]
A_eq4 = A_eq4[1: N - 1]
A_eq11 = [1]*(K*M) + [0]*(N ** 2) + ([0]*(N - 1)+[-1])*N + [0]*(L*M)
A_eq.append(A_eq3)
A_eq.append(A_eq4)
A_eq.append(A_eq11)
#тут должно быть 12 условие, но его украли цыгане, извините
#2, 5, 6, 7, 9
A_ub = []
b_ub = []
#в условиях перепроверить надо, они  стремные.
#2
for l in range(L):
    A_ub.append(list(Alk[l]) + [0] * 2 * (N ** 2)+[-1]*(L*M))
b_ub += 0
#5
for k in range(K):
    A_ub.append([1] * (K*M) + [0]*(2 * (N ** 2))) + [0]*(L*M)
b_ub += list(Qk)
#6
for j in range(N ** 2):
    A_ub.append([0] * (K + 2 * N ** 2))
    A_ub[-1][K + N ** 2 + j] = 1
A_ub+=[0]*(M*L)
b_ub += list(D.flatten())
#7
for j in range(N ** 2):
    A_ub.append([0] * (K + 2 * N ** 2))
    A_ub[-1][K + j] = -10
    A_ub[-1][K + N ** 2 + j] = 1
A_ub+=[0]*(M*L)
b_ub += [0] * (N ** 2)
#9
for j in range(N ** 2):
    A_ub.append([0] * (K + M*L + 2 * N ** 2))
    A_ub[-1][K + j] = 1

b_ub += [1] * (N ** 2)
res = linprog(c=C, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, integrality=np.ones(K+2*N**2))
# Решаем задачу
b = res.x
b = b[-(len(Cij) ** 2):].reshape(len(Cij), len(Cij))
print(b, "\nGraph ======\n", D, "\nResolution======\n", res.x)

def vs(matrix1, matrix2, matrix3=[]):
    G = nx.DiGraph(matrix1)
    pos = nx.spring_layout(G)  # Определяем позиции узлов
    nx.draw(G, pos, with_labels=True, node_size=700,
            node_color="lightblue")  # Рисуем граф
    edge_labels = {}
    for i, j, w in G.edges(data=True):
        label = f'{w["weight"] if w["weight"] != 0 else 0} , {matrix2[i, j]}'
        if len(matrix3) > 0:
            label += f' | {matrix3[i, j]}'
        edge_labels[(i, j)] = label
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='blue',font_size=7)  # Рисуем подписи на ребрах
    plt.show()
vs(D, Cij)

vs(b, D, Cij)  # Визуализация доставки товаров,  пропускной способности, стоимости перевозки
