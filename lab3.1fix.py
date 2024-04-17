import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import networkx as nx

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
              [0, 0, 0, 2, 3, 0, 0, 0, 0, 0, 0, 0],
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
            #-p                     reshape(c)      #0(нет z)
    C = list(np.negative(p)) + list(c.flatten()) + [0]*(N ** 2)
    A_eq = []
    b_eq = []
            #K        #лямбды(их нет)   #z,первая строка из 1(reverse), 1 аналогично ноль, потому что туда ничегоне везем
    A_eq3 = [1] * K + [0] * (N ** 2) + [0] + [-1] * (N - 1) + [0] * (N * (N - 1))
    print(A_eq3)
    A_eq.append(A_eq3)
    #аналогично 3, только уже заполняем N столбец
    A_eq11 = [1] * K + [0] * (N ** 2) + ([0] * (N - 1) + [-1]) * N
    A_eq.append(A_eq11)
    A_eq4 = [[0] * N ** 2 for i in range(N)] #по принципу задачи 2 <-тут размерность матрицы (как D и Cij, но вектором), цикл вложенный для матрицы этой
    for i in range(N):  #<-этот для нижнего
        for j in range(N):
            if d[i][j] != 0:
                A_eq4[i][i*N + j] = 1
                A_eq4[j][i*N + j] = -1
#изменение размерности матрицы A_eg4 согласно размерностям других матриц
    for i in range(N):
        #чтобы удовлетворить размерность матриц, вместо K, лямбд ставятся нули
        A_eq4[i] = [0] * (K + N ** 2) + A_eq4[i]
    A_eq4 = A_eq4[1: N - 1]

    A_eq += A_eq4
    b_eq += [0] * (N)

    A_ub = []
    b_ub = []
    #2
    for l in range(L):
        #есть только x, а лямбды и z нет
        A_ub.append(list(Alk[l]) + [0] * 2 * (N ** 2))
    b_ub += list(b)
    #5
    for k in range(K):
        A_ub.append([0] * (K + 2 * (N ** 2)))
        A_ub[-1][k] = 1
    b_ub += list(q)
    #6
    for j in range(N ** 2):
        A_ub.append([0] * (K + 2 * N ** 2))
        A_ub[-1][K + N ** 2 + j] = 1
    b_ub += list(d.flatten())
    #7
    for j in range(N ** 2):
        A_ub.append([0] * (K + 2 * N ** 2))
        A_ub[-1][K + j] = -10
        A_ub[-1][K + N ** 2 + j] = 1
    b_ub += [0] * (N ** 2)
    #9
    for j in range(N ** 2):
        A_ub.append([0] * (K + 2 * N ** 2))
        A_ub[-1][K + j] = 1

    b_ub += [1] * (N ** 2)

    res = linprog(c=C, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, integrality=np.ones(K+2*N**2))
    return res


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

# Решаем задачу
res = solution(K, L, N, Pk, Bl, Alk, D, Qk, Cij)
b = res.x
b = b[-(len(Cij) ** 2):].reshape(len(Cij), len(Cij))
print(b, "\nGraph ======\n", D, "\nResolution======\n", res.x)

vs(b, D, Cij)  # Визуализация доставки товаров,  пропускной способности, стоимости перевозки
