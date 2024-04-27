import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import networkx as nx
import math

K = int(input("Ведите количество типов товаров: "))  # количество типов товаров
L = int(input("Введите количество типов сырья: "))
M = int(input("Введите количество дней: "))
MM  = []
for i in range(M):
    MM.append(i)

N = 12
Pkm = np.random.randint(1, 10, size=(K, M)) #цена реализации типа k в день m
Ylm = np.random.randint(30, 500, size=(L, M)) #объем сырья l в день m
Alk = np.random.randint(1, 10, size=(K, L)) #объем сырья типа l для производства товара типа k
Qk = np.random.randint(50, 70, size=(K)) #спрос товаров типа k

D = np.array([[0, 90, 60, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 50, 40, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 30,  30, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0,40, 40, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0,  70, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0,  0,30,10, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 110, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 15, 15, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 40, 70, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 45],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 85],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])  #<-матрица пропускной способности
Cij = np.array([[0, 2, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 7, 3, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 4, 9, 0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 8, 5, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 4, 5, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 3, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 0],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2],
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]) #<-матрица стоимости перевозок
# res = [x11,lyam11, z11, b11]
# 𝑥𝑘𝑚 – объем производства товара типа 𝑘 в день 𝑚;
# 𝜆𝑖𝑗 – факт перевозки товаров из пункта 𝑖 в 𝑗;
# 𝑧𝑖𝑗 – объем перевозки товаров из пункта 𝑖 в 𝑗;
# 𝑏𝑙𝑚 – запас сырья типа 𝑙 в день 𝑚;

C1 = list(-np.reshape(Pkm, K*M))
C2 = list(np.reshape(Cij, N*N))
C34 = [0]*(N*N + L*M)
C = []
C.extend(C1)
C.extend(C2)
C.extend(C34)
print("Целевая функция:\n", C)

A_eq = []

#3 ограничение
A_eq3 = [1]*(K*M) + [0]*(N**2) + ([0] + [-1]*(N-1) + [0]*(N*(N-1))) + [0]*(L*M)
A_eq.append(A_eq3)

# 4 ограничение
A_eq4 = []
for i in range(N):
    A_eq4al = [0] * (K*M + N*N)
    A_eq4b = [0]*(L*M)
    A_eq4z = [0]*(N**2)
    for j in range(N):
        if D[i][j] != 0:
            A_eq4z[i*N + j] = 1
            A_eq4z[j*N + i] = -1
    A_eq4.append(A_eq4al+A_eq4z+A_eq4b)
A_eq.extend(A_eq4[1:N-1])

#11 ограничение
A_eq11 = [1]*(K*M) + [0]*(N**2) + ([0]*(N-1)+[-1])*N + [0]*(L*M)
A_eq.append(A_eq11)

# 12 ограничение
A_eq12 = []
a2 = [0] * (N**2)  # лямбды
a3 = [0] * (N**2)  # зет
for l in range(L):
    for m in MM[1:]:
        a1 = [1] * (K*M) #x
        a4 = [0] * (L*M) #b
        for k in range(K):
            a1[k*M+m] = Alk[l][k]
            a4[l*M+m] = 1
            a4[l*M+(m-1)] = -1
        A_eq12.append(a1 + a2 + a3 + a4)
A_eq.extend(A_eq12)

print("Матрица A_eq:\n")
for i in range(len(A_eq)):
    print(A_eq[i])

b_eq = [0]+[0]*(N-2)+[0]
b1 = np.reshape(Ylm, L*M)
b1 = b1[:L*(M-1)]
b_eq.extend(list(b1))
print("Матрица b_eq:\n", b_eq)

A_ub = []
#2
for l in range(L):
    aa = [0] * (K*M)
    bb = [0] * (L*M)
    for m in range(M):
        for k in range(K):
            aa[k*M + m] = Alk[l, k]
            bb[l*M + m] = -1
    A_ub.append(aa + [0]*(2*(N**2)) + bb)

#7
for i in range(N):
    for j in range(N):
        aoq = [0]*(N**2)
        aoq1 = [0] * (N**2)
        aoq[i*N + j] = -100000
        aoq1[i*N + j] = -1
        A_ub.append([0]*(K*M) + aoq + aoq1 + [0]*(L*M))
#5
for k in range(K):
    ah = [0] * (K*M)
    for m in range(M):
        ah[k*M + m] = 1
    A_ub.append(ah + [0]*(2*(N**2)) + [0]*(L*M))

print("Матрица bA_ub:\n")
for i in range(len(A_ub)):
    print(A_ub)

b_ub = [0]*(L) +[0]*(N**2)
b_ub.extend(list(Qk))
print("Матрица b_ub:\n", b_ub)

res = linprog(c=C, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, integrality=(M*K+2*N*N + M*L))
r = res.x
print(r, '\n', len(r))

# res = [x(K*M),lyam(N**2), z(N**2), b(M*L)]


#для транспортного графа
zij1=[0]*(N*N)
for i in range(N*N):
    zij1[i] = r[K*M+N*N+i]
zij = np.array(zij1)
zij = zij.reshape(N, N)
print(zij)

#рисование исходного графа
def GraphDraw(x, DD):
    G = nx.DiGraph()
    for i in range(x):
        G.add_node(i)
        for j in range(x):
            if (DD[i, j] > 0):
                G.add_edges_from([(i, j)])
                G.add_edge(i, j, weight=DD[i][j])
    pos = nx.circular_layout(G)
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, with_labels=True)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.show()
    return G
G = GraphDraw(N, D)
G1 = GraphDraw(N, Cij)

pos = nx.circular_layout(G)
plt.figure(figsize=(8, 8))
nx.draw(G1, pos, with_labels=True)
labels = nx.get_edge_attributes(G1, 'weight')
nx.draw_networkx_edge_labels(G1, pos, edge_labels=labels)
plt.show()

#график 4.3

blm1 = [1]*(L*M)
for i in range(L*M):
    blm1[i]= r[(K*M+2*N*N)+i]
blm = np.array(blm1)
blm = blm.reshape(L, M)

plt.figure(figsize=(10, 10))
for j in range(M):
    plt.plot(blm[:, j], label=f'Type {j + 1}')
plt.xlabel('День')
plt.ylabel('Общий запас сырья на складе')
plt.title('График объемов сырья на складе на каждый день по каждому типу сырья')
plt.legend()
plt.show()

#сумма, 4.4
plt.figure(figsize=(10, 10))
plt.plot(np.sum(blm, axis=1))
plt.xlabel('День')
plt.ylabel('Общий запас материалов')
plt.title('График суммарных объемов сырья на складе')
plt.show()

xkm1 = [1]*(K*M)
for i in range(K*M):
    xkm1[i] = r[i]
xkm = np.array(xkm1)
xkm = xkm.reshape(K, M)

plt.figure(figsize=(10, 10))
for j in range(M):
    plt.plot(xkm[:, j], label=f'Type {j + 1}')
plt.xlabel('День')
plt.ylabel('Общий объем производства на складе')
plt.title('График объемов производства каждый день по каждому типу')
plt.legend()
plt.show()

plt.figure(figsize=(10, 10))
plt.plot(np.sum(xkm, axis=1))
plt.xlabel('День')
plt.ylabel('Общий объем производства на складе')
plt.title('Суммарный график объемов производства каждый день по каждому типу')
plt.show()

