import numpy as np
from scipy.optimize import linprog
import matplotlib.pyplot as plt
import networkx as nx
import math

K = int(input("Ведите количество типов товаров: "))
L = int(input("Введите количество типов сырья: "))
M = int(input("Введите количество дней: "))
M1 = int(input("Введите количество сырья в первый день: "))
Pkm = np.random.randint(1, 10, size=(K, M)) #цена реализации типа k в день m
Ylm = np.random.randint(10, 70, size=(L, M)) #объем сырья l в день m
Alk = np.random.randint(1, 5, size=(L, K)) #объем сырья типа l для производства товара типа k
Qk = np.random.randint(30, size=(K)) #спрос товаров типа k

              #0   1   2   3   4   5   6   7    8   9   10 11
D = np.array([[0, 90, 60,  0,  0,  0,  0,  0,   0,  0,  0,  0], #0
              [0,  0,  0, 50, 40,  0,  0,  0,   0,  0,  0,  0], #1
              [0,  0,  0, 30, 30,  0,  0,  0,   0,  0,  0,  0], #2
              [0,  0,  0,  0,  0, 40, 40,  0,   0,  0,  0,  0], #3
              [0,  0,  0,  0,  0,  0, 70,  0,   0,  0,  0,  0], #4
              [0,  0,  0,  0,  0,  0,  0, 30,  10,  0,  0,  0], #5
              [0,  0,  0,  0,  0,  0,  0,  0, 110,  0,  0,  0], #6
              [0,  0,  0,  0,  0,  0,  0,  0,   0, 15, 15,  0], #7
              [0,  0,  0,  0,  0,  0,  0,  0,   0, 40, 70,  0], #8
              [0,  0,  0,  0,  0,  0,  0,  0,   0,  0,  0, 45], #9
              [0,  0,  0,  0,  0,  0,  0,  0,   0,  0,  0, 85], #10
              [0,  0,  0,  0,  0,  0,  0,  0,   0,  0,  0,  0],])  #11 #<-матрица пропускной способности
                # 0  1  2   3 4 5   6  7  8  9  10 11
Cij = np.array([[0, 2, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0],  #0
                [0, 0, 0, 7, 3, 0, 0, 0, 0, 0, 0, 0], #1
                [0, 0, 0, 4, 9, 0, 0, 0, 0, 0, 0, 0], #2
                [0, 0, 0, 0, 0, 8, 5, 0, 0, 0, 0, 0], #3
                [0, 0, 0, 0, 0, 0, 6, 0, 0, 0, 0, 0], #4
                [0, 0, 0, 0, 0, 0, 0, 4, 5, 0, 0, 0], #5
                [0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 0], #6
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 3, 0], #7
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 3, 0], #8
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4], #9
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2], #10
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])  #11 #<-матрица стоимости перевозок

N=len(D)

# res = [x11,lyam11, z11, b11]
# 𝑥𝑘𝑚 – объем производства товара типа 𝑘 в день 𝑚;
# 𝜆𝑖𝑗 – факт перевозки товаров из пункта 𝑖 в 𝑗;
# 𝑧𝑖𝑗 – объем перевозки товаров из пункта 𝑖 в 𝑗;
# 𝑏𝑙𝑚 – запас сырья типа 𝑙 в день 𝑚;

#целевая

C = []
c1 = list(-np.reshape(Pkm, K*M))
c2 = list(np.reshape(Cij, N*N))
c3 = [0]*(N*N + L*M)
C.extend(c1)
C.extend(c2)
C.extend(c3)

#матрица A_eq

A_eq = []

#3 ограничение
A_eq3 = [1]*(K*M) + [0]*(N**2) + ([-1]*(N) + [0]*(N*(N-1))) + [0]*(L*M)

# 4 ограничение
A_eq4 = []
for i in range(1,N-1):
    xl4 = [0] * (K*M + N*N) #x и лямбда
    b4 = [0]*(L*M) #b
    z4 = [0]*(N*N) #z
    for j in range(N):
        z4[i*N + j] = 1
        z4[j*N + i] = -1
    A_eq4.append(xl4 + z4 + b4)

#11 ограничение
A_eq11 = [-1]*(K*M) + [0]*(N**2) + ([0]*(N-1)+[1])*N + [0]*(L*M)

# 12 ограничение
A_eq12=[]
l12 = [0] * (N**2)  # lambda
z12 = [0] * (N**2)  # z
for l in range(L):
    for m in range(M):
        x12 = [0] * (K*M) #x
        b12 = [0] * (L*M) #b
        for k in range(K):
            x12[k*M+m] = Alk[l][k] #тут все ок
            if (m==0):
                b12[l*M+m] = M1
            else:
                b12[l*M+m] = 1
            b12[l*M+(m-1)] = -1
        A_eq12.append(x12 + l12 + z12 + b12)

#создание полной матрицы

A_eq.append(A_eq3)
A_eq.extend(A_eq4)
A_eq.append(A_eq11)
A_eq.extend(A_eq12)
print("Матрица A_eq:\n", len(A_eq))

# print("Матрица A_eq:\n")
# for i in range(len(A_eq)):
#     print(A_eq[i])

# создание матрицы b_eq
b_eq = [0] #(3)
b_eq += [0]*(N-2) #(4)
b_eq += [0] #(11)
b1 = np.reshape(Ylm, L*M) #(12)
b1[0] += M1
b_eq.extend(list(b1))
print("Матрица b_eq:\n", len(b_eq))

#создание матрицы A_ub

A_ub = []

#2
A_ub2 = []
for l in range(L):
    for m in range(M):
        x2 = [0] * (K*M)
        b2 = [0] * (L*M)
        for k in range(K):
            x2[k*M + m] = Alk[l][k]
            b2[l*M + m] = -1
        A_ub2.append(x2 + [0]*(2*(N**2)) + b2)

#7
A_ub7 = []
for i in range(N):
    for j in range(N):
        l7 = [0] * (N**2)
        z7 = [0] * (N**2)
        l7[i*N + j] = -10000000
        z7[i*N + j] = 1
        A_ub7.append([0]*(K*M) + l7 + z7 + [0]*(L*M))


#5
A_ub5 = []
for k in range(K):
    x5 = [0] * (K*M)
    for m in range(M):
        x5[k*M + m] = 1
    A_ub5.append(x5 + [0]*(2*(N**2)) + [0]*(L*M))

#создание полной матрицы A_ub

A_ub.extend(A_ub2)
A_ub.extend(A_ub7)
A_ub.extend(A_ub5)
print("Матрица b_eq:\n", len(A_ub))

#создание матрицы b_ub
b_ub = [0]*(L*M) #(2)
b_ub += [0]*(N**2) #(7)
b_ub.extend(list(Qk)) #(5)
print("Матрица b_eq:\n", len(b_ub))

lb = [0]*(K*M + (2*N*N) + L*M)
ub = list(Qk)*M + [1]*(N*N) + list(np.reshape(D, N*N)) + [math.inf]*(L*M)
bbb = list(zip(lb, ub))
xi = [1]*(K*M)+ [0]*(N*N)+ [0]*(N*N) + [0]*(L*M)
res = linprog(c=C, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bbb, integrality = xi)
r = res.x
print(r)

# res = [x(K*M),lyam(N**2), z(N**2), b(M*L)]


#для транспортного графа
zij1=[0]*(N*N)
for i in range(N*N):
    zij1[i] = (r[K*M+N*N+i])
zij = np.array(zij1)
zij = zij.reshape(N, N)
print(zij)

#рисование графа
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
    return G
#
# for i in range(N):
#     for j in range(N):
#         if zij[i, j]>0:
#             zij[0, 1] = zij[i, j]
#             zij[1, 4] = zij[i, j]
#             zij[4, 6] = zij[i, j]
#             zij[6, 8] = zij[i, j]
#             zij[8, 10] = zij[i, j]
#             zij[10, 11] = zij[i, j]

G = GraphDraw(N, D)
plt.show()
G1 = GraphDraw(N, zij)
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

