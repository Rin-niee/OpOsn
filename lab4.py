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
              [0, 0, 0, 0, 0,54, 70, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0,  1, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0,  0,90,78, 0, 0, 0],
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
def A_eqcreate(K, M, N, L, Alk):
    A_eq = []
    #3, 4, 11, 12
    # ∑𝑥𝑘𝑚𝑘,𝑚=∑𝑧1𝑗𝑗 (3)
    # ∑𝑧𝑖𝑗𝑗=∑𝑧𝑗𝑖𝑗 (4)
    # ∑𝑧𝑖𝑁𝑖=∑𝑥𝑘𝑚𝑘,𝑚 (11)
    # 𝑏𝑙(𝑚+1)=𝑏𝑙𝑚−∑𝐴𝑙𝑘𝑥𝑘𝑚𝑘+𝛾𝑙𝑚 (12)
    A_eq3 = [1]*(K*M) + [0]*(N**2) + [-1]*(N) + [0]*(N*(N-1)) + [0]*(L*M) #все верно
    A_eq.append(A_eq3)
    for i in range(N):
        for j in range(N):
            A_eq41 = [0] * (K * M + 2 * N ** 2)
            if D[i][j] != 0:
                A_eq41[K*M + (N**2) + i*N+j] = 1
                A_eq41[K*M + (N**2) + j*N + i] = -1
            A_eq41.extend([0] * (L * M))
        if (i!=0 and i!=N):
            A_eq.append(A_eq41)
    A_eq11 = [1]*(K*M) + [0]*(N**2) + ([0]*(N-1)+[-1])*N + [0]*(L*M)
    A_eq.append(A_eq11)
    #тут должно быть 12 условие, но его украли цыгане, извините
    A_eq12 = []
    a1 = [1]*(K*M)
    a2 = [0]*(N**2) #лямбды
    a3 = [0]*(N**2) #зет
    a4 =[0]*(L*M)
    for l in range(L): #заполняем единичную матрицу Alk c k
        for m in range(M-1):
            for k in range(K):
                a1[k*M+m] = Alk[l][k]
                a4[l*M+m] = -1
                a4[l*M+(m-1)] = 1
    A_eq12.extend(a1)
    A_eq12.extend(a2)
    A_eq12.extend(a3)
    A_eq12.extend(a4)
    A_eq.append(A_eq12)
    return A_eq
def b_eqcreate(N, Ylm, L, M):
    b_eq = []
    b_eq+=[0]
    b_eq+=[0]*(N-2)
    b_eq+=[0]
    b1 = np.reshape(Ylm, L*M)
    b1 = b1[0:M]
    b_eq.extend(list(b1))
    return b_eq
A_eq = A_eqcreate(K, M, N, L, Alk)
b_eq = b_eqcreate(N, Ylm, L, M)

print(len(A_eq), len(b_eq))
A_ub = []
#2
for l in range(L):
    for m in range(M):
        aa = [0] * (K * M)
        bb = [0] * (L * M)
        for k in range(K):
            aa[k * M + m] = Alk[l][k]
            bb[l*M+m] = -1
        A_ub.append(aa + [0]*(2*(N**2)) + bb)
#7
for i in range(N):
    for j in range(N):
        aoq = [0] * (K*M + 2 * N**2 + M*L)
        # A_ub.append([0] * (K*M + 2 * N ** 2 + M*L))
        aoq[K*M + i * N + j] = -1000
        aoq[K*M + N ** 2 + i * N + j] = 1
        A_ub.append(aoq)
#5
for k in range(K):
    aaaaa = [0] * (K * M)
    for m in range(M):
        aaaaa[k*M + m] = 1
    A_ub.append(aaaaa + [0]*(2*(N**2)) + [0]*(L*M))
def b_ubcreate(N, L, Qk):
    b_ub=[]
    b_ub.extend([0]*(L*M))
    b_ub.extend([0] * (N ** 2))
    b_ub.extend(list(Qk))
    return b_ub
b_ub = b_ubcreate(N, L, Qk)
print(b_ub)
# bounds_lower = np.concatenate((np.zeros(K), np.zeros(L), np.zeros(L)))
# bounds_upper = np.concatenate((Qk, np.inf * np.ones(L), np.inf * np.ones(L)))
res = linprog(c=C, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq)
print(res)
