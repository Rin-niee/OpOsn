import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog

#создание исходной матрицы +
               #1   #2  #3  #4  #5  #6   #7  #8    #9  #10 #11 #12  #13  #14  #15 #16  #17  #18
D = np.array([[0,   0,  0,  10, 11, 12,   0,  0,   0,  0,  0,  0,    0,   0,   0,   0,    0,   0], #1
              [0,   0,  0,  24, 25, 27,   0,  0,   0,  0,  0,  0,    0,   0,   0,   0,    0,   0],  #2
              [0,   0,  0,  29, 30, 31,   0,  0,   0,  0,  0,  0,    0,   0,   0,   0,    0,   0], #3
              [0,   0,  0,   0,  0,  0,  40, 42,  44,  0,  0,  0,    0,   0,   0,   0,    0,   0], #4
              [0,   0,  0,   0,  0,  0,  47, 49,  54,  0,  0,  0,    0,   0,   0,   0,    0,   0], #5
              [0,   0,  0,   0,  0,  0,  56, 57,  59,  0,  0,  0,    0,   0,   0,   0,    0,   0], #6
              [0,   0,  0,   0,  0,  0,   0,  0,   0, 70, 72, 75,    0,   0,   0,   0,    0,   0], #7
              [0,   0,  0,   0,  0,  0,   0,  0,   0, 76, 75, 74,    0,   0,   0,   0,    0,   0], #8
              [0,   0,  0,   0,  0,  0,   0,  0,   0, 81, 82, 83,    0,   0,   0,   0,    0,   0], #9
              [0,   0,  0,   0,  0,  0,   0,  0,   0,  0,  0,  0,    0,  80,   0,   0,    0,   0], #10
              [0,   0,  0,   0,  0,  0,   0,  0,   0,  0,  0,  0,    0,  90,   0,   0,    0,   0], #11
              [0,   0,  0,   0,  0,  0,   0,  0,   0,  0,  0,  0,    0,  95,   0,   0,    0,   0], #12
              [0,   0,  0,   0,  0,  0,   0,  0,   0,  0,  0,  0,    0,   0,   0, 121,  123, 125], #13
              [0,   0,  0,   0,  0,  0,   0,  0,   0,  0,  0,  0,   99,   0, 108,   0,    0,   0], #14
              [0,   0,  0,   0,  0,  0,   0,  0,   0,  0,  0,  0,    0,   0,   0, 250,  300, 290], #15
              [0,   0,  0,   0,  0,  0,   0,  0,   0,  0,  0,  0,    0,   0,   0,   0,    0,   0], #16
              [0,   0,  0,   0,  0,  0,   0,  0,   0,  0,  0,  0,    0,   0,   0,   0,    0,   0], #17
              [0,   0,  0,   0,  0,  0,   0,  0,   0,  0,  0,  0,    0,   0,   0,   0,    0,   0]]) #18
matrix_size = len(D)
print(matrix_size)

#матрица инцедентности +
D1 = np.zeros((matrix_size, matrix_size))
for i in range(matrix_size):
    for j in range(matrix_size):
        if D[i, j] != 0:
            D1[i, j] = 1
print("Incedent matrix\n", D1)
edges_num=0
for i in range(matrix_size):
    for j in range(matrix_size):
        if D1[i, j] != 0:
         edges_num=edges_num +1
print(edges_num)

#рисование графа +
G = nx.DiGraph()
for i in range(matrix_size):
    G.add_node(i)
    for j in range(matrix_size):
        if (D1[i, j] > 0):
            G.add_edges_from([(i, j)])
            G.add_edge(i, j, weight=D[i][j])
pos = nx.circular_layout(G)
plt.figure(figsize=(8, 8))
nx.draw(G, pos, with_labels=True)
labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
plt.show()

# Создание целевой функции +
c = np.zeros((edges_num))
m = -1
for i in range(matrix_size):
    for j in range(matrix_size):
        if (D1[i, j] > 0):
            m=m+1
            if ((j == 15 or j == 16 or j == 17) & (m < edges_num)):
                c[m] = 1
print("C function: \n", c)
#матрица A_eq
A_eq = np.zeros((matrix_size, edges_num)) #создает матрицу смежности, если есть из в i -> j, то 1, наоборот -1
e = 0
for i in range(matrix_size):
    for j in range(matrix_size):
            if (D1[i, j] > 0 & e < edges_num):
                A_eq[i, e] = 1
                A_eq[j, e] = -1
                e +=1
print("Matrix A_eq :\n", A_eq)
b_eq = np.zeros(A_eq.shape[0])

# b_ub = []
# for i in range(matrix_size):
#     for j in range(matrix_size):
#             if D1[i, j] == 1:
#                 b_ub.append(D[i][j])
# b_ub = np.asarray(b_ub)

b_ub = np.zeros((edges_num))
m = -1
for i in range(matrix_size):
    for j in range(matrix_size):
        if (D1[i, j] != 0):
            m=m+1
            if ((m < edges_num)):
                b_ub[m] = D[i, j]

print(b_ub)
A_ub = np.eye(b_ub.size)
# Решение задачи линейного программирования
res = linprog(c=-c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, method = 'HIGHS')
print(res)
sv = res.x

# # Вывод максимального потока
print(sv)
