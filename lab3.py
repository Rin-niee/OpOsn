
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import linprog

# 𝑐=[{−𝑝𝑘},𝑟𝑒𝑠ℎ𝑎𝑝𝑒(𝑐𝑖𝑗,1,𝑁2),𝑧𝑒𝑟𝑜𝑠(1,𝑁2)]1×(𝐾+2𝑁2)
# 𝐴𝑒𝑞=[𝐴𝑒𝑞(3);𝐴𝑒𝑞(4);𝐴𝑒𝑞(11)](1+𝑁+1)×(𝐾+2𝑁2)
#𝑏𝑒𝑞=𝑧𝑒𝑟𝑜𝑠(1,1+𝑁)
# 𝐴=[𝐴(2);𝐴(7)](𝐿+𝑁2)×(𝐾+2𝑁2) 𝐴(2)=[𝐴𝑙𝑘,𝑧𝑒𝑟𝑜𝑠(𝐿,2𝑁2)]𝐿×(𝐾+2𝑁2)
# 𝑏=𝑏𝑙
# 𝑏𝑜𝑢𝑛𝑑𝑠=[𝑧𝑒𝑟𝑜𝑠(1,(𝐾+2𝑁2));[{𝑄𝑘},𝑧𝑒𝑟𝑜𝑠(1,)]]

K = int(input("введите количество типов товаров"))  # количество типов товаров
L = int(input("введите количество типов сырья")) # количество типов сырья

pk = np.random.randint(100, size=(K)) # цена реализации товара типа k
bl = np.random.randint(1000, size=(L))  # запас сырья типа l

Alk = np.random.randint(20, size=(K, L))

dij = np.array([[0,  90, 60,   75,  0,  0,   0,  0,   0,  0,  0,  0,    0,   0,   0,   0,    0,   0], #1
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
matrix_size = len(dij)
N = 0
for i in range(len(dij)):
    for j in range(len(dij)):
        if dij[i, j]!=0:
            N+=1
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
D = matrix_incidence(dij, matrix_size)
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
D1 = GraphDraw(dij, D)





print("матрица инцидентности", D1)



#граф переделать
Qk = [50, 70]  # спрос на товар типа k

cij = np.array([[2, 3, 4, 5],
                [3, 2, 5, 4],
                [4, 5, 3, 2]])
