import d2lzh as d2l
from mxnet import nd

# -------------------zip-------------------
# a = [1, 2, 3]
# b = [4, 5, 6]
# c = [4, 5, 6, 7, 8]
# z1 = zip(a, b)     # 打包为元组的列表
# print(z1)
#
# for a1, b1 in z1:
#     print(a1, b1)
#
# for a1, b1 in zip(a, b):
#     print(a1, b1)
#
#
# def TTT(a1, b1):
#     print(a1, b1)
#
# z2 = zip(a, c)              # 元素个数与最短的列表一致
# print(z2)
# z3 = zip(*z1)          # 与 zip 相反，*zipped 可理解为解压，返回二维矩阵式
# print(z3)
# -------------------zip-------------------

# -------------------generator-------------------
# L = [x * x for x in range(10)]
# # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
# print(L)
#
# g = (x * x for x in range(10))
# # <generator object <genexpr> at 0x00000144F7EF4D00>
# print(g)
# # 0
# print(g.__next__())
# # 1
# print(g.__next__())
# # 4
# print(g.__next__())
# # 9
# print(g.__next__())
# # 16
# print(g.__next__())
# # 25
# print(g.__next__())
# -------------------generator-------------------

# -------------------*param-------------------
# 带一个星号（*）参数的函数传入的参数存储为一个元组（tuple）
# 带两个星号（**）  表示字典（dict）
# -------------------*param-------------------

def corr2d_multi_in(X, K):
    # 首先沿着X和K的第0维（通道维）遍历。然后使用*将结果列表变成add_n函数的位置参数
    # （positional argument）来进行相加

    #
    # [d2l.corr2d(x, k) for x, k in zip(X, K)]
    # [0]
    # [[19. 25.]
    # [37. 43.]]

    # [1]
    # [[37. 47.]
    # [67. 77.]]

    # [0] + [1] =
    # 56    72
    # 104   120
    return nd.add_n(*[d2l.corr2d(x, k) for x, k in zip(X, K)])

# 2 x 3 x 3
X = nd.array([[[0, 1, 2], [3, 4, 5], [6, 7, 8]],
              [[1, 2, 3], [4, 5, 6], [7, 8, 9]]])

# 2 x 2 x 2
K = nd.array([[[0, 1], [2, 3]], [[1, 2], [3, 4]]])

# [[ 56.  72.]
#  [104. 120.]]
# <NDArray 2x2 @cpu(0)>
print(corr2d_multi_in(X, K))

def corr2d_multi_in_out(X, K):
    # 对K的第0维遍历，每次同输入X做互相关计算。所有结果使用stack函数合并在一起
    return nd.stack(*[corr2d_multi_in(X, k) for k in K])

# 3 x 2 x 2 x 2
#
# [[[[0. 1.]
#   [2. 3.]]

#  [[1. 2.]
#   [3. 4.]]]


# [[[1. 2.]
#   [3. 4.]]

#  [[2. 3.]
#   [4. 5.]]]


# [[[2. 3.]
#   [4. 5.]]

#  [[3. 4.]
#   [5. 6.]]]]

# 2 x 2 x 2
# 2 x 2 x 2
# 2 x 2 x 2
# 结果是 ===> 3 x 2 x 2 x 2
K = nd.stack(K, K + 1, K + 2)

# (3, 2, 2, 2)
print(K.shape)

# 数组X与核数组K做
# [[[ 56.  72.]
#   [104. 120.]]
#
#  [[ 76. 100.]
#   [148. 172.]]
#
#  [[ 96. 128.]
#   [192. 224.]]]
# 三个通道的最终结果
# X 2x3x3
# K 3x2x2x2
TmpR = corr2d_multi_in_out(X, K)

def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]

    # 3 x 9
    X = X.reshape((c_i, h * w))

    # 2 x 3
    K = K.reshape((c_o, c_i))

    # 2 x 9
    Y = nd.dot(K, X)  # 全连接层的矩阵乘法

    # 2 x 3 x 3
    return Y.reshape((c_o, h, w))

# 三通道 3x3
X = nd.random.uniform(shape=(3, 3, 3))

# 输入3通道，输出2通道，1x1卷积核
K = nd.random.uniform(shape=(2, 3, 1, 1))

Y1 = corr2d_multi_in_out_1x1(X, K)
Y2 = corr2d_multi_in_out(X, K)

(Y1 - Y2).norm().asscalar() < 1e-6