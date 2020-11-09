from mxnet import autograd, nd
from mxnet.gluon import nn

def corr2d(X, K):  # 本函数已保存在d2lzh包中方便以后使用
    # K.shape (2, 2)
    # h --- 2
    # w --- 2
    h, w = K.shape

    # 算出输出的形状
    Y = nd.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))

    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y

X = nd.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])

# kernel
K = nd.array([[0, 1], [2, 3]])

# --------------------------------
# 0     1       2
# 3     4       5
# 6     7       8
# --------------------------------
# 0     1
# 2     3
# --------------------------------
#  1 + 6 + 12 = 19       2 + 8 + 15 = 25
# 4 + 12 + 21 = 37      5 + 14 + 24 = 43
print(corr2d(X, K))

class Conv2D(nn.Block):
    def __init__(self, kernel_size, **kwargs):
        super(Conv2D, self).__init__(**kwargs)

        # 在模型里面声明w
        self.weight = self.params.get('weight', shape=kernel_size)

        # 在模型里面声明b
        self.bias = self.params.get('bias', shape=(1,))

    def forward(self, x):
        return corr2d(x, self.weight.data()) + self.bias.data()

X = nd.ones((6, 8))
X[:, 2:6] = 0
# [[1. 1. 0. 0. 0. 0. 1. 1.]
#  [1. 1. 0. 0. 0. 0. 1. 1.]
#  [1. 1. 0. 0. 0. 0. 1. 1.]
#  [1. 1. 0. 0. 0. 0. 1. 1.]
#  [1. 1. 0. 0. 0. 0. 1. 1.]
#  [1. 1. 0. 0. 0. 0. 1. 1.]]
# <NDArray 6x8 @cpu(0)>
print(X)

K = nd.array([[1, -1]])
# [[ 1. -1.]]
# <NDArray 1x2 @cpu(0)>
print(K)

Y = corr2d(X, K)
print(Y)

# 构造一个输出通道数为1（将在“多输入通道和多输出通道”一节介绍通道），核数组形状是(1, 2)的二
# 维卷积层
conv2d = nn.Conv2D(1, kernel_size=(1, 2))
conv2d.initialize()

# 二维卷积层使用4维输入输出，格式为(样本, 通道, 高, 宽)，这里批量大小（批量中的样本数）和通
# 道数均为1
X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))

for i in range(10):
    with autograd.record():
        # Y_hat     <NDArray 1x1x6x7 @cpu(0)>
        # Y         <NDArray 1x1x6x7 @cpu(0)>
        Y_hat = conv2d(X)
        l = (Y_hat - Y) ** 2

    # <NDArray 1x1x6x7 @cpu(0)>
    l.backward()
    # 简单起见，这里忽略了偏差
    # 0.03是学习率
    conv2d.weight.data()[:] -= 3e-2 * conv2d.weight.grad()
    if (i + 1) % 2 == 0:
        print('batch %d, loss %.3f' % (i + 1, l.sum().asscalar()))

# [[ 0.9895    -0.9873705]]
# <NDArray 1x2 @cpu(0)>
# 此处为学习到的核数组
print(conv2d.weight.data().reshape((1, 2)))

# 3小数点后2位
print(3e-2)

# 3.23412小数点后4位
print(3.23412e-4)