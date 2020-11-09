from mxnet import nd
from mxnet.gluon import nn

#
def pool2d(X, pool_size, mode='max'):
    # 2, 2
    p_h, p_w = pool_size

    Y = nd.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            if mode == 'max':
                Y[i, j] = X[i: i + p_h, j: j + p_w].max()
            elif mode == 'avg':
                Y[i, j] = X[i: i + p_h, j: j + p_w].mean()
    return Y

X = nd.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
print(pool2d(X, (2, 2)))

print(pool2d(X, (2, 2), 'avg'))

X = nd.arange(16).reshape((1, 1, 4, 4))
print(X)

pool2d = nn.MaxPool2D(3)
pool2d(X)  # 因为池化层没有模型参数，所以不需要调用参数初始化函数

pool2d = nn.MaxPool2D(3, padding=1, strides=2)
print(pool2d(X))

pool2d = nn.MaxPool2D((2, 3), padding=(1, 2), strides=(2, 3))
print(pool2d(X))

X = nd.concat(X, X + 1, dim=1)
print(X)

pool2d = nn.MaxPool2D(3, padding=1, strides=2)
print(pool2d(X))