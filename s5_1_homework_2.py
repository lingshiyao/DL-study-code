from mxnet import autograd, nd
from mxnet.gluon import nn

# (样本, 通道, 高, 宽)

# - **data**: *(batch_size, channel, height, width)*
# - **weight**: *(num_filter, channel, kernel[0], kernel[1])*
# - **bias**: *(num_filter,)*
# - **out**: *(batch_size, num_filter, out_height, out_width)*.

def corr2d(X, K):
    h, w = K.shape
    Y = nd.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            # ----------------------------------------------------------------------
            # corr2d因为用了[i,j]=导致自动求导失败，这是由于autograd目前还有局限性。
            # ----------------------------------------------------------------------
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y


class Conv2D_ex3(nn.Block):
    def __init__(self, channels,  kernel_size, **kwargs):
        super(Conv2D_ex3, self).__init__(**kwargs)
        self.weight = self.params.get(
            'weight', shape=(channels, 1,) + kernel_size)
        self.bias = self.params.get('bias', shape=(channels,))
        self.num_filter = channels
        self.kernel_size = kernel_size

    def forward(self, x):
        # return corr2d(x, self.weight.data())
        return nd.Convolution(data=x, weight=self.weight.data(), bias=self.bias.data(), num_filter=self.num_filter, kernel=self.kernel_size)

X = nd.ones((6, 8))
X[:, 2:6] = 0
K = nd.array([[1, -1]])
Y = corr2d(X, K)

net = nn.Sequential()
net.add(Conv2D_ex3(1, kernel_size=(1, 2)))
net.initialize()

X = X.reshape((1, 1, 6, 8))
Y = Y.reshape((1, 1, 6, 7))

for i in range(10):
    with autograd.record():
        Y_hat = net(X)
        l = (Y_hat - Y) ** 2
    l.backward()
    net[0].weight.data()[:] -= 3e-2 * net[0].weight.grad()
    if (i + 1) % 2 == 0:
        print('batch %d, loss %.3f' % (i + 1, l.sum().asscalar()))

print(net[0].weight)
print(net[0].weight.data())