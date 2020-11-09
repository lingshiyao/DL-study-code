from mxnet import gluon, nd
from mxnet.gluon import nn

# 定义一个将输入减掉均值后输出的层
class CenteredLayer(nn.Block):
    def __init__(self, **kwargs):
        super(CenteredLayer, self).__init__(**kwargs)

    def forward(self, x):
        return x - x.mean()

layer = CenteredLayer()

# [1 - 3. 2 - 3.  3 - 3.  4 - 3.  5 - 3.]
# [-2. -1.  0.  1.  2.]
print(layer(nd.array([1, 2, 3, 4, 5])))

# 把中间层放到模型第二层
net = nn.Sequential()
net.add(nn.Dense(128),
        CenteredLayer())

net.initialize()
y = net(nd.random.uniform(shape=(4, 8)))
print(y.mean().asscalar())

params = gluon.ParameterDict()
params.get('param2', shape=(2, 3))
print(params)

class MyDense(nn.Block):
    # units为该层的输出个数，in_units为该层的输入个数
    def __init__(self, units, in_units, **kwargs):
        super(MyDense, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=(in_units, units))
        self.bias = self.params.get('bias', shape=(units,))

    def forward(self, x):
        linear = nd.dot(x, self.weight.data()) + self.bias.data()
        return nd.relu(linear)

dense = MyDense(units=3, in_units=5)
print(dense.params)

dense.initialize()
dense(nd.random.uniform(shape=(2, 5)))

net = nn.Sequential()
net.add(MyDense(8, in_units=64),
        MyDense(1, in_units=8))
net.initialize()
net(nd.random.uniform(shape=(2, 64)))