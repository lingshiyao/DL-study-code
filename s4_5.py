from mxnet import nd
from mxnet.gluon import nn

x = nd.ones(3)

# 将x存在名字为x的文件中
nd.save('x', x)

# 从x的文件读取数据
x2 = nd.load('x')
print(x2)

y = nd.zeros(4)
# 将[x, y]存在名字为xy的文件中
nd.save('xy', [x, y])

# 从xy的文件中读取数据
x2, y2 = nd.load('xy')
print(x2, y2)

mydict = {'x': x, 'y': y}

# 存储dict
nd.save('mydict', mydict)

# 读取dict
mydict2 = nd.load('mydict')
print(mydict2)

class MLP(nn.Block):
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')
        self.output = nn.Dense(10)

    def forward(self, x):
        return self.output(self.hidden(x))

net = MLP()
net.initialize()
X = nd.random.uniform(shape=(2, 20))
Y = net(X)

filename = 'mlp.params'
# 把模型存下来
net.save_parameters(filename)

net2 = MLP()
# 读取之前保存的模型
net2.load_parameters(filename)

Y2 = net2(X)
print(Y2 == Y)