from mxnet import init, nd
from mxnet.gluon import nn

net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()  # 使用默认初始化方式

# <NDArray 2x20 @cpu(0)>
# 此处的训练数据感觉很少很少，只有两个
# hidden: 2 x 20     20 x 256
# out:    2 x 256   256 x 10

X = nd.random.uniform(shape=(2, 20))
Y = net(X)  # 前向计算

print(net[0].params, type(net[0].params))

# net[0].weight的可读性更好
#
print(net[0].params['dense0_weight'], net[0].weight)

print("net[0].weight.data()", net[0].weight.data())

print("net[0].weight.grad()", net[0].weight.grad())

print("net[0].bias.data()", net[0].bias.data())

print("net[1].bias.data()", net[1].bias.data())

# 返回的是由参数名称到参数实例的字典
print("net.collect_params()", net.collect_params())

# 也可以用字典来返回需要的参数
print("net.collect_params('.*weight')", net.collect_params('.*weight'))

# 非首次对模型初始化需要指定force_reinit为真
# 第二此初始化的时候需要force_reinit=True，就是强制重新init
net.initialize(init=init.Normal(sigma=0.01), force_reinit=True)
print("net[0].weight.data()[0]", net[0].weight.data()[0])

# 使用常数初始化权重参数
net.initialize(init=init.Constant(1), force_reinit=True)
print("net[0].weight.data()[0]", net[0].weight.data()[0])

# Xavier初始化
net[0].weight.initialize(init=init.Xavier(), force_reinit=True)
print("net[0].weight.data()[0]", net[0].weight.data()[0])

# MyInit继承自init.Initializer
class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print('Init', name, data.shape)

        # 一半概率为0，一半为[-10, -5] [5, 10]之间的数字
        data[:] = nd.random.uniform(low=-10, high=10, shape=data.shape)
        data *= data.abs() >= 5

net.initialize(MyInit(), force_reinit=True)
print("net[0].weight.data()[0]", net[0].weight.data()[0])

# 直接把weight的数据加上1
net[0].weight.set_data(net[0].weight.data() + 1)
print("net[0].weight.data()[0]", net[0].weight.data()[0])

# 共享模型参数，让第二层和第三层共享模型参数
net = nn.Sequential()

# 定义一个网络神经层
# 让其中的两层共享模型
shared = nn.Dense(8, activation='relu')
net.add(nn.Dense(8, activation='relu'),
        shared,
        nn.Dense(8, activation='relu', params=shared.params),
        nn.Dense(10))
net.initialize()

X = nd.random.uniform(shape=(2, 20))
net(X)

print("net[1].weight.data()[0] == net[2].weight.data()[0]", net[1].weight.data()[0] == net[2].weight.data()[0])

print("net[0].weight.data()[0]", net[0].weight.data()[0])
print("net[1].weight.data()[0]", net[1].weight.data()[0])
print("net[2].weight.data()[0]", net[2].weight.data()[0])
print("net[3].weight.data()[0]", net[3].weight.data()[0])