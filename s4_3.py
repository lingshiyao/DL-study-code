from mxnet import init, nd
from mxnet.gluon import nn

class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print('Init', name, data.shape)
        # 实际的初始化逻辑在此省略了

net = nn.Sequential()
net.add(nn.Dense(256, activation='relu'),
        nn.Dense(10))

net.initialize(init=MyInit())

X = nd.random.uniform(shape=(2, 20))

# 调用 MyInit._init_weight
Y = net(X)

# 不会调用 MyInit._init_weight
Y = net(X)

# 调用 MyInit._init_weight
# force_reinit = True
net.initialize(init=MyInit(), force_reinit=True)

net = nn.Sequential()
net.add(nn.Dense(256, in_units=20, activation='relu'))
net.add(nn.Dense(10, in_units=256))

# 调用 MyInit._init_weight
net.initialize(init=MyInit())