import d2lzh as d2l
from mxnet import gluon, init
from mxnet.gluon import loss as gloss, nn

batch_size = 256

# 获取训练数据和测试数据
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 获取模型
net = nn.Sequential()

# 输出层数为10
net.add(nn.Dense(10))

# 默认的数据为0.0正态分布
net.initialize(init.Normal(sigma=0.01))

# loss为softmax的交叉熵损失
loss = gloss.SoftmaxCrossEntropyLoss()

# sgd梯度下降，学习率0.1
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})

# 5轮
num_epochs = 5

# 开始学习
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None,
              None, trainer)
