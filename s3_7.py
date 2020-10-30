import d2lzh as d2l
from mxnet import gluon, init
from mxnet.gluon import loss as gloss, nn

# 批次大小每一批训练的大小为256
batch_size = 256

# 获取训练数据和测试数据
# （train iter训练项目）
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 获取模型
net = nn.Sequential()

# 设置模型的输出类别有10类
net.add(nn.Dense(10))

# 设置训练初始数据为标准差为0.01的正态分布
net.initialize(init.Normal(sigma=0.01))

# 设置loss函数 lose为softmax后求熵
loss = gloss.SoftmaxCrossEntropyLoss()

# sgd梯度下降，学习率0.1
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.1})

# 学习5轮
num_epochs = 5

# 开始学习
# net           模型
# train_iter    训练项目
# test_iter     测试项目
# loss          损失
# num_epochs    学习轮数（当前为5）
# batch_size    批次（当前一批256）
# trainer       训练器（梯度下降sgd的训练器，学习率0.1）
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None,
              None, trainer)

