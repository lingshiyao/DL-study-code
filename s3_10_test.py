import d2lzh as d2l
from mxnet import gluon, init
from mxnet.gluon import loss as gloss, nn

# 获取模型
net = nn.Sequential()

# 设置两层
# 隐藏层256个，用relu激活函数
# 输出层10类
net.add(nn.Dense(256, activation='sigmoid'),
        nn.Dense(128, activation='sigmoid'),
        nn.Dense(64, activation='sigmoid'),
        # nn.Dense(32, activation='sigmoid'),
        # nn.Dense(16, activation='sigmoid'),
        nn.Dense(10))

# 初始化Wh bh Wo bo数据，为标准差为0.01的正态分布
net.initialize(init.Normal(sigma=0.01))

# 批次大小为256
batch_size = 256

# 装载训练项目
# 装载测试项目
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 设置损失函数
loss = gloss.SoftmaxCrossEntropyLoss()

# 设置训练教练
# sgd(梯度下降)，学习率为0.5
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.01})

# 设置轮数为5轮
num_epochs = 50

# 开始学习
# net           模型
# train_iter    训练数据
# test_iter     测试数据
# loss          损失函数
# num_epochs    迭代周期
# batch_size    批次大小
# trainer       训练器
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None,
              None, trainer)