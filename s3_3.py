from mxnet import autograd, nd

# 训练数据为长度2的向量
num_inputs = 2

# 训练数据1000个
num_examples = 1000

# 设置初始权重
true_w = [2, -3.4]

# 设置b的值
true_b = 4.2

# 生成训练数据，均值为0，标准差为1的矩阵 (1000, 2)
# 1000 x 2
# ---训练特征数据---
#
# ------训练出来的带有特征的数据------
features = nd.random.normal(scale = 1, shape = (num_examples, num_inputs))

# 求𝑿𝒘+𝑏      1000 * 2  2 * 1  b
# [2] * (1000) + [-3.4] * (1000)
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b

# 加上噪声
labels += nd.random.normal(scale = 0.01, shape = labels.shape)

from mxnet.gluon import data as gdata

# Gluon提供了data包来读取数据

# 批次大小为10
batch_size = 10

# 将训练数据的特征和标签组合(线性回归么，首先把x，y的值设置进去)，生成一个dataset对象
dataset = gdata.ArrayDataset(features, labels)

# 随机读取小批量，传入数据对象，传入批次
# 参考之前的代码，之前也有这个函数，只不过是我自己写的
data_iter = gdata.DataLoader(dataset, batch_size, shuffle = True)

# 打印一批来看下，打印X，y
for X, y in data_iter:
    print(X, y)
    break

# 导入nn模块，nn指的是neural networks（神经网络）的缩写
# 该模块定义了大量神经网络的层
from mxnet.gluon import nn

# 定义一个模型变量net
# 之前的net实际上是用来做计算的
# 𝒚=𝑿𝒘+𝑏
net = nn.Sequential()

# 单层神经网络，层数为1
net.add(nn.Dense(1))

# 导入init模块
from mxnet import init

# init.Normal(sigma=0.01) 均值为0，标准差为0.01的正态分布
net.initialize(init.Normal(sigma=0.01))

# 导入loss模块，别名为gloss，使用他的平方损失作为模型loss函数
from mxnet.gluon import loss as gloss

# 平方损失又称L2范数损失
# 之前的loss是自己写的
loss = gloss.L2Loss()

from mxnet import gluon

# 定义一个训练助理
# 当前用sgd，梯度下降，学习率为0.03
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})

# --------------------------
# 开始训练模型


# 模拟之前的训练三次
num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        with autograd.record():
            # 可以和之前的代码对比，之前的loss是自己写的
            l = loss(net(X), y)
        l.backward()

        # 训练器开始step一次的梯度
        trainer.step(batch_size)

    l = loss(net(features), labels)
    print('epoch %d, loss: %f' % (epoch, l.mean().asnumpy()))

dense = net[0]
true_w, dense.weight.data()

true_b, dense.bias.data()