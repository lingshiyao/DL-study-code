from mxnet import autograd, nd

# 训练数据为长度2的向量
num_inputs = 2

# 训练数据1000个
num_examples = 1000

# 设置初始权重
true_w = [2, -3.4]

# 设置b的值
true_b = 4.2

# 生成X
# 1000 x 2
# 均值为0，标准差为1
features = nd.random.normal(scale = 1, shape = (num_examples, num_inputs))

# 生成y
# 1000
# y = w1 * 𝑿1 + w2 * X2 + 𝑏
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b

# ϵ ~ Epsilon
# 加上噪声
epsilon = nd.random.normal(scale = 0.01, shape = labels.shape)

labels += epsilon

# Gluon提供了data包来读取数据
from mxnet.gluon import data as gdata

# 批次大小为10
batch_size = 10

# 把features（X）和labels（y）放入dataset中
dataset = gdata.ArrayDataset(features, labels)

# dataset
# batch_size 10
data_iter = gdata.DataLoader(dataset, batch_size, shuffle = True)

# 打印一批来看下，打印X，y
for X, y in data_iter:
    # X     10 x 2
    # y     10
    # print(X, y)
    break

# 导入nn模块，(nn  ---  neural networks（神经网络）的缩写)
# 该模块定义了大量神经网络的层
from mxnet.gluon import nn

# sequential    线性序列
# 获取一个训练的模型
# 𝒚 = 𝑿𝒘 + 𝑏
net = nn.Sequential()

# 单层神经网络，层数为1
net.add(nn.Dense(1))

# 导入init模块
from mxnet import init

# 生成模型w初始值，均值为0，标准差为0.01的正态分布
net.initialize(init.Normal(sigma=0.01))

# 导入loss模块，别名为gloss，使用他的平方损失作为模型loss函数
from mxnet.gluon import loss as gloss

# 平方损失又称L2范数损失
# 参考后面的L2范数
loss = gloss.L2Loss()

from mxnet import gluon

# 定义一个训练助理
# sgd（梯度下降)，学习率为0.03
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})

# 训练三轮
num_epochs = 3
for epoch in range(1, num_epochs + 1):
    for X, y in data_iter:
        with autograd.record():
            l = loss(net(X), y)

        # 计算梯度
        l.backward()

        # 学习（梯度下降）
        trainer.step(batch_size)

    # 用当前的模型查看损失
    l = loss(net(features), labels)
    print('epoch %d, loss: %f' % (epoch, l.mean().asnumpy()))

# 对比真实的w和模型训练出来的w
print(true_w, net[0].weight.data())

# 对比真实的b和模型训练出来的b
print(true_b, net[0].bias.data())