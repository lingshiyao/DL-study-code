from IPython import display
from matplotlib import pyplot as plt
from mxnet import autograd, nd
import random

# 特征数为2
num_inputs = 2

# 训练样本个数为1000
num_examples = 1000

# 这里相当于先定义真实的w，b，然后训练模型，训练完成后，训练的w和b会向真实的数值靠近
# 真实的w为2，-3.4
true_w = [2, -3.4]

# 真实的b为4.2
true_b = 4.2

# 生成数据（暂时训练和测试都用他）
# 1000 x 2
features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))

# w1 * x1 + w2 * x2 + b
# w1 * x1 --- 广播
# w2 * x2 --- 广播
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b

# 如果不加上Epsilon 算出来的结果都在一条直线，或者一个面，或者一个三维区域里面（根据参数数量，维度增加）
# 算出用来训练的真实的1000个结果数据
# ϵ ~ Epsilon
# labels += nd.random.normal(scale=0.01, shape=labels.shape)

#
# 加大噪音测试
# labels += nd.random.normal(scale=0.01, shape=labels.shape)

# 此时x是features
print(features[0])

# 此时y是labels
print(labels[0])

def use_svg_display():
    # 用矢量图显示
    display.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize

set_figsize()
plt.scatter(features[:, 1].asnumpy(), labels.asnumpy(), 1);  # 加分号只显示图

# 本函数已保存在d2lzh包中方便以后使用
def data_iter(batch_size, features, labels):
    num_examples = len(features)

    # indices   ---     长度1000
    # [0, 1, 2, 3, ......, 999]
    indices = list(range(num_examples))

    # 将indices变成随机排列
    random.shuffle(indices)  # 样本的读取顺序是随机的

    # for 0-1000， 每次递增10（batch_size的大小）
    for i in range(0, num_examples, batch_size):
        # indices[i: min(i + batch_size, num_examples)]
        # 此处是做截取操作，截取[i:i + 10]
        # 但是如果i是998
        # [998: 1008] 会下标越界， min(1008, 1000) = 1000能避免

        # 取出下标数组j
        j = nd.array(indices[i: min(i + batch_size, num_examples)])

        # 从features 里面取出下标的数据
        # 从labels 里面取出下标的数据
        # 这样就实现了取出一批训练数据x，y的功能
        yield features.take(j), labels.take(j)  # take函数根据索引返回对应元素

# 批次大小10
batch_size = 10

# batch_size    --      10
# features      --      x
# labels        --      y
for X, y in data_iter(batch_size, features, labels):
    # 打印第一批训练数据
    print(X, y)
    break

# 生成训练w，均值为0，标准差0.01的正态分布随机数
# 2 x 1
w = nd.random.normal(scale=0.01, shape=(num_inputs, 1))

# 初始的b的结果为0
# 1
b = nd.zeros(shape=(1,))

# 生成w的梯度
w.attach_grad()

# 生成b的梯度
b.attach_grad()

# 模型
# y = Xw + b
def linreg(X, w, b):  # 本函数已保存在d2lzh包中方便以后使用
    return nd.dot(X, w) + b

# 自定义损失
# (y - y_hat) ** 2 / 2
def squared_loss(y_hat, y):  # 本函数已保存在d2lzh包中方便以后使用
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

# 自定义的梯度下降
def sgd(params, lr, batch_size):  # 本函数已保存在d2lzh包中方便以后使用
    for param in params:
        param[:] = param - lr * param.grad / batch_size

# 学习率0.03
lr = 0.03

# 训练3轮
num_epochs = 3

# 配置模型
# 当前为很简单的Xw + b
net = linreg

# 配置损失函数
loss = squared_loss

for epoch in range(num_epochs):  # 训练模型一共需要num_epochs个迭代周期
    # 在每一个迭代周期中，会使用训练数据集中所有样本一次（假设样本数能够被批量大小整除）。X
    # 和y分别是小批量样本的特征和标签
    for X, y in data_iter(batch_size, features, labels):

        # X     ---     10  x 2
        # w     ---     2   x 1
        # b     ---     1
        # y     ---     10  x 1
        with autograd.record():

            # 先训练模型
            # 10 x 1
            tmpNet = net(X, w, b)

            # 然后求出损失
            # 10 x 1
            l = loss(tmpNet, y)  # l是有关小批量X和y的损失

        # 计算w，b的梯度
        # --------------------------------------------------
        # 此时求出的梯度，是10次梯度的总和
        # 所以后面梯度下降的时候，需要除以批次10
        # --------------------------------------------------
        l.backward()  # 小批量的损失对模型参数求梯度

        # 梯度下降
        sgd([w, b], lr, batch_size)  # 使用小批量随机梯度下降迭代模型参数

    # 训练完每一轮，用1000个数据来检测一次
    # 1000x1
    train_l = loss(net(features, w, b), labels)

    # 打印出损失的平均值
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().asnumpy()))

# 训练完成后打印，可以看出训练的模型和真实的比较接近了
# 但是此处真实模型里面加了随机噪音，所以训练的不可能和真实的一摸一样
print(true_w)
print(w)

print(true_b)
print(b)