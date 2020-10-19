from IPython import display
from matplotlib import pyplot as plt
from mxnet import autograd, nd
import random

######################################################
# 生成测试数据
######################################################

# 𝒚=𝑿𝒘+𝑏+𝜖

# 训练数据为长度2的向量
num_inputs = 2

# 训练数据1000个
num_examples = 1000

# 设置初始权重
true_w = [2, -3.4]

# 设置b的值
true_b = 4.2

# 生成训练数据，均值为0，标准差为1的矩阵 (1000, 2)
features = nd.random.normal(scale = 1, shape = (num_examples, num_inputs))

# features[:, 0]把矩阵裁剪成向量
# features[:, 1]把矩阵裁剪成向量
# 求𝑿𝒘+𝑏      1000 * 2  2 * 1  b
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b

# 加上噪声，均值为0，标准差为0.01，最终求出y，注意此处噪声是服从正态分布的向量
# labels += nd.random.normal(scale = 0.01, shape = labels.shape)

# 打印训练数据和y的第一个值
print('-' * 20, "打印训练数据和y的第一个值")
print(features[0], labels[0])

# 设置打印数据操作
def use_svg_display():
    # 用矢量图显示
    display.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    # 设置图的尺寸
    plt.rcParams['figure.figsize'] = figsize

set_figsize()

# 打印训练数据
# features[:, 1].asnumpy()可以把NDArray对象转换成数组
# 第一个参数是x轴的向量，第二个参数是y轴的向量
# x是训练数据的值，y是输出结果的值，然后可以根据公式求出线性回归，也可以根据梯度算出线性回归
plt.scatter(features[:, 1].asnumpy(), labels.asnumpy(), 1)

# 打印训练数据
# plt.show()

######################################################
# 读取数据集合
######################################################

def data_iter(batch_size, features, labels):
    # 获取训练数据的长度，当前长度为1000
    num_examples = len(features)

    # 根据长度生成一个下标index的数组，数组长度为1000
    indices = list(range(num_examples))

    # 把数组的顺序打乱
    random.shuffle(indices)

    for i in range(0, num_examples, batch_size):
        # 当前的batch_size为10，相当于每一批选择10个
        #
        # indices[i: min(i + batch_size, num_examples)]
        # 首先是切片操作取出10个
        # 然后转换成NDArray对象
        j = nd.array(indices[i: min(i + batch_size, num_examples)])

        # 根据（数组 or 向量）j的索引取出对应的值
        yield features.take(j), labels.take(j)

# 当前训练的批次为10个一批
batch_size = 10

# for X, y in data_iter(batch_size, features, labels):
#     # 打印10个一批的训练数据？
#     print(X, y)
#     break

######################################################
# 初始化模型参数
######################################################

# 随机生成符合正态分布的权重向量
w = nd.random.normal(scale = 0.01, shape = (num_inputs, 1))

# 随机生成符合正态分布的偏置数值
b = nd.zeros(shape = (1,))

# 生成梯度
w.attach_grad()

# 默认的梯度为0
print("w.grad:", w.grad)

# 生成梯度
b.attach_grad()

# 默认梯度为0
print("b.grad:", b.grad)

# 输入训练数据，然后输出算出的结果
# X:入参矩阵  当前为 10 x 2
# w:权重向量  当前为 2 x 1
# b:偏置（此处用了广播）
# 结果是 10 x 1
def linreg(X, w, b):
    return nd.dot(X, w) + b

# 计算loss函数，实际上是计算误差C
# y_hat 计算出来的数值 10 x 1
# y 真实结果
# 返回的损失 10 x 1
def squared_loss(y_hat, y):  # 本函数已保存在d2lzh包中方便以后使用
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

# 梯度下降
# params = [w, b]
# lr 当前为0.03
# batch_size 当前为10
def sgd(params, lr, batch_size):  # 本函数已保存在d2lzh包中方便以后使用
    for param in params:
        param[:] = param - lr * param.grad / batch_size

# 梯度下降率
lr = 0.03

# 一共梯度下降3次
num_epochs = 3

net = linreg

loss = squared_loss

for epoch in range(num_epochs):  # 训练模型一共需要num_epochs个迭代周期
    # 在每一个迭代周期中，会使用训练数据集中所有样本一次（假设样本数能够被批量大小整除）。X
    # 和y分别是小批量样本的特征和标签
    for X, y in data_iter(batch_size, features, labels):
        # 首先从样本中选出10个样本

        with autograd.record():
            # 求出损失函数
            l = loss(net(X, w, b), y)
        # 求梯度，求w和b的导数，然后把数值代入进去
        # 小批量的损失对模型参数求梯度
        l.backward()

        # 使用小批量随机梯度下降迭代模型参数
        # [w, b]是NDArray数组，两个参数都是NDArray对象
        sgd([w, b], lr, batch_size)
    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().asnumpy()))