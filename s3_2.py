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
# 2 x 1000
features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))

# w1 * x1 + w2 * x2 + b
labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b

# ϵ ~ Epsilon
labels += nd.random.normal(scale=0.01, shape=labels.shape)

print(features[0])
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
    indices = list(range(num_examples))
    random.shuffle(indices)  # 样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        j = nd.array(indices[i: min(i + batch_size, num_examples)])
        yield features.take(j), labels.take(j)  # take函数根据索引返回对应元素

batch_size = 10

for X, y in data_iter(batch_size, features, labels):
    print(X, y)
    break

w = nd.random.normal(scale=0.01, shape=(num_inputs, 1))
b = nd.zeros(shape=(1,))

w.attach_grad()
b.attach_grad()

def linreg(X, w, b):  # 本函数已保存在d2lzh包中方便以后使用
    return nd.dot(X, w) + b

def squared_loss(y_hat, y):  # 本函数已保存在d2lzh包中方便以后使用
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

def sgd(params, lr, batch_size):  # 本函数已保存在d2lzh包中方便以后使用
    for param in params:
        param[:] = param - lr * param.grad / batch_size

lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):  # 训练模型一共需要num_epochs个迭代周期
    # 在每一个迭代周期中，会使用训练数据集中所有样本一次（假设样本数能够被批量大小整除）。X
    # 和y分别是小批量样本的特征和标签
    for X, y in data_iter(batch_size, features, labels):
        with autograd.record():
            l = loss(net(X, w, b), y)  # l是有关小批量X和y的损失
        l.backward()  # 小批量的损失对模型参数求梯度
        sgd([w, b], lr, batch_size)  # 使用小批量随机梯度下降迭代模型参数
    train_l = loss(net(features, w, b), labels)
    print('epoch %d, loss %f' % (epoch + 1, train_l.mean().asnumpy()))

print(true_w)
print(w)

print(true_b)
print(b)