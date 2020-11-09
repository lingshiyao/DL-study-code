import d2lzh as d2l
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn
import time

net = nn.Sequential()
net.add(nn.Conv2D(channels=6, kernel_size=5, activation='sigmoid'),

        # 池化2x2窗口
        nn.MaxPool2D(pool_size=2, strides=2),

        # 卷积
        nn.Conv2D(channels=16, kernel_size=5, activation='sigmoid'),

        # 池化2x2窗口
        nn.MaxPool2D(pool_size=2, strides=2),
        # Dense会默认将(批量大小, 通道, 高, 宽)形状的输入转换成
        # (批量大小, 通道 * 高 * 宽)形状的输入

        # 隐藏层
        nn.Dense(120, activation='sigmoid'),

        # 隐藏层
        nn.Dense(84, activation='sigmoid'),

        # 输出层
        nn.Dense(10))
net.initialize()
net.load_parameters("s5_5")

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

for X, y in test_iter:
    break

true_labels = d2l.get_fashion_mnist_labels(y.asnumpy())
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1).asnumpy())

titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

# 打印9张图片以及9张图片正确数据和预测数据的比较
d2l.show_fashion_mnist(X[0:9], titles[0:9])
d2l.plt.show()