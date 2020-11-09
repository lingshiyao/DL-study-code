import d2lzh as d2l
import mxnet as mx
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn
import time

net = nn.Sequential()

# 卷积层用来识别图像里的空间模式，如线条和物体局部

        # 卷积6通道5x5窗口
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

# 1 x 1 x 28 x 28
X = nd.random.uniform(shape=(1, 1, 28, 28))
net.initialize()
for layer in net:
    # 1x1x28x28 => 1x6x24x24	Conv2D(1 -> 6, kernel_size=(5, 5), stride=(1, 1), Activation(sigmoid))
    # (28 - 5 + 0 + 1) / 1 = 24

    # 1x6x24x24 => 1x6x12x12	MaxPool2D(size=(2, 2), stride=(2, 2), padding=(0, 0), ceil_mode=False, global_pool=False, pool_type=max, layout=NCHW)
    # (24 - 2 + 0 + 2) / 2 = 12

    # 1x6x12x12 => 1x16x8x8	    Conv2D(6 -> 16, kernel_size=(5, 5), stride=(1, 1), Activation(sigmoid))
    # (12 - 5 + 0 + 1) / 1 = 8

    # 1x16x8x8  => 1x16x4x4	    MaxPool2D(size=(2, 2), stride=(2, 2), padding=(0, 0), ceil_mode=False, global_pool=False, pool_type=max, layout=NCHW)
    # (8 - 2 + 0 + 2) / 2 = 4

    # 1x16x4x4  => 1x120		Dense(256 -> 120, Activation(sigmoid))
    # 1x256     => 1x120

    # 1x120	    => 1x84		    Dense(120 -> 84, Activation(sigmoid))

    # 1x84	    => 1x10		    Dense(84 -> 10, linear)
    X = layer(X)
    print(layer.name, 'output shape:\t', X.shape)

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size=batch_size)

def try_gpu():  # 本函数已保存在d2lzh包中方便以后使用
    try:
        ctx = mx.gpu()
        _ = nd.zeros((1,), ctx=ctx)
    except mx.base.MXNetError:
        ctx = mx.cpu()
    return ctx

ctx = try_gpu()
print(ctx)

# 本函数已保存在d2lzh包中方便以后使用。该函数将被逐步改进：它的完整实现将在“图像增广”一节中
# 描述
def evaluate_accuracy(data_iter, net, ctx):
    acc_sum, n = nd.array([0], ctx=ctx), 0
    for X, y in data_iter:
        # 如果ctx代表GPU及相应的显存，将数据复制到显存上
        X, y = X.as_in_context(ctx), y.as_in_context(ctx).astype('float32')
        acc_sum += (net(X).argmax(axis=1) == y).sum()
        n += y.size
    return acc_sum.asscalar() / n

# 本函数已保存在d2lzh包中方便以后使用
def train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx,
              num_epochs):
    print('training on', ctx)

    # 获取交叉熵loss
    loss = gloss.SoftmaxCrossEntropyLoss()

    # 5轮
    for epoch in range(num_epochs):
        # train_l_sum       0.0
        # train_acc_sum     0.0
        # n                 0
        # start             time.time()
        train_l_sum, train_acc_sum, n, start = 0.0, 0.0, 0, time.time()
        for X, y in train_iter:
            X, y = X.as_in_context(ctx), y.as_in_context(ctx)
            with autograd.record():
                y_hat = net(X)
                l = loss(y_hat, y).sum()
            l.backward()
            trainer.step(batch_size)
            y = y.astype('float32')
            train_l_sum += l.asscalar()
            train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
            n += y.size
        test_acc = evaluate_accuracy(test_iter, net, ctx)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f, '
              'time %.1f sec'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc,
                 time.time() - start))

# 学习率0.9
# 轮数5轮
lr, num_epochs = 0.9, 50
net.initialize(force_reinit=True, ctx=ctx, init=init.Xavier())
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
train_ch5(net, train_iter, test_iter, batch_size, trainer, ctx, num_epochs)

net.save_parameters("s5_5")

