import d2lzh as d2l
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn

# y = 0.05 + (0.01 * x1 + 0.01 * x2 + ...... + 0.01 * xp) + ϵ
# y = 0.05 + (0.01 * x1 + 0.01 * x2 + ...... + 0.01 * x200) + ϵ

# n_train       20      训练长度大小20
# n_test        100     测试数据大小100
# num_inputs    200     p有200个
n_train, n_test, num_inputs = 20, 100, 200

# true_w        <NDArray 200x1 @cpu(0)>         [[0.01][0.01]......[0.01]]
#                                                        |-   200   -|
#                                               后面用广播
#
# true_b                                        0.05
#                                               后面用广播
true_w, true_b = nd.ones((num_inputs, 1)) * 0.01, 0.05

# features      <NDArray 120x200 @cpu(0)>       120 [......]
#                                                      200
#
#               n_train(20) + n_test(100), num_inputs
#               120           x            200
features = nd.random.normal(shape=(n_train + n_test, num_inputs))

# nd.dot(features, true_w)      <NDArray 120x200 @cpu(0)> x <NDArray 200x1 @cpu(0)>
# true_b                        0.05
labels = nd.dot(features, true_w) + true_b

# <NDArray 120x1 @cpu(0)>
# ϵ
epsilon = nd.random.normal(scale=0.1, shape=labels.shape)

# labels.shape (120, 1)
#
# y = 0.05 + (0.01 * x1 + 0.01 * x2 + ...... + 0.01 * xp) + ϵ
# y = 0.05 + (0.01 * x1 + 0.01 * x2 + ...... + 0.01 * x200) + ϵ
#------------------------------------------------------------------------
# labels    =   true_b  +   nd.dot(features,     true_w)    +   epsilon
#------------------------------------------------------------------------
# 120x1         1x1                120x200       200x1          120x1
#------------------------------------------------------------------------
labels += epsilon

# train_features        训练数据        20 x 200
# test_features         测试数据       100 x 200
train_features, test_features = features[:n_train, :], features[n_train:, :]

# train_labels          训练标签        20 x 1
# test_labels           测试标签       100 x 1
train_labels, test_labels = labels[:n_train], labels[n_train:]

def init_params():
    # 200 x 1
    w = nd.random.normal(scale=1, shape=(num_inputs, 1))

    # 1 x 1
    b = nd.zeros(shape=(1,))

    # 生成w梯度
    w.attach_grad()

    # 生成b梯度
    b.attach_grad()

    # 返回[w, b]
    return [w, b]

# l2惩罚函数
# penalty       ----------      惩罚的意思
def l2_penalty(w):
    return (w ** 2).sum() / 2

# batch_size        批次大小        1        1
# num_epochs        训练轮数        100
# lr                学习率          0.003
batch_size, num_epochs, lr = 1, 100, 0.003

# net               模型
# loss              损失
net, loss = d2l.linreg, d2l.squared_loss

# train_iter        训练器
train_iter = gdata.DataLoader(gdata.ArrayDataset(train_features, train_labels), batch_size, shuffle=True)

# lambd     λ
def fit_and_plot(lambd):
    # w     <NDArray 200x1 @cpu(0)>
    # b     <NDArray 1x1 @cpu(0)>
    w, b = init_params()
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        # 训练100次（梯度下降100次)
        for X, y in train_iter:
            # 训练批次为20，每一次取出一个训练，一共训练20次

            # X <NDArray 1x200 @cpu(0)>
            # y <NDArray 1x1 @cpu(0)>
            with autograd.record():
                # 添加了L2范数惩罚项，广播机制使其变成长度为batch_size的向量
                # loss + λ / 2 * ||w||2
                #
                #                L2范数的平方
                #
                # 带惩罚函数的损失

                # 惩罚函数
                # λ / 2 * (∥w∥ ** 2)
                # l2_penalty = (w ** 2).sum() / 2
                # lambd * l2_penalty(w) = lambd / 2 * ((w ** 2).sum())

                # 损失计算
                # nd.dot(X, w) + b
                # X - <NDArray 1x200 @cpu(0)>
                # w - <NDArray 200x1 @cpu(0)>
                # b - <NDArray 1 @cpu(0)>
                netResult = net(X, w, b)
                lossO = loss(netResult, y)
                l = lossO + lambd * l2_penalty(w)

                # l = loss(net(X, w, b), y) + lambd * l2_penalty(w)
            # 计算梯度
            l.backward()

            # batch_size        1
            # lr                0.003
            # [w, b]            [200 x 1, 1 x 1]
            # 梯度下降
            d2l.sgd([w, b], lr, batch_size)

        # train_features    *   w   +   b       =       y
        # 20x200                200x1   1x1             20x1
        train_ls.append(loss(net(train_features, w, b),
                             train_labels).mean().asscalar())
        test_ls.append(loss(net(test_features, w, b),
                            test_labels).mean().asscalar())
    d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
                 range(1, num_epochs + 1), test_ls, ['train', 'test'])

    # --------------------normTest--------------------
    # TTT1 = nd.array((1,2,3,4)).reshape((1, 4))
    # 1 + 4 + 9 + 16 = 30
    # TTT2 = TTT1.norm().asscalar()
    # --------------------normTest--------------------

    print('L2 norm of w:', w.norm().asscalar())

fit_and_plot(lambd=0)

fit_and_plot(lambd=3)

def fit_and_plot_gluon(wd):
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize(init.Normal(sigma=1))
    # 对权重参数衰减。权重名称一般是以weight结尾
    trainer_w = gluon.Trainer(net.collect_params('.*weight'), 'sgd',
                              {'learning_rate': lr, 'wd': wd})
    # 不对偏差参数衰减。偏差名称一般是以bias结尾
    trainer_b = gluon.Trainer(net.collect_params('.*bias'), 'sgd',
                              {'learning_rate': lr})
    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        for X, y in train_iter:
            with autograd.record():
                l = loss(net(X), y)
            l.backward()
            # 对两个Trainer实例分别调用step函数，从而分别更新权重和偏差
            trainer_w.step(batch_size)
            trainer_b.step(batch_size)
        train_ls.append(loss(net(train_features),
                             train_labels).mean().asscalar())
        test_ls.append(loss(net(test_features),
                            test_labels).mean().asscalar())
    d2l.semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
                 range(1, num_epochs + 1), test_ls, ['train', 'test'])
    print('L2 norm of w:', net[0].weight.data().norm().asscalar())

fit_and_plot_gluon(0)

fit_and_plot_gluon(3)