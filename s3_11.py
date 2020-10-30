import d2lzh as d2l
from mxnet import autograd, gluon, nd
from mxnet.gluon import data as gdata, loss as gloss, nn

##-----------------------------------模型-----------------------------------
# y = 1.2x − 3.4x**2 + 5.6x**3 + 5 + ϵ
##-------------------------------------------------------------------------

# n_train           100                 训练数据大小100
# n_test            100                 测试数据大小100
# true_w            [1.2, -3.4, 5.6]
# true_b            5
n_train, n_test, true_w, true_b = 100, 100, [1.2, -3.4, 5.6], 5

# 生成训练数据
# 100 + 100, 1
# 200, 1
# <NDArray 200x1 @cpu(0)>

# feature           特点，特征，容貌
# --------------------
# 生成的X
# --------------------
features = nd.random.normal(shape=(n_train + n_test, 1))

# features                  x
# nd.power(features, 2)     x**2    平方操作
# nd.power(features, 3)     x**3    三次方操作
#----------------------------------------------------------
# <NDArray 200x3 @cpu(0)>
#       1          1           1       =>          3
#   200 []      200[]       200[]      =>       200[]
#----------------------------------------------------------

# poly features     聚合特点，特征，容貌
# --------------------
# 生成的X X**2 X**3
# --------------------
poly_features = nd.concat(features, nd.power(features, 2),
                          nd.power(features, 3))

# y = 1.2x − 3.4x**2 + 5.6x**3 + 5 + ϵ
# 此处用了广播
# true_w[0]         1.2             poly_features[:, 0](<NDArray 200 @cpu(0)>)
# true_w[1]         -3.4            poly_features[:, 1](<NDArray 200 @cpu(0)>)
# true_w[2]         5.6             poly_features[:, 2](<NDArray 200 @cpu(0)>)
# true_b            5

#
# [200] * [200] + [200] * [200] + [200] * [200] + 5 (广播)  == [200]

# labels            标签
# ----------
# 生成的Y
# ----------
labels = (true_w[0] * poly_features[:, 0] + true_w[1] * poly_features[:, 1]
          + true_w[2] * poly_features[:, 2] + true_b)

# -----------
# labels+=ϵ
# -----------
# 生成均值为0，标准差为0.1的正态分布
#
# ----------------------------------
# ----------正确数据在此时生成----------
# ----------------------------------
#
# -----labels-----
# <NDArray 200 @cpu(0)>
# ----------------
epsilon = nd.random.normal(scale=0.1, shape=labels.shape)

labels += epsilon

# print(features[:2])
# print(poly_features[:2])
# print(labels[:2])

# 打印图像
# 本函数已保存在d2lzh包中方便以后使用
def semilogy(x_vals, y_vals, x_label, y_label, x2_vals=None, y2_vals=None,
             legend=None, figsize=(3.5, 2.5)):
    d2l.set_figsize(figsize)
    d2l.plt.xlabel(x_label)
    d2l.plt.ylabel(y_label)
    d2l.plt.semilogy(x_vals, y_vals)
    if x2_vals and y2_vals:
        d2l.plt.semilogy(x2_vals, y2_vals, linestyle=':')
        d2l.plt.legend(legend)
    d2l.plt.show()

# num_epochs    学习轮数100轮
# loss
num_epochs, loss = 100, gloss.L2Loss()

# train_features    训练数据    100个
# test_features     测试数据    100个
# train_labels      训练标签    100个
# test_labels       测试标签    100个
# 每次调用都会生成一个新的模型，然后进行训练，测试
def fit_and_plot(train_features, test_features, train_labels, test_labels):

    # 获取模型
    net = nn.Sequential()

    # 设置模型有一层，并且输出只有1个
    # nn.Dense(1)
    #          这个1表示输出只有1个
    #
    # （此处没有添加激活函数，例如nn.Dense（256, activation='relu'））
    net.add(nn.Dense(1))

    # 初始化模型
    net.initialize()

    # train_labels
    # <NDArray 100 @cpu(0)>
    #
    # train_labels.shape[0]
    # 100
    # 每一批取10
    # 此处为什么要用min函数，而不是直接写死10呢???
    batch_size = min(10, train_labels.shape[0])

    # dataset ----------
    # {ArrayDataset: 100}   <mxnet.gluon.data.dataset.ArrayDataset object at 0x0000028D595AF128>
    # train_features        <NDArray 100x3 @cpu(0)>
    # train_labels          <NDArray 100 @cpu(0)>

    # dataset._length     =   100
    # dataset._data[0]    =   <NDArray 100x3 @cpu(0)>
    # dataset._data[1]    =   <NDArray 100 @cpu(0)>
    # 把数据封装成训练器需要的格式
    dataset = gdata.ArrayDataset(train_features, train_labels)

    # 装载训练器
    # dataset       训练数据
    # batch_size    批次      10
    train_iter = gdata.DataLoader(dataset, batch_size, shuffle=True)

    # 设置训练教练
    # params            net.collect_params()    从模型里面取出参数
    # sgd
    # learning_rate     0.01
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.01})

    train_ls, test_ls = [], []
    for _ in range(num_epochs):
        # 这里用一个匿名参数，只是为了迭代100次的操作

        for X, y in train_iter:
            # X <NDArray 10x3 @cpu(0)>
            # y <NDArray 10 @cpu(0)>

            with autograd.record():
                # 在做这一步之后
                # 模型的W就初始化了
                # 模型的b也初始化了
                net_result = net(X)
                l = loss(net_result, y)

            # 求出梯度
            l.backward()

            # 训练模型，
            # net[0].weight.data()  训练的W
            #     0表示模型的第一层（这个是单层的模型）
            # net[0].bias.data()    训练的b
            #     bias  偏置
            #     0表示模型的第一层（这个是单层的模型）
            trainer.step(batch_size)

        # 首先用训练过的模型
        train_loss_tmp = loss(net(train_features), train_labels)
        train_loss_tmp_tmp = train_loss_tmp.mean().asscalar()
        train_ls.append(train_loss_tmp_tmp)
        # train_ls.append(loss(net(train_features), train_labels).mean().asscalar())

        test_ls.append(loss(net(test_features), test_labels).mean().asscalar())

    print('final epoch: train loss', train_ls[-1], 'test loss', test_ls[-1])

    semilogy(range(1, num_epochs + 1), train_ls, 'epochs', 'loss',
             range(1, num_epochs + 1), test_ls, ['train', 'test'])

    print('weight:', net[0].weight.data().asnumpy(),
          '\nbias:', net[0].bias.data().asnumpy())

# 三阶多项式函数拟合（正常）
# poly_features[:n_train, :]
# <NDArray 100x3 @cpu(0)>

# labels[:n_train]
# <NDArray 100 @cpu(0)>
# --------------------------------------------------------
# poly_features                 <NDArray 200x3 @cpu(0)>
# --------------------------------------------------------

TTT1 = poly_features[:n_train, :]
TTT2 = poly_features[n_train:, :]

# 三阶多项式函数拟合（正常）
# w1， w2， w3
fit_and_plot(
            # poly_features[:n_train, :]    ==  poly_features[:100, :]      poly_features(0-100, :)
            # 此处把算好的特征数值（X X**2 X**3）传进来，0-100作为训练数据
            poly_features[:n_train, :],
            # poly_features[n_train:, :]    ==  poly_features[100:, :]      poly_features(100-200, :)
            # 此处把算好的特征数值（X X**2 X**3）传进来，100-200作为测试数据
            poly_features[n_train:, :],
            # labels[:n_train]      ==      labels[:100]    labels（0-100）
            # 此处把算好的标签（Y）传进来，0-100作为训练数据真实结果
            labels[:n_train],
            # labels[n_train:]      ==      labels[100:]    labels（100-200）
            # 此处把算好的标签（Y）传进来，100-200作为测试数据的真实结果
            labels[n_train:])

# 线性函数拟合（欠拟合）
fit_and_plot(
            features[:n_train, :],
            features[n_train:, :],
            labels[:n_train],
            labels[n_train:])

# 训练样本不足（过拟合）
# w1， w2， w3
fit_and_plot(
            poly_features[0:2, :],
            poly_features[n_train:, :],
            labels[0:2],
            labels[n_train:])