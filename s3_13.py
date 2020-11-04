import d2lzh as d2l
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import loss as gloss, nn

# drop_prob     丢弃概率
# keep_prob     保留概率
def dropout(X, drop_prob):
    assert 0 <= drop_prob <= 1
    keep_prob = 1 - drop_prob
    # 这种情况下把全部元素都丢弃
    if keep_prob == 0:
        return X.zeros_like()

    # 生成 0 ~ 1 之间的随机数
    # 2 x 8
    random_prob = nd.random.uniform(0, 1, X.shape)

    # 2 x 8
    mask = random_prob < keep_prob
    return mask * X / keep_prob

# 2 x 8
X = nd.arange(16).reshape((2, 8))

print(dropout(X, 0))

print(dropout(X, 0.5))

print(dropout(X, 1))

# num_inputs    输入层     784
# num_outputs   输出层     10
# num_hiddens1  隐藏层1    256
# num_hiddens2  隐藏层2    256
num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256

# 784 x 256
W1 = nd.random.normal(scale=0.01, shape=(num_inputs, num_hiddens1))

# 256
b1 = nd.zeros(num_hiddens1)

# 256 x 256
W2 = nd.random.normal(scale=0.01, shape=(num_hiddens1, num_hiddens2))

# 256
b2 = nd.zeros(num_hiddens2)

# 256 x 10
W3 = nd.random.normal(scale=0.01, shape=(num_hiddens2, num_outputs))

# 10
b3 = nd.zeros(num_outputs)

# 批量生成梯度
params = [W1, b1, W2, b2, W3, b3]
for param in params:
    param.attach_grad()

# 隐藏层第一层丢弃率为0.2
# 隐藏层第二层丢弃率为0.5
drop_prob1, drop_prob2 = 0.2, 0.5

# 定义模型
def net(X):
    # 256x1x28x28   =>  256 x 784
    X = X.reshape((-1, num_inputs))

    # 计算隐藏层H1的输出
    # relu激活函数
    H1 = (nd.dot(X, W1) + b1).relu()

    # 当训练模型的时候
    if autograd.is_training():  # 只在训练模型时使用丢弃法
        H1 = dropout(H1, drop_prob1)  # 在第一层全连接后添加丢弃层

    # 计算隐藏层H2的输出
    H2 = (nd.dot(H1, W2) + b2).relu()

    # 当训练模型的时候
    if autograd.is_training():
        H2 = dropout(H2, drop_prob2)  # 在第二层全连接后添加丢弃层

    return nd.dot(H2, W3) + b3

# num_epochs    学习5轮
# lr            学习率0.5
# batch_size    批次大小256
num_epochs, lr, batch_size = 5, 0.5, 256

# 损失函数为交叉熵
loss = gloss.SoftmaxCrossEntropyLoss()

# batch_size    批次大小256
# 获取训练器
# 获取测试器
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# net           模型
# train_iter    训练器
# test_iter     测试器
# loss          损失
# num_epochs    训练轮数
# batch_size    批次
# params        训练参数
# lr            学习率

# epoch 1, loss 1.2458, train acc 0.518, test acc 0.765
# epoch 2, loss 0.6049, train acc 0.775, test acc 0.837
# epoch 3, loss 0.5105, train acc 0.815, test acc 0.849
# epoch 4, loss 0.4536, train acc 0.834, test acc 0.854
# epoch 5, loss 0.4279, train acc 0.843, test acc 0.862
# d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)

# --------------------------------------------------------------
# 1.如果把本节中的两个丢弃概率超参数对调，会有什么结果？
#
#   感觉没有什么变化
# --------------------------------------------------------------
# drop_prob1, drop_prob2 = 0.5, 0.2

# epoch 1, loss 1.2373, train acc 0.517, test acc 0.765
# epoch 2, loss 0.6306, train acc 0.763, test acc 0.821
# epoch 3, loss 0.5308, train acc 0.802, test acc 0.839
# epoch 4, loss 0.4877, train acc 0.820, test acc 0.844
# epoch 5, loss 0.4624, train acc 0.829, test acc 0.857
# d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)

# --------------------------------------------------------------
# 2.增大迭代周期数，比较使用丢弃法与不使用丢弃法的结果。
#
#   感觉没有什么变化
# --------------------------------------------------------------
def net_no_dropout(X):
    # 256x1x28x28   =>  256 x 784
    X = X.reshape((-1, num_inputs))

    # 计算隐藏层H1的输出
    # relu激活函数
    H1 = (nd.dot(X, W1) + b1).relu()

    # 当训练模型的时候
    # if autograd.is_training():  # 只在训练模型时使用丢弃法
    #     H1 = dropout(H1, drop_prob1)  # 在第一层全连接后添加丢弃层

    # 计算隐藏层H2的输出
    H2 = (nd.dot(H1, W2) + b2).relu()

    # 当训练模型的时候
    # if autograd.is_training():
    #     H2 = dropout(H2, drop_prob2)  # 在第二层全连接后添加丢弃层

    return nd.dot(H2, W3) + b3

# epoch 1, loss 1.2564, train acc 0.506, test acc 0.750
# epoch 2, loss 0.5852, train acc 0.775, test acc 0.833
# epoch 3, loss 0.4798, train acc 0.823, test acc 0.841
# epoch 4, loss 0.4326, train acc 0.840, test acc 0.854
# epoch 5, loss 0.3950, train acc 0.852, test acc 0.862
# d2l.train_ch3(net_no_dropout, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)

num_epochs = 10

# epoch 1, loss 1.1070, train acc 0.561, test acc 0.719
# epoch 2, loss 0.6169, train acc 0.771, test acc 0.821
# epoch 3, loss 0.4885, train acc 0.818, test acc 0.852
# epoch 4, loss 0.4211, train acc 0.843, test acc 0.860
# epoch 5, loss 0.3905, train acc 0.856, test acc 0.864
# epoch 6, loss 0.3767, train acc 0.861, test acc 0.865
# epoch 7, loss 0.3484, train acc 0.871, test acc 0.868
# epoch 8, loss 0.3352, train acc 0.875, test acc 0.882
# epoch 9, loss 0.3202, train acc 0.880, test acc 0.881
# epoch 10, loss 0.3125, train acc 0.884, test acc 0.879
# d2l.train_ch3(net_no_dropout, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)

# epoch 1, loss 1.1294, train acc 0.556, test acc 0.781
# epoch 2, loss 0.5885, train acc 0.782, test acc 0.835
# epoch 3, loss 0.4892, train acc 0.821, test acc 0.822
# epoch 4, loss 0.4424, train acc 0.838, test acc 0.855
# epoch 5, loss 0.4182, train acc 0.848, test acc 0.864
# epoch 6, loss 0.3960, train acc 0.855, test acc 0.866
# epoch 7, loss 0.3793, train acc 0.862, test acc 0.875
# epoch 8, loss 0.3710, train acc 0.866, test acc 0.873
# epoch 9, loss 0.3569, train acc 0.869, test acc 0.878
# epoch 10, loss 0.3466, train acc 0.873, test acc 0.878
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)

# --------------------------------------------------------------
# ----------------------------简洁实现----------------------------
# --------------------------------------------------------------

net = nn.Sequential()

net.add(nn.Dense(256, activation="relu"),
        nn.Dropout(drop_prob1),  # 在第一个全连接层后添加丢弃层
        nn.Dense(256, activation="relu"),
        nn.Dropout(drop_prob2),  # 在第二个全连接层后添加丢弃层
        nn.Dense(10))
net.initialize(init.Normal(sigma=0.01))

trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None,
              None, trainer)