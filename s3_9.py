import d2lzh as d2l
from mxnet import nd
from mxnet.gluon import loss as gloss

# 每一批的图片256张
batch_size = 256

# 装载训练项目
# 装载测试项目
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# num_inputs    输入      28 x 28 = 784
# num_outputs   输出类别   10
# num_hiddens   隐藏层     256
num_inputs, num_outputs, num_hiddens = 784, 10, 256

# 生成W1  （784 x 256）
# 均值为0，标准差为0.01的正态分布
W1 = nd.random.normal(scale=0.01, shape=(num_inputs, num_hiddens))

# 生成b1  （256）
b1 = nd.zeros(num_hiddens)

# 生成W2  （256 x 10）
W2 = nd.random.normal(scale=0.01, shape=(num_hiddens, num_outputs))

# 生成b2  （10）
b2 = nd.zeros(num_outputs)

params = [W1, b1, W2, b2]

# W1,W2,b1,b2生成梯度
for param in params:
    param.attach_grad()

# 自定义relu函数
def relu(X):
    return nd.maximum(X, 0)

# ----------------自定义relu函数测试----------------
TTTX = nd.array([-1,2,-3,4]).reshape((2,2))
TTTY = relu(TTTX)
# ------------------------------------------------

# 定义模型
def net(X):
    # X [256, 784]
    X = X.reshape((-1, num_inputs))

    # H Hidden
    # H = relu(XW1 + b1)
    H = relu(nd.dot(X, W1) + b1)
    # O = HW2 + b2
    return nd.dot(H, W2) + b2

# 设置loss函数 lose为softmax后求熵
loss = gloss.SoftmaxCrossEntropyLoss()

# num_epochs    学习次数5
# 学习率         学习率0.5
num_epochs, lr = 5, 0.5

# net           模型
# train_iter    训练项目
# test_iter     测试项目
# loss          损失函数
# num_epochs    迭代周期
# batch_size    批次
# params        需要学习的参数
# lr            学习率
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params, lr)