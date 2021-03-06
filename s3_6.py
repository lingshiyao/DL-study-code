import d2lzh as d2l
from mxnet import autograd, nd
from time import time

start = time()

# 批次大小为256（每一批的图片数量为256）
batch_size = 256

# 加载训练器和测试器（输入每一批的批次数量）
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

# 28 * 28 = 784（图片的分辨率为28 x 28，总像素为784，所以输入层大小为784）
num_inputs = 784

# 设置输出层的大小为10（输出层10个类别，最终输出层的个数为10，这个是输入图像分类问题）
num_outputs = 10

# shape=(num_inputs, num_outputs)
#       784     x   10
# 生成初始数据W，W为均值是0，标准差为0.01的正态分布
# （随便填写什么数值，反正后面会用梯度下降来校验）
W = nd.random.normal(scale=0.01, shape=(num_inputs, num_outputs))

# y = w(1)1 * x1 + w(1)2 * x2 + w(1)3 * x3 + ...... + w(1)784 * x784 + b(1)
# 生成偏置数据，初始数值0(长度为10，分别为b(1).b(2).b(3).b(4)......b(10))
b = nd.zeros(num_outputs)

# W生成梯度
W.attach_grad()

# b生成梯度
b.attach_grad()

##-----X.sum(axis=1, keepdims=True)---------------------------------W
##-----求SUM值的测试例子-----------------------------------------------W
##------------------------------------------------------------------W
X = nd.array([[1, 2, 3], [4, 5, 6]])

# 按照列求值
#   1 2 3
#   4 5 6
#------------
#   5 7 9

# print(X.sum(axis=0, keepdims=True))

# 按照行求值
#   1 2 3  |  6
#   4 5 6  |  15

# print(X.sum(axis=1, keepdims=True))

##------------------------------------------------------------------M
##------------------------------------------------------------------M
##------------------------------------------------------------------M


##-----softmax------------------------------------------------------W
##-----把结果集换算成百分-----------------------------------------------W
##------------------------------------------------------------------W
# exp(o1) /  exp(o1) + exp(o2) + ...... + exp(on)
def softmax(X):
    # 对矩阵里面的数值做e的指数运算
    X_exp = X.exp()

    # 按照行求值
    partition = X_exp.sum(axis=1, keepdims=True)
    return X_exp / partition  # 这里应用了广播机制
##------------------------------------------------------------------M
##------------------------------------------------------------------M
##------------------------------------------------------------------M


##------------------------------------------------------------------W
##-----softmax功能测试------------------------------------------------W
##------------------------------------------------------------------W
# 生成标准正态分布 2 x 5 测试数据
X = nd.random.normal(shape=(2, 5))

# 使用自定义的softmax函数
X_prob = softmax(X)

# X_prob的数据格式如下面例子所示
# [[0.6264712  0.126293   0.01826552 0.10885343 0.12011679]
# [0.25569436 0.2917251  0.07546549 0.3024068  0.07470828]]
# print(X_prob)

# 校验softmax函数是否生效
# 结果是[1, 1]
# 按行求值，一行的值应该总和为1
print(X_prob.sum(axis=1))
##------------------------------------------------------------------M
##------------------------------------------------------------------M
##------------------------------------------------------------------M

##-------测试reshape-------------------------------------------------W
##------------------------------------------------------------------W
##------------------------------------------------------------------W
# print("+" * 40)
# print(X)
#
# # 2 x 5 => 1 x 9
# TTT = X.reshape((-1, 9))
#
# # 2 x 5 => 1 x 10
# TTT = X.reshape((-1, 10))
# print(TTT)
#
# TTT = nd.random.normal(shape=(3,3,3))
# print(TTT)
# print('*' * 100, '-' * 100)
# print(TTT.reshape(-1, 3))
# print(TTT.reshape(-1, 9))
# print(TTT.reshape(-1, 27))
##------------------------------------------------------------------M
##------------------------------------------------------------------M
##------------------------------------------------------------------M

##------------------------------------------------------------------W
##--------dot就是标准的矩阵的点乘法-------------------------------------W
##------------------------------------------------------------------W
# TTT1 = nd.array([1,2])
# print(TTT1)
# TTT2 = nd.array([3,4]).reshape(2, 1)
# print(TTT2)
# TTT3 = nd.dot(TTT1, TTT2)
# print(TTT3)
# print('*' * 100, '-' * 100)
##------------------------------------------------------------------M
##------------------------------------------------------------------M
##------------------------------------------------------------------M

# 定义模型函数
def net(X):
    # 把256x1x28x28 变成256x784
    # 256x784  784x10  => 256x10
    # softmax(XW + b)
    return softmax(nd.dot(X.reshape((-1, num_inputs)), W) + b)

# 预测值
# 0.1   0.3   0.6       第一个样本预测取下标为2的0.6的值，最有可能是2的样本
# 0.3   0.2   0.5       第二个样本预测去下表为2的0.5的值
y_hat = nd.array([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])

# 样本真实的下标
# 0, 2
y = nd.array([0, 2], dtype='int32')

#----------------------PICK操作-------------------
# pick是根据下标取值的操作
# y_hat第一行取第0个
# y_hat第二行取第2个
print(nd.pick(y_hat, y))

##------------------------------------------------------------------W
##--------pickTest--------------------------------------------------W
##------------------------------------------------------------------W
# XXXX1 = nd.array([0,1,2,3,4,5,6,7,8]).reshape((3, 3))
# XXXX2 = nd.array([0,0,0])
# print("#" * 100)
# print(XXXX1)
# print(nd.pick(XXXX1, XXXX2))
##------------------------------------------------------------------M
##------------------------------------------------------------------M
##------------------------------------------------------------------M

# ======用交叉熵计算损失======
# 求熵值的公式ln1/p
# 熵是求出概率意外程度的计算方式
def cross_entropy(y_hat, y):

    # 取出选中正确衣服的预测概率
    tmp = nd.pick(y_hat, y)

    # 此处把真实数据对应的概率选中，然后求ln
    # 如果都选对了，ln1 = 0

    # ln1/p = -lnp
    # (log就是求ln)
    return -tmp.log()

##------------------------------------------------------------------W
##--------mean是求中值的操作-------------------------------------------W
##------------------------------------------------------------------W
# tmp1 = nd.array([1,2,3,4,100])
# print("@" * 100)
# print(tmp1.mean())
##------------------------------------------------------------------M
##------------------------------------------------------------------M
##------------------------------------------------------------------M

# 计算整体样本预测的准确度
# （结果数值在0~1之间）
# （单个预测对是1，错是0，整体是加起来求平均值）
def accuracy(y_hat, y):
    # 返回y_hat中最大的索引
    # 也就是说，把两次的计算出来的数值下标取出来
    # print(y_hat.argmax(axis=1))

    #-------------------------test
    # print("+=" * 100)
    # XXXX1 = nd.array([1,2,3,4,5])
    # XXXX2 = nd.array([2, 2, 3, 4, 5])
    # XXXX3 = nd.array([1,2,3])
    # print(XXXX1 == XXXX2)
    # print(XXXX3.norm())
    # print(XXXX3.norm().asscalar())
    #-------------------------test

    # 预测的是[2,2] （y_hat.argmax(axis=1)取出最大数值的标记）
    # 真实的是[0,2]
    # 第一个是错的，第二个是对的 [0,1]  [错, 对] （y_hat.argmax(axis=1) == y.astype('float32') 返回数值相同的索引 [0,1]）
    # 取平均值是0.5，记录到一个向量中 (mean操作)
    # 最后把长度为1的向量（一维数组）转换成标量(asscalar把1x1的数组转换成数字)
    # 返回0.5
    return (y_hat.argmax(axis=1) == y.astype('float32')).mean().asscalar()

print(accuracy(y_hat, y))

# 本函数已保存在d2lzh包中方便以后使用。该函数将被逐步改进：它的完整实现将在“图像增广”一节中
# data_iter     （data iter 数据通路）
# net           （softmax模型）
def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        # for每次取出来的是256张图片，可能是训练数据，可能是测试数据

        #-----------------------???-----------------------
        #   256x1x28x28
        #   256张图
        #   单通道？0~1？
        #   28x28
        #-------------------------------------------------

        # 把y转换成float32类型
        y = y.astype('float32')

        # (net(X)算出256x10)
        tmp_y_hat = net(X)

        # 求出当前计算出的准确度
        # （argmax是取出最大数值arg的下标）
        acc_sum += (tmp_y_hat.argmax(axis=1) == y).sum().asscalar()

        n += y.size

    # 如果是60000的训练数据，此时sum是正确的总数，n=60000
    return acc_sum / n


##------------------------------------------------------------------W
##--------准确度函数测试-----------------------------------------------W
##------------------------------------------------------------------W
# test_iter 加载的训练器
# net 当前的模型
# accuracyTest = evaluate_accuracy(test_iter, net)

# 打印当前的准确率
# print(accuracyTest)
##------------------------------------------------------------------M
##------------------------------------------------------------------M
##------------------------------------------------------------------M

# num_epochs    迭代周期
# lr            学习率
num_epochs, lr = 5, 0.1

# 本函数已保存在d2lzh包中方便以后使用
# num_epochs 迭代5次（5次梯度？）
#
# net               当前定义的模型
# train_iter        当前的训练器
# test_iter         当前的测试器
# loss              cross_entropy用来计算损失的交叉熵
# num_epochs        迭代周期，当前为5
# batch_size        每一批次的大小
# params            [W, b]
# lr                学习率，梯度下降的速度
oldW = nd.array(W)
def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, trainer=None):
    # 此处需要迭代5次
    for epoch in range(num_epochs):
        # train_l_sum 训练60000张图片正确图片概率熵值的总和
        #             熵值的损失  train_l_sum / n
        # train_acc_sum 训练60000张图片的准确率
        #               正确结果为R一张图片正确记为1，错误记为0，
        #               (R1 + ...... + R60000) / n
        #               (R1 + ...... + R60000) / 60000
        #                                        n作为训练数据为60000
        # n训练数据（训练图片的总数60000）
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            # 每次取出256张图片数据，参考下面打印的结果
            # X <NDArray 256x1x28x28 @cpu(0)>
            # y <NDArray 256 @cpu(0)>

            with autograd.record():
                # y_hat 256 x 10
                # 用模型进行计算
                y_hat = net(X)

                # 选出当前训练数据，对应正确的下标记的概率的总和
                # lose(y_hat, y)求出的是当前正确衣服预测概率的熵，（如果预测正确，概率为P，熵为-lnp = -0，熵值越小，意外度越小，损失越小）
                # loss(y_hat, y).sum()求出一批数据（256张图像）的熵值的总数
                # 下面是简单的ln百分比参考
                # -ln0.01   = 4.605
                # -ln0.10   = 2.302
                # -ln0.50   = 0.693
                # -ln0.90   = 0.105
                # -ln0.99   = 0.010
                # -----!!!!!l是一批数据256张图正确图片出现概率熵值的总和!!!!!-----
                l = loss(y_hat, y).sum()

            # 计算梯度
            l.backward()

            if trainer is None:
                #W----------------!!!!!!!!!!!!!----------------W
                # 梯度下降（开始“学习”W和b）
                #M----------------!!!!!!!!!!!!!----------------M
                # batch_size    256
                # lr            0.1
                d2l.sgd(params, lr, batch_size)
            else:
                trainer.step(batch_size)  # “softmax回归的简洁实现”一节将用到

            # y原来是int类型的正确衣服的下标，转换成float类型
            y = y.astype('float32')

            # 转换成标量，然后加过去
            train_l_sum += l.asscalar()

            # 求出正确的总数
            train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
            n += y.size

        # 每学习一次，用测试数据进行一次测试操作，算出准确度
        # 用当前训练轮数的模型，进行softmax，然后当前预测的服装的下标是否正确（正确为1，错误为0），(i1 + ...... + i60000) / 60000
        test_acc = evaluate_accuracy(test_iter, net)

        # epoch         第几轮
        # loss          损失
        # train acc     训练数据acc（准确率）
        # test acc      测试数据acc（准确率）
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

#
# net               当前定义的模型
# train_iter        当前的训练器
# test_iter         当前的测试器
# cross_entropy     用来计算损失的交叉熵
# num_epochs        迭代周期
# batch_size        每一批次的大小
# [W,b]
# W                 W初始的W
# b                 偏置数据
# lr                学习率（梯度下降的速度）
# !!!------!!!开始学习!!!------!!!
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, batch_size,
          [W, b], lr)

# 随便取出第一批的数据256个
for X, y in test_iter:
    break

# y.asnumpy()只是把数据类型进行转换
# 然后取出对应标签
true_labels = d2l.get_fashion_mnist_labels(y.asnumpy())

# 使用模型预测，然后取出对应的标签
pred_labels = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1).asnumpy())

titles = [true + '\n' + pred for true, pred in zip(true_labels, pred_labels)]

# 打印9张图片以及9张图片正确数据和预测数据的比较
d2l.show_fashion_mnist(X[0:9], titles[0:9])
# d2l.plt.show()

# 22.080242395401
# 22.875707149505615
# 21.316385984420776
print(time() - start)