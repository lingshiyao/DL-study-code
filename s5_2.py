from mxnet import nd
from mxnet.gluon import nn

# 定义一个函数来计算卷积层。它初始化卷积层权重，并对输入和输出做相应的升维和降维
def comp_conv2d(conv2d, X):
    conv2d.initialize()
    # (1, 1)代表批量大小和通道数（“多输入通道和多输出通道”一节将介绍）均为1

    # (8 8) => (1 1 8 8)
    X = X.reshape((1, 1) + X.shape)
    Y = conv2d(X)
    #                       裁剪
    return Y.reshape(Y.shape[2:])  # 排除不关心的前两维：批量和通道

# 注意这里是两侧分别填充1行或列，所以在两侧一共填充2行或列

# 通道数为              1
# 核的数量为             3
# 填充                 1
# 定义一个卷积层
conv2d = nn.Conv2D(1, kernel_size=3, padding=1)

# 8 x 8
X = nd.random.uniform(shape=(8, 8))
comp_conv2d(conv2d, X).shape

# 使用高为5、宽为3的卷积核。在高和宽两侧的填充数分别为2和1
conv2d = nn.Conv2D(1, kernel_size=(5, 3), padding=(2, 1))
comp_conv2d(conv2d, X).shape

conv2d = nn.Conv2D(1, kernel_size=3, padding=1, strides=2)
comp_conv2d(conv2d, X).shape

conv2d = nn.Conv2D(1, kernel_size=(3, 5), padding=(0, 1), strides=(3, 4))
comp_conv2d(conv2d, X).shape