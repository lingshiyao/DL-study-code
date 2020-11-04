import d2lzh as d2l
from mxnet.gluon import data as gdata
import sys
import time

# 下载训练数据（train = True）表示下载训练数据
mnist_train = gdata.vision.FashionMNIST(train = True)

# 下载测试数据（train = False）表示下载测试数据
mnist_test = gdata.vision.FashionMNIST(train = False)

# 当前训练数据大小为60000
print(len(mnist_train))

# 当前测试数据大小为10000
print(len(mnist_test))

# --------------获取 训练数据 和 label（python语言特性）--------------------
# (feature, label) = (p1, p2)
# feature, label = (p1, p2)
# feature, label = mnist_train[0]
# ---------------------------------------------------------------------

# feature   ---     <NDArray 28x28x1 @cpu(0)>
# 1指的是通道，因为是灰度图像
# label     ---     2
feature, label = mnist_train[0]

# ----------------------------python数据类型----------------------------
# Int8, 占1个字节.   有符号整型
# Int16, 占2个字节.  有符号整型
# Int32, 占4个字节.  有符号整型
# Int64, 占8个字节.  有符号整型
# uint8, 占8个字节.     0-255
# ---------------------------------------------------------------------

# 打印feature的形状
# 结果为(28, 28, 1)，为高28，宽28，通道为1的灰度图片
# {tuple:3} 元素空间为3 (28, 28, 1)
print(feature.shape)

# 打印图数据的类型
# <class 'numpy.uint8'>
print(feature.dtype)

# 打印label
print(label)

# 打印label的类型
print(type(label))
# print(label.dtype)

def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

def show_fashion_mnist(images, labels):
    d2l.use_svg_display()
    # 这里的_表示我们忽略（不使用）的变量
    _, figs = d2l.plt.subplots(1, len(images), figsize=(12, 12))

    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.reshape((28, 28)).asnumpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)

    # d2l.plt.show()

# X 取出10个图片的数据
# y 取出10个图片对应标签的下标
X, y = mnist_train[0:9]

# 打印当前取出的数据（把图像打印出来）
show_fashion_mnist(X, get_fashion_mnist_labels(y))

# 批次大小256
batch_size = 256

# 通过ToTensor将数据从unit8格式变成32位浮点数格式
transformer = gdata.vision.transforms.ToTensor()

# 0表示不用额外的进程来加速读取数据
# sys.platform == win32
if sys.platform.startswith('win'):
    num_workers = 0
else:
    num_workers = 4

# 训练数据
# 把训练数据加载到训练器里面
# 数据
# 批次      ??
# 用来训练的额外进程，当前为0
train_iter = gdata.DataLoader(mnist_train.transform_first(transformer),
                              batch_size, shuffle=True,
                              num_workers=num_workers)

# 测试数据
# 把测试数据加载到训练器里面
test_iter = gdata.DataLoader(mnist_test.transform_first(transformer),
                             batch_size, shuffle=False,
                             num_workers=num_workers)

start = time.time()
for X, y in train_iter:
    continue
'%.2f sec' % (time.time() - start)