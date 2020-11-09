import mxnet as mx
from mxnet import nd
from mxnet.gluon import nn

print(mx.cpu())
print(mx.gpu())
print(mx.gpu(1))

x = nd.array([1, 2, 3])
print("x", x)

print("x.context", x.context)

a = nd.array([1, 2, 3], ctx=mx.gpu())
print("a", a)

# 把内存上面的数据拷贝到GPU上面
y = x.copyto(mx.gpu())
print("y", y)

#
z = x.as_in_context(mx.gpu())
print("z", z)

print("y.as_in_context(mx.gpu()) is y", y.as_in_context(mx.gpu()) is y)

print("y.copyto(mx.gpu()) is y", y.copyto(mx.gpu()) is y)

print("(z + 2).exp() * y", (z + 2).exp() * y)

# 将模型参数初始化在显存上
net = nn.Sequential()
net.add(nn.Dense(1))
net.initialize(ctx=mx.gpu())

print("net(y)", net(y))

print("net[0].weight.data()", net[0].weight.data())