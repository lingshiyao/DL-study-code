import d2lzh as d2l
from mxnet import autograd, gluon, nd
from mxnet.gluon import data as gdata, loss as gloss, nn

w = nd.array([1]).reshape((1))
x = nd.array([2]).reshape((1))
b = nd.array([3]).reshape((1))

w.attach_grad()
x.attach_grad()
# b.attach_grad()

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#--------------------------------------------!
#--------------------------------------------!
#----------代码梯度的算法，以后一定要看------------!
#--------------------------------------------!
#--------------------------------------------!
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
with autograd.record():
    y = x * w + b
y.backward()

print("w.grad", w.grad)
print("x.grad", x.grad)
print("b.grad", b.grad)

w.attach_grad()
x.attach_grad()
# b.attach_grad()

with autograd.record():
    y = x**2 * w + b
y.backward()

print("w.grad", w.grad)
print("x.grad", x.grad)
print("b.grad", b.grad)