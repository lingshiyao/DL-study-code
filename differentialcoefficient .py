import mxnet.ndarray as nd
import mxnet.autograd as ag
x = nd.array([[1, 2], [3, 4]])
x.attach_grad()
with ag.record():
    y = x * 2
z = y * x
y.backward()
print(x.grad)
x.grad == 4*x

print('-' * 40)

# 求梯度的简单例子，求导后把参数带进去
x = nd.array([100, 100, 100])
x.attach_grad()
with ag.record():
    y = x * x * x
y.backward()
print(x.grad)
