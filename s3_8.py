import d2lzh as d2l
from mxnet import autograd, nd

## ReLU-----线性整流函数-----（Rectified Linear Unit）

def xyplot(x_vals, y_vals, name):
    d2l.set_figsize(figsize=(5, 2.5))
    d2l.plt.plot(x_vals.asnumpy(), y_vals.asnumpy())
    d2l.plt.xlabel('x')
    d2l.plt.ylabel(name + '(x)')
    d2l.plt.show()

## <NDArray 160 @cpu(0)>
## -8到8每隔0.1加上0.1，一共160个数据
x = nd.arange(-8.0, 8.0, 0.1)

## x生成梯度
x.attach_grad()

# 定义模型
with autograd.record():
    y = x.relu()

# 显示relu曲线
# xyplot(x, y, 'relu')

# 计算梯度
y.backward()

# grad---梯度的意思
# xyplot(x, x.grad, 'grad of relu')

#---------------------------grad review---------------------------
# TTTX = nd.array([3])
# TTTX.attach_grad()
# with autograd.record():
#     TTTY = 3 * TTTX**3
# TTTY.backward()
# print(TTTX.grad)
#---------------------------grad review---------------------------

# -----sigmoid
# 1 / (1 + e**(-z))

with autograd.record():
    y = x.sigmoid()
# xyplot(x, y, 'sigmoid')

y.backward()
# xyplot(x, x.grad, 'grad of sigmoid')

with autograd.record():
    y = x.tanh()
# xyplot(x, y, 'tanh')

y.backward()
# xyplot(x, x.grad, 'grad of tanh')