from mxnet import nd
from time import time

# 生成一个1000维的向量a
a = nd.ones(shape = 1000)

# 生成一个1000维的向量b
b = nd.ones(shape = 1000)

start = time()

# 生成一个1000维度的向量c
c = nd.zeros(shape = 1000)

# 向量相加的方法1
for i in range(1000):
    c[i] = a[i] + b[i]
print(time() - start)

# 向量相加的方法2，直接做矢量的加法，此方法比前一个方法快很多
start = time()
d = a + b
print(time() - start)