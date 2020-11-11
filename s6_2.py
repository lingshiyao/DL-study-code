from mxnet import nd

# X         3 x 1
# W_xh      1 x 4
X, W_xh = nd.random.normal(shape=(3, 1)), nd.random.normal(shape=(1, 4))

# H         3 x 4
# W_hh      4 x 4
H, W_hh = nd.random.normal(shape=(3, 4)), nd.random.normal(shape=(4, 4))

# X * W_xh  3 x 1   1 x 4   =   3 x 4
# H * W_hh  3 x 4   4 x 4   =   3 x 4
print(nd.dot(X, W_xh) + nd.dot(H, W_hh))

concat1 = nd.concat(X, H, dim=1)
concat2 = nd.concat(W_xh, W_hh, dim=0)

# 链接之后点乘和之前点乘相加是一样的
print(nd.dot(nd.concat(X, H, dim=1), nd.concat(W_xh, W_hh, dim=0)))