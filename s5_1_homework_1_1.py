from mxnet import nd

X = nd.ones((8, 8))
X[2:6,:] = 0
print(X)

K = nd.array([[1], [-1]])
print(K)

def corr2d(X, K):
    h, w = K.shape
    Y = nd.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i: i + h, j: j + w] * K).sum()
    return Y

print(corr2d(X, K))