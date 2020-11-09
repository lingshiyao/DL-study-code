from mxnet import nd

X = nd.ones((8, 8))
for i in range(0, 4):
    for j in range(4 + i, X.shape[1]):
        X[i, j] = 0

for i in range(4, X.shape[0]):
    for j in range(0, i - 4):
        X[i, j] = 0

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