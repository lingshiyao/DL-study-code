from mxnet import nd
import random
import zipfile

# with open() as 可以用来打开文件
# 这里可以用with as来读写zip
with zipfile.ZipFile('../data/jaychou_lyrics.txt.zip') as zin:
    with zin.open('jaychou_lyrics.txt') as f:
        corpus_chars = f.read().decode('utf-8')
print(corpus_chars[:40])

# 把换行符替换成空格
corpus_chars = corpus_chars.replace('\n', ' ').replace('\r', ' ')

# 只是用前一万字符来训练模型
corpus_chars = corpus_chars[0:10000]

# 建立字的索引，set会自动去重
# idx to char
idx_to_char = list(set(corpus_chars))

# 创建字典，字典里面每个字都会有自己的索引index
# char to idx
char_to_idx = dict([(char, i) for i, char in enumerate(idx_to_char)])

# 获取字典的长度
vocab_size = len(char_to_idx)

print(vocab_size)

# 把训练数据转换成索引
corpus_indices = [char_to_idx[char] for char in corpus_chars]

# 取出训练数据前20个索引作为例子
sample = corpus_indices[:20]

# 通过索引打印出句子
print('chars:', ''.join([idx_to_char[idx] for idx in sample]))

# 打印出句子的索引
print('indices:', sample)

# 本函数已保存在d2lzh包中方便以后使用

# corpus_indice     传入的索引列表
# batch_size        批次大小            2
# num_steps                           6
def data_iter_random(corpus_indices, batch_size, num_steps, ctx=None):
    # 减1是因为输出的索引是相应输入的索引加1
    # / 浮点数除法
    # // 整数除法

    # 4
    num_examples = (len(corpus_indices) - 1) // num_steps

    # 2
    epoch_size = num_examples // batch_size

    # [0, 1, 2, 3]
    example_indices = list(range(num_examples))

    # [0, 3, 1, 2]
    random.shuffle(example_indices)

    # 返回从pos开始的长为num_steps的序列
    def _data(pos):
        return corpus_indices[pos: pos + num_steps]

    for i in range(epoch_size):
        # 每次读取batch_size个随机样本
        i = i * batch_size

        # [0, 2]
        batch_indices = example_indices[i: i + batch_size]

        #
        X = [_data(j * num_steps) for j in batch_indices]

        Y = [_data(j * num_steps + 1) for j in batch_indices]

        yield nd.array(X, ctx), nd.array(Y, ctx)

# [0, 1, 2, ..., 29]
my_seq = list(range(30))
for X, Y in data_iter_random(my_seq, batch_size=2, num_steps=6):
    print('X: ', X, '\nY:', Y, '\n')

print('-' * 100)

# 本函数已保存在d2lzh包中方便以后使用
def data_iter_consecutive(corpus_indices, batch_size, num_steps, ctx=None):

    # 语料库索引
    corpus_indices = nd.array(corpus_indices, ctx=ctx)

    # 30
    data_len = len(corpus_indices)

    # 15
    batch_len = data_len // batch_size

    # 2 x 15
    indices = corpus_indices[0: batch_size * batch_len].reshape((batch_size, batch_len))

    # 2
    epoch_size = (batch_len - 1) // num_steps

    for i in range(epoch_size):
        i = i * num_steps
        X = indices[:, i: i + num_steps]
        Y = indices[:, i + 1: i + num_steps + 1]
        yield X, Y

for X, Y in data_iter_consecutive(my_seq, batch_size=2, num_steps=6):
    print('X: ', X, '\nY:', Y, '\n')