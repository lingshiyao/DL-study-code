from mxnet import nd
from mxnet.gluon import nn

# ---------------------------------------------------------
# def fun(*args, ** kwargs):
#     print('args=', args)
#     print('kwargs=', kwargs)

# # args= (1, 2, 3, 4)
# # kwargs= {'A': 'a', 'B': 'b', 'C': 'c', 'D': 'd'}
# fun(1, 2, 3, 4, A='a', B='b', C='c', D='d')
# ---------------------------------------------------------

# MLP继承自nn.Block（python class 继承）
# Block是一个通用部件，Sequential是从Block继承的

# ```py
# net = nn.Sequential()
# ```

# ------------------- __init__ Demo -------------------
class Person:
    ## __init__并不是构造函数
    def __init__(self, name):
        self.name = name
    def sayHi(self):
        print("Hello", "my name is", self.name)

# 这句话可以理解为
# ```py
# a=object.__new__(A)
# A.__init__(a,'hello')
# ```
p = Person("Swaroop")
p.sayHi()

class MLP(nn.Block):

    # 声明带有模型参数的层，这里声明了两个全连接层
    def __init__(self, **kwargs):

        # 调用MLP父类Block的构造函数来进行必要的初始化。这样在构造实例时还可以指定其他函数
        # 参数，如“模型参数的访问、初始化和共享”一节将介绍的模型参数params
        super(MLP, self).__init__(**kwargs)

        # 隐藏层
        self.hidden = nn.Dense(256, activation='relu')

        # 输出层
        self.output = nn.Dense(10)

    # 定义模型的前向计算，即如何根据输入x计算返回所需要的模型输出
    def forward(self, x):
        return self.output(self.hidden(x))

# 2 x 20
X = nd.random.uniform(shape=(2, 20))
net = MLP()
net.initialize()
print(net(X))

# ---------------------------------------------------------W
# OrderedDict
# ---------------------------------------------------------W
# 可以顺序打印当前的项目
import collections

my_order_dict = collections.OrderedDict()
my_order_dict["name"] = "lowman"
my_order_dict["age"] = 45
my_order_dict["money"] = 998
my_order_dict["hourse"] = None

for key, value in my_order_dict.items():
    print(key, value)
# ---------------------------------------------------------M
# OrderedDict
# ---------------------------------------------------------M

class MySequential(nn.Block):
    def __init__(self, **kwargs):
        super(MySequential, self).__init__(**kwargs)

    def add(self, block):
        # block是一个Block子类实例，假设它有一个独一无二的名字。我们将它保存在Block类的
        # 成员变量_children里，其类型是OrderedDict。当MySequential实例调用
        # initialize函数时，系统会自动对_children里所有成员初始化
        self._children[block.name] = block

    def forward(self, x):
        # OrderedDict保证会按照成员添加时的顺序遍历成员
        for block in self._children.values():
            x = block(x)
        return x

# new 的时候会调用__init__
net = MySequential()
net.add(nn.Dense(256, activation='relu'))
net.add(nn.Dense(10))
net.initialize()
net(X)

class FancyMLP(nn.Block):
    def __init__(self, **kwargs):
        super(FancyMLP, self).__init__(**kwargs)
        # 使用get_constant创建的随机权重参数不会在训练中被迭代（即常数参数）
        self.rand_weight = self.params.get_constant(
            'rand_weight', nd.random.uniform(shape=(20, 20)))
        self.dense = nn.Dense(20, activation='relu')

    def forward(self, x):
        x = self.dense(x)
        # 使用创建的常数参数，以及NDArray的relu函数和dot函数
        x = nd.relu(nd.dot(x, self.rand_weight.data()) + 1)
        # 复用全连接层。等价于两个全连接层共享参数
        x = self.dense(x)
        # 控制流，这里我们需要调用asscalar函数来返回标量进行比较
        while x.norm().asscalar() > 1:
            x /= 2
        if x.norm().asscalar() < 0.8:
            x *= 10
        return x.sum()

net = FancyMLP()
net.initialize()
net(X)

class NestMLP(nn.Block):
    def __init__(self, **kwargs):
        super(NestMLP, self).__init__(**kwargs)
        self.net = nn.Sequential()
        self.net.add(nn.Dense(64, activation='relu'),
                     nn.Dense(32, activation='relu'))
        self.dense = nn.Dense(16, activation='relu')

    def forward(self, x):
        return self.dense(self.net(x))

net = nn.Sequential()
net.add(NestMLP(), nn.Dense(20), FancyMLP())

net.initialize()
net(X)