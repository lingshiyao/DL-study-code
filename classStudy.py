# 创建类
class Foo:
    # 创建类中的函数
    def bar(self):
        pass

    def hello(self):
        print("hello")

# 根据Foo创建对象obj
obj = Foo()
obj.bar()
obj.hello()

class FooChild(Foo):
    pass

obj = FooChild()
obj.bar()
obj.hello()
