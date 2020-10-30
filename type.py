# ========================================Numbers（数字）
var1 = 1
var2 = 2

# int       （有符号整型）
# 32位       范围为-2**31～2**31-1，即-2147483648～2147483647
# 64位       范围为-2**63～2**63-1，即-9223372036854775808～9223372036854775807

# long      （长整型[也可以代表八进制和十六进制]）
# float     （浮点型）
# complex   （复数）

# ========================================String（字符串）
str = 'Hello World'

print(str)  # 输出完整字符串
print(str[0])  # 输出字符串中的第一个字符
print(str[2:5])  # 输出字符串中第三个至第五个之间的字符
print(str[2:])  # 输出从第三个开始到最后的字符串
print(str * 2)  # 输出字符串两次
print('say: ' + str)  # 输出连接的字符串

# ========================================List（列表）
list = ['apple', 'jack', 798, 2.22, 36]
otherlist = [123, 'xiaohong']

print(list)  # 输出完整列表
print(list[0])  # 输出列表第一个元素
print(list[1:3])  # 输出列表第二个至第三个元素
print(list[2:])  # 输出列表第三个开始至末尾的所有元素
print(otherlist * 2)  # 输出列表两次
print(list + otherlist)  # 输出拼接列表

# ========================================Tuple（元组）
# 更像是张量

# ========================================Dictionary（字典）
dict = {}
dict['one'] = 'This is one'
dict[2] = 'This is two'
tinydict = {'name': 'john', 'code': 5762, 'dept': 'sales'}

print(dict['one'])  # 输出键为'one'的值
print(dict[2])  # 输出键为2的值
print(tinydict)  # 输出完整的字典
print(tinydict.keys())  # 输出所有键
print(tinydict.values())  # 输出所有值
