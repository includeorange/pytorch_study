#一些常见的张量运算函数
#sum() max() min() mean()求和 求最大 求最小 求平均值 这四个有dim参数，指定维度，默认是所有元素
#pow() sqrt() exp() log() log2() log10() 没有dim，逐个计算
#pow(n)是n次幂，sqrt()是平方根，exp()是e的指数，log()是以e为底的对数，log2()是以2为底的对数，log10()是以10为底的对数
#pow 可以用 ** 替代，t1.pow(3) 等价于 t1 ** 3


import torch
t1 = torch.tensor([[1,2,3],[4,5,6]],dtype = torch.float32)
print(t1.sum(dim = 0))#按列求和
print(t1.sum(dim = 1))#按行求和
print(t1.sum())#所有元素求和
print('-------------------')
print(t1.max(dim = 0))#按列求最大值
print(t1.max(dim = 1))#按行求最大值
print(t1.max())#所有元素求最大值
print('-------------------')

#mean要的必须是float类型的张量，int类型的张量不能求平均值
print(t1.mean(dim = 0))#按列求平均值
print(t1.mean(dim = 1))#按行求平均值
print(t1.mean())#所有元素求平均值
print('-------------------')



