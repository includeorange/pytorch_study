'''
ANN CNN RNN 底层处理数据时都是使用张量进行处理的
张量 -> 存储同一类型元素的容器，且元素必须是数值才可以

张量的创建方式：
    torch.tensor 根据指定数据创建张量
    torch.Tensor
    torch.IntTensor torch.FloatTensor torch.DoubleTensor

'''
import torch
import numpy as np

#1.定义函数tensor
def def01():
    #标量->张量
    t1 = torch.tensor(10)
    print(f't1:{t1},type:{type(t1)}')
    print('-' * 10)
    #二维列表->张量
    data = [[1,2,3],[4,5,6]]
    t2 = torch.tensor(data)
    print(f't2:{t2},type:{type(t2)}')
    print('-' * 10)
    #numpy nd数组->张量
    data = np.random.randint(0,10,size = (2,3))
    t3 = torch.tensor(data)
    print(f't3:{t3},type:{type(t3)}')   

#2.定义函数Tensor
def def02():
    #标量->张量
    t1 = torch.Tensor(10)
    print(f't1:{t1},type:{type(t1)}')
    print('-' * 10)
    #二维列表->张量
    data = [[1,2,3],[4,5,6]]
    t2 = torch.Tensor(data)
    print(f't2:{t2},type:{type(t2)}')
    print('-' * 10)
    #numpy nd数组->张量
    data = np.random.randint(0,10,size = (2,3))
    t3 = torch.Tensor(data)
    print(f't3:{t3},type:{type(t3)}') 
    #直接创建张量
    t4= torch.Tensor(2,3)
    print(f't4:{t4},type:{type(t4)}')
#Tensor方式可以制定维度直接创建张量  

#3.定义函数torch.IntTensor torch.FloatTensor torch.DoubleTensor
def def03():
    #标量->张量
    t1 = torch.IntTensor(10)
    print(f't1:{t1},type:{type(t1)}')
    print('-' * 10)
    #二维列表->张量
    data = [[1,2,3],[4,5,6]]
    t2 = torch.IntTensor(data)
    print(f't2:{t2},type:{type(t2)}')
    print('-' * 10)
    #numpy nd数组->张量
    data = np.random.randint(0,10,size = (2,3))
    t3 = torch.IntTensor(data)
    print(f't3:{t3},type:{type(t3)}') 
    #如果类型不匹配，会尝试自动转换
    data = np.random.randint(0,10,size = (2,3))
    t4 = torch.FloatTensor(data)
    print(f't4:{t4},type:{type(t4)}') 

#测试函数
if __name__ == '__main__':
    #def01()
    def02()
    #def03()
