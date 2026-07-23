#生成线性或者随机张量
#torch.arange(start, end, step) 生成线性张量
#torch.linspace(start, end, steps) 生成线性张量
#随机种子设置：
#torch.random.initial_seed() 根据时间戳产生种子
#torch.random.manual_seed(seed)  设置CPU随机种子
#创建随机浮点数张量
#torch.rand(*size) 生成随机张量
#torch.randn(*size) 生成正态分布随机张量
#创建随机整数张量
#torch.randint(low, high, size) 生成指定范围的随机整数张量
import torch
def dm01():
    #生成线性张量
    t1 = torch.arange(0,10,2)#从零到十，步长为2(包左不包右)
    print(f't1:{t1},type:{type(t1)}')
    print('-' * 10)
    t2 = torch.linspace(0,10,5)#从零到十，元素个数是五
    print(f't2:{t2},type:{type(t2)}')
    print('-' * 10)

def dm02():
    #随机种子设置：
    #torch.initial_seed()#默认采用当前系统时间戳作为随机种子
    torch.manual_seed(100)
    #生成随机张量：
    t1 = torch.rand(size(2,3))#两行三列随机张量
    print(f't1:{t1},type:{type(t1)}')
    print('-' * 10)
    t2 = torch.randn(size=(3,5))#两行三列正态分布随机张量
    print(f't2:{t2},type:{type(t2)}')
    print('-' * 10)
    t3 = torch.randint(0,10,size=(2,3))#从零到十两行三列
    print(f't3:{t3},type:{type(t3)}')  
    print('-' * 10)

#测试
if __name__ == '__main__':
    dm01()
    dm02()