#创建全零全一和指定值张量
#torch.zeros() 创建全零张量
#torch.zeros_like() 创建与给定张量形状相同的全零张量
#torch.ones() 创建全一张量
#torch.ones_like() 创建与给定张量形状相同的全一张量
#torch.full() 创建指定值张量
#torch.full_like() 创建与给定张量形状相同的指定值张量


import torch
def dm01():
    t1 = torch.zeros(size = (2,3))#两行三列全零张量
    print(f't1:{t1},type:{type(t1)}')
    t2 = torch.tensor([[1,2,3],[4,5,6]])
    print(f't2:{t2},type:{type(t2)}')
    t3 = torch.zeros_like(t2)#与t2形状相同的全零张量
    print(f't3:{t3},type:{type(t3)}')

def dm02():
    t1 = torch.ones(size = (2,3))#两行三列全一张量
    print(f't1:{t1},type:{type(t1)}')
    t2 = torch.tensor([[1,2,3],[4,5,6]])
    print(f't2:{t2},type:{type(t2)}')
    t3 = torch.ones_like(t2)#与t2形状相同的全一张量
    print(f't3:{t3},type:{type(t3)}')

def dm03():
    t1 = torch.full(size = (2,3),fill_value = 255)#两行三列全5张量
    print(f't1:{t1},type:{type(t1)}')
    t2 = torch.tensor([[1,2,3],[4,5,6]])
    print(f't2:{t2},type:{type(t2)}')
    t3 = torch.full_like(t2,fill_value = 255)#与t2形状相同的全9张量
    print(f't3:{t3},type:{type(t3)}')

#测试函数
if __name__ == '__main__':
    dm01()
    dm02()
    dm03()