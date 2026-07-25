#张量的拼接操作
#cat() 不改变维度，(2,3) + (2,3) = (4,3),除了被拼接的那个维度外，其他维度必须一致
#stack() 改变维度，所有维度必须一致

import torch

torch.manual_seed(100)
t1 = torch.randint(0,10,(2,3))
t2 = torch.randint(0,10,(5,3))
t3 = torch.randint(0,10,(2,3))
print(f't1:{t1},shape:{t1.shape}')
print(f't2:{t2},shape:{t2.shape}')

t4 = torch.cat((t1,t2),dim = 0)#(2,3) + (5,3) = (7,3)
print(f't4:{t4},shape:{t4.shape}')

t5 = torch.stack((t1,t3),dim = 0)#dim = 0意为在第零维度上进行拼接，变成(2,2,3)
print(f't5:{t5},shape:{t5.shape}')

t6 = torch.stack((t1,t3),dim = 1)#dim = 1意为在第一维度上进行拼接，变成(2,2,3)
print(f't6:{t6},shape:{t6.shape}')

t7 = torch.stack((t1,t3),dim = 2)#dim = 2意为在第二维度上进行拼接，变成(2,3,2)
print(f't7:{t7},shape:{t7.shape}')



