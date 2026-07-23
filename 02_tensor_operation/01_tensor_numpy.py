#tensor和ndarray的相互装换以及标量张量和标量的相互装换

#tensor -> ndarray
#tensor.numpy()   共享内存
#tensor.numpy().copy()   不共享内存

#ndarray -> tensor
#torch.from_numpy(ndarray)   共享内存
#torch.tensor(ndarray)   不共享内存

#标量张量 -> 标量
#tensor.item()   共享内存 括号里必须是标量张量

import torch
import numpy as np

def dm01():
    #tensor -> ndarray
    t1 = torch.tensor([1,2,3],dtype = torch.int)
    print(f't1:{t1}')
    n1 = t1.numpy() #共享内存
    n2 = t1.numpy().copy() #不共享内存
    n1[0] = 100
    print(f't1:{t1}')
    print(f'n1:{n1}')
    print(f'n2:{n2}')

def dm02():
    #ndarray -> tensor
    n1 = np.array([1,2,3])
    t1 = torch.from_numpy(n1) #共享内存
    t2 = torch.tensor(n1) #不共享内存
    t1[0] = 100
    print(f't1:{t1}')
    print(f't2:{t2}')
    print(f'n1:{n1}')


def dm03():
    #标量张量 -> 标量
    t1 = torch.tensor([1,])
    a = t1.item() 
    print(f't1:{t1},type:{type(t1)}')
    print(f'a:{a},type:{type(a)}')

#测试函数
if __name__ == '__main__':
    dm01()
    dm02()
    dm03()
