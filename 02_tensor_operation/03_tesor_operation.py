#张量的点乘和矩阵相乘
#点乘：t1 * t2   或者   t1.mul(t2)
#矩阵相乘：t1 @ t2  或者   t1.matmul(t2)
#点乘要求行列数相同，矩阵相乘要求A列等于B行


import torch
def dm01():
    t1 = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
    t2 = torch.tensor([[1,2,3],[4,5,6],[7,8,9]])
    t3 = t1 * t2
    t4 = t1.mul(t2)
    print(f't1:{t1}')
    print(f't2:{t2}')
    print(f't3:{t3}')
    print(f't4:{t4}')

def dm02():
    t1 = torch.tensor([[1,2,3],[4,5,6]])
    t2 = torch.tensor([[1,2],[3,4],[5,6]])
    t3 = t1 @ t2
    t4 = t1.matmul(t2)
    print(f't1:{t1}')
    print(f't2:{t2}')
    print(f't3:{t3}')
    print(f't4:{t4}')
    


#测试函数：
if __name__ == '__main__':
    dm01()
    dm02()