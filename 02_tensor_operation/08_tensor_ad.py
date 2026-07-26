#自动微分的流程演示
#其中损失函数仅仅举例用以演示过程

#权重更新 ： w新 = w旧 - 学习率 * 梯度
#梯度计算：梯度 = 损失函数的导数

#自动微分模块，专门实现求导，结合反向传播，更新权重w 和 偏置b
##只有标量张量才能求导，且底层一般是浮点型

#可以进行自动微分的张量(requires_grad = True),无法转换成ndarray
#但是可以通过detach()拷贝一个不进行自动微分的张量
#可以通过detach().numpy()拷贝一个不进行自动微分的ndarray
#这个和原来的张量是共享内存的


import torch
#dm01用来演示一次反向传播更新权重的过程
def dm01():
    w = torch.tensor(10,requires_grad = True,dtype = torch.float32)
    #只有标量张量才能求导，且底层一般是浮点型
    #requires_grad = True 表示可以进行自动微分
    loss = 2 * w ** 2 #实际是2w^2

    loss.sum().backward() #反向传播，计算梯度
    #实际应用中，会有很多数据，因为backword只能对标量张量进行求导，因此通过sum将所有数据合并成一个标量

    w.data = w.data - 0.01 * w.grad #权重更新,设定学习率为0.01

    print(w)

#dm02用来演示多次反向传播更新权重的过程
def dm02():
    w = torch.tensor(10,requires_grad = True,dtype = torch.float32)
    loss = w ** 2 + 20 #实际是w^2 + 20

    print(f'w开始时的权重是{w},loss = {loss}')
    for i in range (1,101):
        #正向传播：
        loss = w ** 2 + 20

        #梯度清零
        #因为梯度默认累加，所以要清零
        if w.grad is not None:
            w.grad.zero_()
        
        #反向传播
        loss.sum().backward()
        
        #更新权重
        w.data = w.data - 0.01 * w.grad

        print(f'第{i}次迭代后,w的权重是{w},梯度: {w.grad},loss = {loss}')


#dm03用来演示detach()函数的使用
def dm03():
    t1 = torch.tensor(10,requires_grad = True,dtype = torch.float32)
    n2 = t1.detach().numpy() #拷贝一个不进行自动微分的ndarray
    n2[...] = 100
    print(f't1:{t1},type:{type(t1)}')
    print(f'n2:{n2},type:{type(n2)}')


#测试函数
if __name__ == '__main__':
    dm01()
    dm02()
    dm03()

