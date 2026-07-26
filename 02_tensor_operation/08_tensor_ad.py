#自动微分的流程

#权重更新 ： w新 = w旧 - 学习率 * 梯度
#梯度计算：梯度 = 损失函数的导数

#自动微分模块，专门实现求导，结合反向传播，更新权重w 和 偏置b
##只有标量张量才能求导，且底层一般是浮点型

#可以进行自动微分的张量(requires_grad = True),无法转换成ndarray
#但是可以通过detach()拷贝一个不进行自动微分的张量
#可以通过detach().numpy()拷贝一个不进行自动微分的ndarray
#这个和原来的张量是共享内存的

#dm01用来演示一次反向传播更新权重的过程,dm02用来演示多次反向传播更新权重的过程
#dm03用来演示detach()函数的使用,dm04用来演示一个自动微分的实际应用过程

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

#dm04用来演示一个自动微分的实际应用过程
def dm04():
    #定义x，假设是2行5列的全一张量(表示：特征)
    #定义y，假设是3行6列的全零张量(表示:标签(真实值))
    x = torch.ones(2,5)
    y = torch.zeros(2,6)
    print(f'x:{x}')
    print(f'y:{y}')


    #初始化可自动微分的权重和偏置
    #z是预测值   z = x @ w + b
    #因此权重w是五行六列，偏置b是6个
    #因为x是2行5列，y是2行6列，x @ w = 2行6列，所以w应该是5行6列
    #b与输出特征数绑定
    w = torch.randn(5,6,requires_grad = True)
    b = torch.randn(6,requires_grad = True)
    print(f'w:{w}')
    print(f'b:{b}')
    #前向传播
    z = x @ w + b
    print(f'z:{z}')

    #损失函数
    criterion = torch.nn.MSELoss()
    loss = criterion(z,y)

    #反向传播
    loss.sum().backward()

    #后续就是w = w - 学习率 * 梯度    b = b - 学习率 * 梯度
    #更新权重和偏置
    w.data = w.data - 0.01 * w.grad
    b.data = b.data - 0.01 * b.grad
    print(f'w:{w},b:{b}')




#测试函数
if __name__ == '__main__':
    dm01()
    dm02()
    dm03()
    dm04()

