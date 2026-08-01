#学习率优化 -> 学习率衰减策略 -> 手动控制学习率的调整
#学习率越小，模型收敛越慢，越大，收敛越快，但是过大会出现震荡，甚至不收敛(梯度爆炸)

#用的不多，一般会用Adam而非手动调节学习率

#学习率衰减策略：
#等间隔学习率衰减   指定间隔学习率衰减  指数学习率衰减

#***************************************
#等间隔用于大型数据集较为简单的任务，学习率变化大，可能跳过最优点
#制定间隔用于队训练平稳性要求较高的任务，易于调试
#指数用于高精度训练避免过快收敛，平滑手链稳定性强，但是需要更多训练资源
#***************************************

#等间隔学习率衰减:
#step_size:间隔的轮数 , 多少轮调整一次
#gamma:衰减系数 lr新 = lr旧 * gamma

#指定间隔学习率衰减:
#milestones:指定的轮数列表,在这些轮数上调整学习率
#gamma:衰减系数 lr新 = lr旧 * gamma

#指数学习率衰减:
#lr新 = lr旧 * gamma ** epoch
#前期学习率衰减快，中期慢后期更慢

import torch
from torch import optim
import matplotlib.pyplot as plt

#等间隔学习率衰减
def dm01():
    #初始化学习率，训练轮数和每轮的批次
    lr,epochs,iteration = 0.1,200,10

    #创建数据集
    y_true = torch.tensor([0])
    x = torch.tensor([1.0],dtype = torch.float32)
    w = torch.tensor([1.0],requires_grad = True,dtype = torch.float32)

    #创建优化器，用动量法
    optimizer = optim.SGD(params=[w],lr = lr,momentum = 0.9)

    #创建学习率衰减器
    #参数一：优化器，参数二：间隔轮数，参数三：衰减系数
    scheduler = optim.lr_scheduler.StepLR(optimizer,step_size = 50,gamma = 0.5)

    #创建两个列表，分别表示训练轮数和每轮的学习率
    epoch_list,lr_list = [],[]
    
    #训练
    for epoch in range(epochs):
        epoch_list.append(epoch)
        lr_list.append(scheduler.get_last_lr())  #获取当前学习率
        #每轮训练
        for i in range(iteration):
            #计算损失
            y_pred = w * x
            loss = (y_pred - y_true) ** 2 

            #梯度清零 + 反向传播 + 参数更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #学习率衰减器更新学习率
        scheduler.step()

    #绘制学习率曲线
    plt.plot(epoch_list,lr_list)
    plt.xlabel('epoch')
    plt.ylabel('learning rate')
    plt.title('Learning Rate Decay Curve')
    plt.legend()
    plt.show()

#指定间隔学习率衰减
def dm02():
    #初始化学习率，训练轮数和每轮的批次
    lr,epochs,iteration = 0.1,200,10

    #创建数据集
    y_true = torch.tensor([0])
    x = torch.tensor([1.0],dtype = torch.float32)
    w = torch.tensor([1.0],requires_grad = True,dtype = torch.float32)

    #创建优化器，用动量法
    optimizer = optim.SGD(params=[w],lr = lr,momentum = 0.9)

    #创建学习率衰减器
    #参数一：优化器，参数二：要修改学习率的轮数(指定)，参数三：衰减系数
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50, 75, 150], gamma=0.5)

    #创建两个列表，分别表示训练轮数和每轮的学习率
    epoch_list,lr_list = [],[]
    
    #训练
    for epoch in range(epochs):
        epoch_list.append(epoch)
        lr_list.append(scheduler.get_last_lr())  #获取当前学习率
        #每轮训练
        for i in range(iteration):
            #计算损失
            y_pred = w * x
            loss = (y_pred - y_true) ** 2 

            #梯度清零 + 反向传播 + 参数更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #学习率衰减器更新学习率
        scheduler.step()

    #绘制学习率曲线
    plt.plot(epoch_list,lr_list)
    plt.xlabel('epoch')
    plt.ylabel('learning rate')
    plt.title('Learning Rate Decay Curve')
    plt.legend()
    plt.show()

#指数学习率衰减
def dm03():
    #初始化学习率，训练轮数和每轮的批次
    lr,epochs,iteration = 0.1,200,10

    #创建数据集
    y_true = torch.tensor([0])
    x = torch.tensor([1.0],dtype = torch.float32)
    w = torch.tensor([1.0],requires_grad = True,dtype = torch.float32)

    #创建优化器，用动量法
    optimizer = optim.SGD(params=[w],lr = lr,momentum = 0.9)

    #创建学习率衰减器
    #参数一：优化器，参数二：衰减系数
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer,gamma = 0.95)

    #创建两个列表，分别表示训练轮数和每轮的学习率
    epoch_list,lr_list = [],[]
    
    #训练
    for epoch in range(epochs):
        epoch_list.append(epoch)
        lr_list.append(scheduler.get_last_lr())  #获取当前学习率
        #每轮训练
        for i in range(iteration):
            #计算损失
            y_pred = w * x
            loss = (y_pred - y_true) ** 2 

            #梯度清零 + 反向传播 + 参数更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #学习率衰减器更新学习率
        scheduler.step()

    #绘制学习率曲线
    plt.plot(epoch_list,lr_list)
    plt.xlabel('epoch')
    plt.ylabel('learning rate')
    plt.title('Learning Rate Decay Curve')
    plt.legend()
    plt.show()

#测试
if __name__ == '__main__':
    #dm01()
    #dm02()
    dm03()




