'''
损失函数介绍：
    分类：
        分类问题：
            多分类交叉熵损失：CrossEntropyLoss
            二分类交叉熵损失：BCELoss
        回归问题：
            MAE：Mean Absolute Error 平均绝对误差
            MSE：Mean Squared Error 均方误差
            RMSE：Root Mean Squared Error 均方根误差
            Smooth L1：结合上述MAE和MSE的损失函数的升级优化
'''
#多分类任务交叉熵损失函数
#设计思路 Loss = - sum(y * log(Softmax(f(x))))
#x:样本 f(x)：加权求和 Softmax(f(x)):处理后的概率 
#y：样本x属于某一个类别的真实概率
#这个损失函数内里包含softmax ，后续使用，输出层不需要额外调用softmax

#二分类任务交叉熵损失函数
#设计思路 Loss = -ylog(预测值) - (1-y)log(1-预测值)
#公式中没有包含sigmoid激活函数，使用时还需手动指定Sigmoid

#MAE : Mean Absolute Error 平均绝对误差
#公式：误差绝对值之和 / 样本总数 类似与L1正则化，权重可以降为零，数据会变得稀疏
#弊端：在零点不平滑，可能错过最小值

#MSE : Mean Squared Error 均方误差
#公式：误差平方之和 / 样本总数
#弊端：如果插值过大，可能存在梯度爆炸的情况

#Smooth L1：结合上述MAE和MSE的损失函数的升级优化
#在[-1,1]是MSE，其他是L1
#解决了L1不平滑和L2梯度爆炸的问题



import torch
import torch.nn as nn

#多分类任务交叉熵损失
def dm01():
    #手动创建样本真实值和预测值
    y_true = torch.tensor([[0,1,0],[0,0,1],[1,0,0]],dtype = torch.float32)
    y_pred = torch.tensor([[0.7,0.1,0.2],[0.1,0.7,0.2],[0.1,0.2,0.7]],requires_grad = True,dtype = torch.float32)
    #计算交叉熵损失
    criterion = nn.CrossEntropyLoss()
    loss = criterion(y_pred,y_true)
    print(f'loss:{loss}')

#二分类任务交叉熵损失
def dm02():
    #手动创建样本真实值和预测值
    y_true = torch.tensor([0,1,0],dtype = torch.float32)
    y_pred = torch.tensor([0.7154,0.3135,0.1354],requires_grad = True,dtype = torch.float32)
    #计算交叉熵损失
    criterion = nn.BCELoss()
    loss = criterion(y_pred,y_true)
    print(f'loss:{loss}')

#MAE
def dm03():
    #手动创建样本真实值和预测值
    y_true = torch.tensor([2.0,2.0,2.0],dtype = torch.float32)
    y_pred = torch.tensor([1.0,1.0,1.9],requires_grad = True,dtype = torch.float32)
    #计算MAE
    criterion = nn.L1Loss()
    loss = criterion(y_pred,y_true)
    print(f'loss:{loss}')

#MSE
def dm04():
    #手动创建样本真实值和预测值
    y_true = torch.tensor([1,2,3],dtype = torch.float32)
    y_pred = torch.tensor([1.5,2.5,3.5],requires_grad = True,dtype = torch.float32)
    #计算MSE
    criterion = nn.MSELoss()
    loss = criterion(y_pred,y_true)
    print(f'loss:{loss}')

#Smooth L1
def dm05():
    #手动创建样本真实值和预测值
    y_true = torch.tensor([1,2,3],dtype = torch.float32)
    y_pred = torch.tensor([1.5,2.5,3.5],requires_grad = True,dtype = torch.float32)
    #计算Smooth L1
    criterion = nn.SmoothL1Loss()
    loss = criterion(y_pred,y_true)
    print(f'loss:{loss}')

if __name__ == '__main__':
    dm01()
    dm02()
    dm03()
    dm04()
    dm05()