#梯度下降优化方法
#梯度下降是结合梯度和学习率更新权重的优化方法
#W新 = W旧 - 学习率 * 梯度
#存在的问题有：
#1. 遇到平缓区域，梯度下降可能会慢
#2. 遇到鞍点，梯度下降可能会停滞
#3. 遇到局部最优，梯度下降可能会停滞

#EMA指数加权平均
#β(调节权重系数)，值越大越依赖指数加权平均，越不依赖本地梯度值，数据越平缓

#具体的优化方法:
#动量法(Momentum) , AdaGrad , RMSProp , Adam
#动量法调整梯度，AdaGrad和RMSProp调整学习率，Adam结合了动量法和RMSProp的优点


#动量法：St = β * St-1 + (1-β) * Gt
#使用动量法后 梯度更新：W新 = W旧 - 学习率 * St
#缺点是可能导致在鞍点附近震荡，无法收敛到最优解

#AdaGrad : St = St-1 + Gt^2
#学习率 = 学习率 / (sqrt(St) + ε)   ε：1e - 10 防止分母成零
#使用AdaGrad后 梯度更新：W新 = W旧 - 调整后的学习率 * 本次的梯度
#缺点是可能导致学习率过早过量的降低，导致后期学习率太小

#RMSProp : St = β * St-1 + (1-β) * Gt^2
#对AdaGrad的改进。加入调和权重系数
#学习率 = 学习率 / (sqrt(St) + ε)
#使用RMSProp后 梯度更新：W新 = W旧 - 调整后的学习率 * 本次的梯度
#通过引入β，控制历史梯度 对历史梯度信息获取的多少

#Adam
#一阶矩：算均值
#Mt = β1 * Mt-1 + (1-β1) * Gt 充当梯度
#St = β2 * St-1 + (1-β2) * Gt^2 充当梯度的方差
#二阶矩：梯度的方差
#Mt^ = Mt / (1-β1^t) 充当梯度
#St^ = St / (1-β2^t) 充当梯度的方差
#使用Adam后 梯度更新：W新 = W旧 - 学习率 / (sqrt(St^) + ε) * Mt^


import torch
import torch.nn as nn

#动量法(Momentum)
def dm01_momentum():
    #初始化权重
    w = torch.tensor([1.0],requires_grad = True,dtype = torch.float32)
    #定义损失函数
    criterion = (w ** 2) / 2.0
    #创建优化器
    #基于SGD，加入参数momentum
    #参数1：待优化的参数列表，参数2：学习率，参数3：动量系数
    optimizer = torch.optim.SGD(params=[w], lr=0.01, momentum=0.9)
    #梯度清零 + 反向传播 + 参数更新
    optimizer.zero_grad()  #梯度清零
    criterion.backward()   #反向传播
    optimizer.step()       #参数更新

    print(f'w:{w},w.grad:{w.grad}')

    #第二次
    criterion = (w ** 2) / 2.0
    optimizer.zero_grad()  #梯度清零
    criterion.backward()   #反向传播
    optimizer.step()       #参数更新

    print(f'w:{w},w.grad:{w.grad}')

#AdaGrad
def dm02_adagrad():
    #初始化权重
    w = torch.tensor([1.0],requires_grad = True,dtype = torch.float32)
    #定义损失函数
    criterion = (w ** 2) / 2.0
    #创建优化器
    #基于Adagrad
    #参数1：待优化的参数列表，参数2：学习率
    optimizer = torch.optim.Adagrad(params=[w], lr=0.01)
    #梯度清零 + 反向传播 + 参数更新
    optimizer.zero_grad()  #梯度清零
    criterion.backward()   #反向传播
    optimizer.step()       #参数更新

    print(f'w:{w},w.grad:{w.grad}')

    #第二次
    criterion = (w ** 2) / 2.0
    optimizer.zero_grad()  #梯度清零
    criterion.backward()   #反向传播
    optimizer.step()       #参数更新

    print(f'w:{w},w.grad:{w.grad}')

#RMSProp
def dm03_rmsprop():
    #初始化权重
    w = torch.tensor([1.0],requires_grad = True,dtype = torch.float32)
    #定义损失函数
    criterion = (w ** 2) / 2.0
    #创建优化器
    #基于RMSProp
    #参数1：待优化的参数列表，参数2：学习率，参数3：动量系数
    optimizer = torch.optim.RMSprop(params=[w], lr=0.01, alpha=0.9)
    #梯度清零 + 反向传播 + 参数更新
    optimizer.zero_grad()  #梯度清零
    criterion.backward()   #反向传播
    optimizer.step()       #参数更新

    print(f'w:{w},w.grad:{w.grad}')

    #第二次
    criterion = (w ** 2) / 2.0
    optimizer.zero_grad()  #梯度清零
    criterion.backward()   #反向传播
    optimizer.step()       #参数更新

    print(f'w:{w},w.grad:{w.grad}')

#Adam
def dm04_adam():
    #初始化权重
    w = torch.tensor([1.0],requires_grad = True,dtype = torch.float32)
    #定义损失函数
    criterion = (w ** 2) / 2.0
    #创建优化器
    #基于Adam
    #参数1：待优化的参数列表，参数2：学习率，betas = (梯度用的，学习率用的)
    optimizer = torch.optim.Adam(params=[w], lr=0.01, betas=(0.9, 0.999))
    #梯度清零 + 反向传播 + 参数更新
    optimizer.zero_grad()  #梯度清零
    criterion.backward()   #反向传播
    optimizer.step()       #参数更新

    print(f'w:{w},w.grad:{w.grad}')

    #第二次
    criterion = (w ** 2) / 2.0
    optimizer.zero_grad()  #梯度清零
    criterion.backward()   #反向传播
    optimizer.step()       #参数更新

    print(f'w:{w},w.grad:{w.grad}')
#测试
if __name__ == '__main__':
    dm01_momentum()
    dm02_adagrad()
    dm03_rmsprop()
    dm04_adam()