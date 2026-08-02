#正则化是为了缓解过拟合

#L1正则化 ： 权重可以变成零
#L2正则化 ： 权重可以无限接近零
#Dropout ： 随机失活
#BN

#dropout让神经元以超参数p的概率停止工作或者激活被置为0，未被置为零的进行缩放，缩放比例是1/(1-p)
#实际应用中p通常取0.2到0.5
#较小的模型或较复杂的任务，可以选0.3或者更小
#非常深的网络，选较大(0.5 0.6)
#实际应用中，会在全连接层激活函数后添加dropout层


import torch
import torch.nn as nn


#dropout
def dm01():
    #创建隐藏层输入结果
    t1 = torch.randint(0,10,(1,4),dtype = torch.float32)

    #加权求和 激活函数
    Linear1 = nn.Linear(4,5)
    l1 = Linear1(t1)
    print(f'l1:{l1}')

    output = torch.relu(l1)
    print(f'output:{output}')

    #dropout
    dropout = nn.Dropout(p = 0.5)#0.5的概率让神经元停止工作，继续工作的神经元的激活值会被缩放为原来的1/(1-0.5) = 2倍
    output = dropout(output)
    print(f'output:{output}')





#测试
if __name__ == '__main__':
    dm01()
