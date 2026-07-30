#神经网络的搭建
#输入层->隐藏层->输出层
#隐藏层中的神经元个数可以自定义，但是输入层和输出层的神经元个数必须是固定的
#神经元通过加权求和和激活函数得到输出
#前向传播依次从输入层到输出层，然后通过损失函数和反向传播更新权重
#常用的激活函数有：relu、sigmoid、tanh、softmax
#参数初始化的常用方法：xavier_normal、xavier_uniform、kaiming_normal、kaiming_uniform、全零(用来初始化偏置b)
#xavier和sigmoid搭配，kaiming和tanh搭配

#深度学习案例的四个步骤：
#1.准备数据 2.搭建神经网络 3.模型训练 4. 模型测试
#神经网络搭建流程
#1.定义一个继承nn.Module的类 2.在__init__方法中定义网络结构 3.在forward方法中定义前向传播

import torch
import torch.nn as nn
from torchsummary import summary


#搭建神经网络，自定义继承nn.Module的类
class ModuleDemo(nn.Module):
    #在init魔法方法中，定义神经网络的结构
    def __init__(self):
        #初始化父类成员
        super().__init__()
        #定义网络结构
        #隐藏层1：3个神经元，激活函数为sigmoid
        self.linear1 = nn.Linear(3,3)#输入特征维度为3，输出特征维度为3
        #隐藏层2：3个神经元，激活函数为relu
        self.linear2 = nn.Linear(3,3)#输入特征维度为3，输出特征维度为3
        #输出层：2个神经元，激活函数为softmax
        self.output = nn.Linear(3,2)#输入特征维度为3，输出特征维度为2
        
        #对隐藏层进行参数初始化
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.zeros_(self.linear1.bias)
        nn.init.kaiming_normal_(self.linear2.weight)
        nn.init.zeros_(self.linear2.bias)
        nn.init.kaiming_normal_(self.output.weight)
        nn.init.zeros_(self.output.bias)
    
    #在forward方法中定义前向传播
    def forward(self,x):
        #加权求和 + 激活函数
        x = torch.sigmoid(self.linear1(x))
        x = torch.relu(self.linear2(x))
        x = torch.softmax(self.output(x),dim = 1)
        #dim = 1,只在列维度上索引移动，因此将行一行一行进行softmax
        return x

#模型训练
def train():
    #创建模型对象
    my_model = ModuleDemo()
    print(f'my_model:{my_model}')

    #创建数据集样本
    x = torch.randn(4,3)
    print(f'x:{x}')
    print(f'x.shape:{x.shape}')
    

    #前向传播
    output = my_model(x)
    print(f'output:{output}')
    print(f'output.shape:{output.shape}')

    #计算 和 查看模型参数
    summary(my_model,(4,3),device = 'cpu')
    for name,param in my_model.named_parameters():
        print(f'name:{name},param:{param}')
        print(f'name:{name},param.shape:{param.shape}')



if __name__ == '__main__':
    train()
