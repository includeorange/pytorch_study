#正则化是为了缓解过拟合

#L1正则化 ： 权重可以变成零
#L2正则化 ： 权重可以无限接近零
#Dropout ： 随机失活
#BN


#BN 批量归一化
#计算均值和方差 -> 标准化 -> 缩放和偏移
#减少内部协方差偏移 加速训练 更加稳定
#在计算机视觉领域用的多
#在卷积层或者全连接层之后，激活函数之前使用

#BatchNorm1d:全连接层或者处理一位数据的网络
#接收形状为(N,num_features)的输入
#BatchNorm2d:卷积层或者处理二维图像数据或特征图
#接收形状为(N,C,H,W)的输入
#BatchNorm3d:处理三维卷积神经网络
#接收形状为(N,C,D,H,W)的输入

import torch
import torch.nn as nn


#处理二维图像数据
def dm01():
    #初始化图像数据 一张图片两个通道三行四列
    input_2d = torch.randn(size = (1,2,3,4))
    print(f'input_2d:{input_2d}')

    #创建BN层
    #参数一：输入特征数 = 图片通道数
    #参数二：噪声值
    #参数三：动量值，用于计算移动平均统计量的动量值
    #参数四：是否需要学习缩放和偏移
    bn1 = nn.BatchNorm2d(num_features = 2,eps = 1e-5,momentum = 0.1,affine = True)
    output_2d = bn1(input_2d)
    print(f'output_2d:{output_2d}')


#一维数据
def dm02():
    #初始化一维数据
    #两个样本，每个有两个特征
    input_1d = torch.randn(2,2)
    print(f'input_1d:{input_1d}')

    #线性层
    linear = nn.Linear(2,2)
    l1 = linear(input_1d)
    print(f'l1:{l1}')

    #归一化层
    bn1 = nn.BatchNorm1d(num_features = 2)
    output_1d = bn1(l1)
    print(f'output_1d:{output_1d}')



#测试
if __name__ == '__main__':
    dm01()
    dm02()