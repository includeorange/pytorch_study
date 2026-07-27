#模拟线性回归
import torch
from torch.utils.data import TensorDataset  #构造数据集对象
from torch.utils.data import DataLoader     #数据加载器
from torch import nn                        #nn模块中含有平方损失函数和假设函数
from torch import optim                     #优化器
from sklearn.datasets import make_regression#创建线性回归模型数据集
import matplotlib.pyplot as plt             #可视化

plt.rcParams['font.sans-serif'] = ['SimHei'] #指定默认字体
plt.rcParams['axes.unicode_minus'] = False   #解决保存图像是负号'-'显示为方块的问题

#ndarray -> tensor -> dataset数据集 -> dataloader数据加载器
#1.定义函数，创建线性回归数据集
def create_dataset():
    x,y,coef = make_regression(
        n_samples = 100,    #样本数量
        n_features = 1,     #特征数量
        noise = 10,         #噪声,噪声越大，样本点越散
        coef = True,        #是否返回系数
        bias = 14.5,        #偏置(生成这批数据时使用的偏置)
        random_state = 1    #随机种子
    )
    #将数据封装成张量对象
    x = torch.tensor(x,dtype = torch.float32)
    y = torch.tensor(y,dtype = torch.float32)
    return x,y,coef
    #仅做模拟用，真实情况下是用数据集来生成的

#2.训练模型
def train(x,y,coef):
    #1.创建数据集对象
    dataset = TensorDataset(x,y)
    #2.创建数据加载器
    #参数一：数据集对象，参数二：每次加载的样本数量，参数三：是否打乱(训练集打乱，测试集不打乱)
    dataloader = DataLoader(dataset,batch_size = 16,shuffle = True)
    #3.创建线性回归模型
    #参数一：输入特征维度数，参数二：输出特征维度数
    model = nn.Linear(1,1)
    #4.创建损失函数
    criterion = nn.MSELoss()
    #5.创建优化器
    #参数一：模型参数，参数二：学习率
    optimizer = optim.SGD(model.parameters(),lr = 0.01)
    #6.训练过程
    #6.1 定义变量：训练轮数，每轮平均损失，训练总损失，训练样本数
    epochs,loss_list,total_loss,total_sample = 100,[],0.0,0
    #6.2 开始训练
    for epoch in range(epochs):#epoch:0,1,2...99
        #6.3 分批次训练
        for train_x,train_y in dataloader:#7批(16,16,16,16,16,16,4)
            #6.4 模型预测
            y_pred = model(train_x)
            #6.5 计算(每批的平均)损失
            loss = criterion(y_pred,train_y.reshape(-1,1))#-1 自动计算
            #6.6 计算总损失 和 样本(批次)数
            total_loss += loss.item()
            total_sample += 1
            #6.7 梯度清零 反向传播 梯度更新
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #6.8 将本轮的平均损失添加到列表中
        loss_list.append(total_loss / total_sample)
        #6.9 打印本轮的平均损失
        print(f'第{epoch + 1}轮平均损失为{total_loss / total_sample }')
        
    #7.绘制损失曲线
    plt.plot(range(epochs),loss_list)
    plt.xlabel('轮数')
    plt.ylabel('损失')
    plt.title('线性回归损失曲线')
    plt.show()
    #8.预测值和真实值之间的关系
    #8.1样本点分布情况
    plt.scatter(x,y)
    #8.2模型预测值
    y_pred = torch.tensor(data = [v * model.weight + model.bias.item() for v in x])
    #8.3真实值 这个计算过程基于创建数据的函数create_dataset , 实际应用中需要用真实数据
    y_true = torch.tensor(data = [v * coef + 14.5 for v in x])
    #8.4绘制预测值和真实值之间的关系
    plt.plot(x,y_pred,color = 'red',label = '预测值')
    plt.plot(x,y_true,color = 'blue',label = '真实值')
    #8.5添加图例,网络
    plt.legend()
    plt.grid()
    plt.show()
    #9.打印系数
    print(f'系数为{model.state_dict()}')

#测试：
if __name__ == '__main__':
    x,y,coef = create_dataset()
    train(x,y,coef)

