#张量的形状操作
#reshape() 在不改变元素内容的前提下，改变形状，比如两行三列变成一行六列
#unsqueeze() 在指定维度位置添加一个值为一的维度(是维度值等于一，不是元素等于一)
#squeeze() 删除一维度的维
#transpose()  permute()  transpose一次只能交换两个维度，permute可以改变多个


#导包
import torch

def dm01():
    torch.manual_seed(100)
    t1 = torch.randint(0,10,(2,3))
    print(f't1:{t1},shape:{t1.shape[0]},shape:{t1.shape[1]}')
    t2 = t1.reshape(1,6)
    print(f't2:{t2},shape:{t2.shape[0]},shape:{t2.shape[1]}')

def dm02():
    torch.manual_seed(100)
    t1 = torch.randint(0,10,(2,3))
    print(f't1:{t1},shape:{t1.shape[0]},shape:{t1.shape[1]}')
    t2 = t1.unsqueeze(0)#在第一维度添加一个维度,变成1,2,3
    print(f't2:{t2},shape:{t2.shape[0]},shape:{t2.shape[1]}')
    t3 = t2.squeeze(0)
    print(f't3:{t3},shape:{t3.shape[0]},shape:{t3.shape[1]}')

def dm03():
    torch.manual_seed(100)
    t1 = torch.randint(0,10,(2,3,4))
    print(f't1:{t1},shape:{t1.shape[0]},shape:{t1.shape[1]}')
    t2 = t1.transpose(0,1)
    print(f't2:{t2},shape:{t2.shape[0]},shape:{t2.shape[1]}')
    t3 = t1.permute(2,0,1)#变成4,2,3
    #permute函数内的参数是*dims，*的意思是不用敲[2,0,1],会自动将2,0,1转换成列表
    #也就是2,0,1是位置参数，指的是原本的第二维，第零维，第一维
    #将对应的维度放到[]这个张量指定的位置上
    #变成4,2,3
    print(f't3:{t3},shape:{t3.shape[0]},shape:{t3.shape[1]}')



#测试函数
if __name__ == '__main__':
    dm01()
    dm02()

