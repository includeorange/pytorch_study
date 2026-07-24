#创建指定类型张量或张量类型转换
#torch.tensor(,dtype = )逗号前面放数据，dtype制定数据类型
#t2 = t1.type(torch.dtype)张量类型转换方法一
#t2 = t1.int()    ... 张量类型转换方法二

import torch
def dm01():
    #在创建的时候指定好元素类型
    t1 = torch.tensor([1,2,3],dtype = torch.float)#默认是float32
    print(f'元素类型:{t1.dtype}，张量类型:{type(t1)}')
    #张量类型转换方法一
    t2 = t1.type(torch.int64)
    print(f'元素类型:{t2.dtype}，张量类型:{type(t2)}')
def dm02():
    t1 = torch.tensor([1,2,3],dtype = torch.float)#默认是float32
    #张量类型转换方法二
    print(t1.half()) #half是float16
    print(t1.float()) #float是float32,默认
    print(t1.double()) #double是float64
    print(t1.short()) #short是int16
    print(t1.int()) #int是int32
    print(t1.long()) #long是int64，默认




#测试函数
if __name__ == '__main__':
    dm01()
    dm02()

