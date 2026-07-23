#基本运算
#加减乘除取反
#+ - * / //(这个是整除)
import torch
t1 = torch.tensor([1,2,3])
t2 = t1 + 2
t3 = t1 - 2
t4 = t1 * 2
t5 = t1 / 2
t6 = t1 // 2
t7 = -t1
print(f't1:{t1}')
print(f't2:{t2}')
print(f't3:{t3}')
print(f't4:{t4}')
print(f't5:{t5}')
print(f't6:{t6}')
print(f't7:{t7}')