#2.3_lineralgebra

import numpy as np
import torch

#x = torch.tensor(3.0)
#y = torch.tensor(2.0)

#print(x * y)

#x = torch.arange(4)
#print(x)
#print(x[3])
#print(x.shape)

x = torch.arange(20).reshape(5,4).cuda()
print(x)
#print(torch.cuda.is_available())  # True 表示有 NVIDIA 显卡
print(x.T)

x = torch.tensor([[1, 2, 3], [2, 0, 4],[3, 4, 5]]).cuda()
print(x)
print(x == x.T)

x = torch.arange(20, dtype=torch.float32).reshape(5, 4).cuda()
print(x)
y = x.clone()
print(x + y)

print(x * y)
a = 2
print(a + x)
print((a * x).shape)

print(x.sum())
print(x.mean())

sum_x = x.sum(axis=0, keepdim=True)
print(sum_x)
print(x)
print(x.cumsum(axis = 0))

x = torch.arange(4, dtype=torch.float32)
y = torch.ones(4, dtype=torch.float32)
print(torch.dot(x, y))

print(x * y)