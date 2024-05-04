import torch
import numpy as np


# # random
# x = torch.rand(2, 2, dtype=torch.double)
# x = torch.rand(2, 2, dtype=torch.float16)
#
#
# # ones or zeros
# x = torch.ones(2, 2)
# x = torch.zeros(2, 2)
#
#
# # tensor from lists
# x_lst = [2.4, 5.2]
# y_lst = [4.4, 7.2]
# z_lst = x_lst + y_lst
#
#
# # operations
# y.add_(x)
# print(x)
# print(y)
# print(x+y)
# print(x.add_(y)) # inplace in x
# print(x)
# print(x.sub_(y))
# print(x.mul_(y))
#
# # slicing
# x = torch.rand(5,3)
# print(x)
# print(x[-2:])
#
#
# # values in different dimensios
# x = torch.rand(4,4)
# y = x.view(16)
# print(x.view(-1, 8))
# print(y)
#
#
# # pointing to the same memory
# a = torch.ones(5)
# print(a)
# b = a.numpy()
# print(b)
# a.add_(1)
# b+=1
# print(a)
# print(b)
# a = np.ones(5) 
# print(a)
# b = torch.from_numpy(a)
# print(b)
# b.add_(1)
# print(a)
# print(b)

# # tpu vs cpu
# if torch.cuda.is_available():
#     device = torch.device("cuda")
#     x = torch.ones(5, device=device)
#     y = torch.ones(5)
#     y = y.to(device)
#     z = x+y
#     # z.numpy() #error: can't convert cuda:0 device type tensor to numpy. Use Tensor.cpu() to copy the tensor to host memory first
#     z = z.to("cpu")
#     print(z.numpy())
#     print(z)

# # requires_grad
# x = torch.ones(5, requires_grad=True)
# print(x)
