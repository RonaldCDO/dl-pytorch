import torch

# # requires grad
# x = torch.randn(3, requires_grad=True)
# y = torch.rand(3, requires_grad=True)
# print(x)
# print(y)
# y = x + 2
# print(y)
# z = y*y*2
# z = z.mean()
# print(z)
# z.backward()
# print(x.grad)

# # detaching grad
# x.requires_grad_(False)
# print(x)
# y = x.detach()
# print(y)

# weights = torch.ones(4, requires_grad=True)

# for epoch in range(3):
#     model_output = (weights*3).sum()
#     model_output.backward()
#
#     print(weights.grad)
#     print(model_output)
#
#     weights.grad.zero_() #pause the grad during the epochs 

# # optimizer way
# optimizer = torch.optim.SGD(weights, lr=0.01)
# optimizer.step()
# optimizer.zero_grad()
