import torch

X = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
Y = torch.tensor([2, 4, 6, 8], dtype=torch.float32)

w = torch.tensor(0.5, dtype=torch.float32, requires_grad=True)

def forward(x):
    return x*w

def loss(y, y_pred):
    return ((y_pred-y)**2).mean()

# training
learning_rate = 0.01
n_iters=100

print(f'Initial prediction f(1) = {forward(1):.3f} w: {w}\n'
      f'Desired result: {2}')

for epoch in range(n_iters):
    y_pred= forward(X)

    l = loss(Y, y_pred)

    l.backward() # dl/dw

    with torch.no_grad():
        w -= w.grad.mul_(learning_rate)

    w.grad.zero_()

    if epoch % 10 ==0:
        print(f'epoch {epoch+1}: w = {w:3f}, loss = {l:.8f}')
