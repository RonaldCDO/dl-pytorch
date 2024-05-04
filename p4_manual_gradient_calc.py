import numpy as np

# f = w * x

# f = 2 * x
X = np.array([1, 2, 3, 4], dtype=np.float32)
Y = np.array([2, 4, 6, 8], dtype=np.float32)

w = 0.1

# model prediction
def forward(x):
    return w * x

def loss(y, y_predicted):
    return ((y_predicted-y)**2).mean()

# gradient
# MSE = 1/N * (w*x - y)**2
# dJ/dw = 1/N 2x(w*x -y)
def gradient(x, y, y_predicted):
    print(f'gradient: {np.dot(2*x, y_predicted-y).mean()}')
    return np.dot(2*x, y_predicted-y).mean()

print(f'Initial prediction: f(1) = {forward(1):.3f} w: {w}\n'
      f'Desired result: {1*2}')

# Training
learning_rate = 0.01
n_iters = 11

for epoch in range(n_iters):
    # prediction = foward pass
    y_pred = forward(X)

    # loss
    l = loss(Y, y_pred)

    # gradients
    dw = gradient(X, Y, y_pred)

    # update weights
    print(f'update rate: {-learning_rate* dw:+}\n')
    w -= learning_rate * dw

    if epoch % 1 == 0:
        print(f'epoch {epoch+1}: w = {w:.3f}, loss = {l:.8f}')
    print(f'Prediction after training: f(5) = {forward(1):.3f}')
