import torch

class NeuralNetwork:
    def __init__(self, X: torch.Tensor, Y: torch.Tensor, w: torch.Tensor):
        self.X = X
        self.Y = Y
        self.w = w
        self.learning_rate = 0.01

        print('Initial_parameters:\n'
        f'X: {self.X}\n'
        f'Y: {self.Y}\n'
        f'w: {self.w}\n'
        f'lr: {self.learning_rate}\n')

    def forward(self, x: torch.Tensor):
        return self.w * x

    def loss(self, y_pred: torch.Tensor):
        return ((y_pred - self.Y)**2).mean()

    def update_weights(self) -> None:
        with torch.no_grad():
            if self.w.grad is not None:
                self.w-= self.w.grad.mul_(self.learning_rate)
            if self.w.grad is not None:
                self.w.grad.zero_()


    def backpropagation(self, loss):
        loss.backward()
        self.update_weights()
        return self.w.grad 

    def train(self, epochs: int):
        print('Starting training...')
        for epoch in range(epochs):
            y_pred = self.forward(self.X)

            l = self.loss(y_pred)

            self.backpropagation(l)

            if epoch % 10 == 0:
                print(f'epoch: {epoch}, w: {self.w:.3f}, loss:{l:.8f}')

def main():
    X = torch.Tensor([1, 2, 3, 4])
    Y = torch.Tensor([2, 4, 6, 8])
    w = torch.tensor(0.5, dtype=torch.float32, requires_grad=True)
    nn = NeuralNetwork(X, Y, w)
    nn.train(100)

if __name__ == '__main__':
    main()
