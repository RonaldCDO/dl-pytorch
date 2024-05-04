import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt


class Preprocessor:

    def regression(self, samples: int, features: int):
        self.X: np.ndarray
        self.y: np.ndarray
        self.X, self.y = datasets.make_regression(n_samples=samples, n_features=features, noise=20, random_state=1)
        return self.X, self.y

    def transform_to_tensor(self, type: type) -> tuple[torch.Tensor, torch.Tensor]:

        X, y = (torch.from_numpy(self.X.astype(type)),torch.from_numpy(self.y.astype(type)))
        y = y.view(y.shape[0], 1)
        return X, y

    def get_samples_and_features(self, X: torch.Tensor):
        return X.shape


class LinearRegressionModel(Preprocessor):
    def __init__(self, lr):
        self.lr: float = lr

    def generate_model(self, input_size: int, output_size: int) -> nn.Linear:
        return nn.Linear(input_size, output_size)

    def sgd_optimizer(self, model: nn.Linear):
        return torch.optim.SGD(model.parameters(), lr=self.lr)

    def forward_pass(self, model, X: torch.Tensor) -> torch.Tensor:
        return model(X)

    def train(self,model: nn.Linear, X: torch.Tensor, y: torch.Tensor, epochs: int):
        optimizer = self.sgd_optimizer(model)
        for epoch in range(epochs):
            y_predicted = self.forward_pass(model, X)
            criterion = nn.MSELoss()
            loss = criterion(y_predicted, y)

            loss.backward()

            optimizer.step()

            optimizer.zero_grad()
            if (epoch+1) % 10 == 0:
                print(f'epoch: {epoch*1}, loss = {loss.item():.4f}')





def main():
    lrm = LinearRegressionModel(0.01)
    X_numpy, y_numpy = lrm.regression(100, 1)
    X, y = lrm.transform_to_tensor(np.float32)
    _, n_features = lrm.get_samples_and_features(X)
    input_size = n_features
    output_size = 1

    model = lrm.generate_model(input_size=input_size, output_size=output_size)

    lrm.train(model, X, y, 150)

    predicted = model(X).detach().numpy()

    plt.plot(X_numpy, y_numpy, 'ro')
    plt.plot(X_numpy, predicted, 'b')
    plt.show()


if __name__ == '__main__':
    main()
