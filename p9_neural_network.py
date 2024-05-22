import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class Model(nn.Module):
    def __init__(self, in_features, hidden_layer1, hidden_layer2, out_features) -> None:
        super().__init__()
        self.fully_connected1 = nn.Linear(in_features, hidden_layer1)
        self.fully_connected2 = nn.Linear(hidden_layer1, hidden_layer2)
        self.output = nn.Linear(hidden_layer2, out_features)
        self.losses = []

    def forward(self, input):
        activation_function = F.relu
        input = activation_function(self.fully_connected1(input))
        input = activation_function(self.fully_connected2(input))
        return self.output(input)

    def train_model(self, x_train, y_desired, criterion, optimizer, epochs) -> None:
        for epoch in range(epochs):
            y_pred = self.forward(x_train)
            loss = criterion(y_pred, y_desired)
            self.losses.append(loss.item())

            if epoch % 10 == 0:
                print(f'Epoch: {epoch} and loss: {loss}')
                torch.save(self.state_dict(), f'./weights/{epoch}.pt')

            optimizer.zero_grad()
            
            loss.backward()

            optimizer.step()

    def evaluate_model(self, x_test, y_desired) -> None:
        correct = 0
        with torch.no_grad():
            for i, data in enumerate(x_test):
                y_val = self.forward(data)
                pred_index = torch.argmax(y_val).item()
                print(f'{i+1}. {str(y_val)} {y_desired[i]} {pred_index} {True if pred_index == y_desired[i] else False}')

                if pred_index == y_desired[i]:
                    correct +=1
        print(f'{correct}/{len(y_desired)} predictions!')

    def plot(self, epochs) -> None:
        plt.plot(range(epochs), self.losses)
        plt.ylabel("Loss/Error")
        plt.xlabel("Epoch")
        plt.show()

def main():
    torch.manual_seed(41)
    # model = Model(in_features=4,hidden_layer1=8,hidden_layer2=9, out_features=3)
    iris = load_iris()
    columns = list(iris.feature_names)
    columns.append('variety')

    x = pd.DataFrame(iris.data)
    y = pd.DataFrame(iris.target)
    x = x.values
    y = y.values.reshape(-1)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    x_train = torch.FloatTensor(x_train)
    x_test = torch.FloatTensor(x_test)
    
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    # criterion = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    # epochs = 100
    # model.train_model(x_train=x_train, y_desired=y_train, criterion=criterion, optimizer=optimizer, epochs=epochs)
    # model.evaluate_model(x_test, y_test)
    # model.plot(epochs=epochs)

    new_model = Model(in_features=4,hidden_layer1=8,hidden_layer2=9, out_features=3)

    new_model.load_state_dict(torch.load('./weights/90.pt'))

    new_model.evaluate_model(x_test=x_test, y_desired=y_test)

if __name__ == '__main__':
    main()
