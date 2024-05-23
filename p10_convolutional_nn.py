import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import time
from typing import Any

class DataBase:
    def __init__(self):
        self.transform = transforms.ToTensor()
        self.train_data = datasets.MNIST(root='./data', train=True, download=True, transform=self.transform)
        self.test_data = datasets.MNIST(root='./data', train=False, download=True, transform=self.transform)
        self.train_loader = DataLoader(self.train_data, batch_size=10, shuffle=True)
        self.test_loader = DataLoader(self.test_data, batch_size=10, shuffle=False)

class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3, stride=1)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(5*5*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        self.train_losses = []
        self.train_correct = []
        self.test_losses = []
        self.test_correct = []

    def forward(self, X):
        # First pass
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)

        # Second pass
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)

        # Flattening
        X = X.view(-1, 16*5*5)

        # Fully Connected
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)

        return F.log_softmax(X, dim=1)

    def train_model(self, criterion, optimizer, epochs, train_loader, test_loader):
        start_time = time.time()
        
        for epoch in range(epochs):
            train_correct_samples = 0
            loss: Any = 0
            for b, (X_train, y_train) in enumerate(train_loader):
                b+=1
                y_pred = self(X_train)
                loss = criterion(y_pred, y_train)

                predicted = torch.max(y_pred.data, 1)[1]
                batch_correct = (predicted == y_train).sum()
                train_correct_samples += batch_correct

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if b%600 == 0:
                    print(f'Epoch: {epoch}, Batch: {b} and loss: {loss}')
                    torch.save(self.state_dict(), f'./weights-cnn/{epoch}.pt')
            self.train_losses.append(loss.item())
            self.train_correct.append(train_correct_samples)

            test_correct_samples = 0
            loss: Any = 0
            with torch.no_grad():
                for b, (X_test, y_test) in enumerate(test_loader):
                    b+=1
                    y_val = self(X_test)
                    loss = criterion(y_val, y_test)

                    predicted = torch.max(y_val.data, 1)[1]
                    test_correct_samples += (predicted == y_test).sum()
                
            self.test_losses.append(loss.item())
            self.test_correct.append(test_correct_samples)

        end_time = time.time()
        print(f'{(end_time-start_time)/60} minutes')
        self.plot_loss()
        self.plot_accuracy()
        
    def plot_loss(self):
        plt.plot(self.train_losses, label='Training Loss')
        plt.plot(self.test_losses, label='Validation Loss')
        plt.title("Loss at Epoch")
        plt.legend()
        plt.show()

    def plot_accuracy(self):
        plt.plot([t/600 for t in self.train_correct], label='Training accuracy')
        plt.plot([t/100 for t in self.test_correct], label='Test accuracy')
        plt.title("Accuracy at Epoch")
        plt.legend()
        plt.show()

def main():

    torch.manual_seed(41)

    db = DataBase()
    model = ConvNet()
    criterion = nn.CrossEntropyLoss()
    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    model.train_model(criterion, optimizer, 5, db.train_loader, db.test_loader)



if __name__ == '__main__':
    main()
