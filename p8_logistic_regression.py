import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class Preprocessor:
    def __init__(self, dataset):
        self.dataset = dataset

    def get_data_and_target(self, dataset) -> tuple[np.ndarray, np.ndarray]:
        return (dataset.data, dataset.target)

    def get_samples_and_features(self, dataset: np.ndarray) -> tuple[int, int]:
        X, y = dataset.shape
        return (X, y)

    def split_dataset(self, X, y, test_size):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=1234
        )
        return (X_train, X_test, y_train, y_test)

    def transform_to_tensor(self, X, type: type) -> torch.Tensor:
        return torch.from_numpy(X.astype(type))


class MetricsCalculator:
    def __init__(self, actual, predicted):
        self.actual = actual
        self.predicted = predicted

    def get_confusion_matrix_tuple(self) -> tuple[int, int, int, int]:
        tp = fp = tn = fn = 0
        for a, p in zip(self.actual, self.predicted):
            for actual_value, predicted_value in zip(a, p):
                if actual_value == 1 and predicted_value == 1:
                    tp += 1
                if actual_value == 1 and predicted_value == 0:
                    fp += 1
                if actual_value == 0 and predicted_value == 0:
                    tn += 1
                if actual_value == 0 and predicted_value == 1:
                    fn += 1

        self.tp = tp
        self.tn = tn
        self.fp = fp
        self.fn = fn
        return (tp, fp, tn, fn)

    def get_accuracy(self):
        return (self.tp + self.tn) / (self.tp + self.tn + self.fp + self.fn)

    def get_precision(self):
        return self.tp / (self.tp + self.fp)

    def get_recall(self):
        return self.tp  / (self.tp + self.fn)


class LogisticRegression(nn.Module):
    def __init__(self, n_input_features, learning_rate):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(n_input_features, 1)
        self.learning_rate = learning_rate

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

    def lr_train(self, epochs, X_train, y_train, criterion, optimizer):
        for epoch in range(epochs):
            y_predicted = self.forward(X_train)
            loss = criterion(y_predicted, y_train)

            loss.backward()

            optimizer.step()

            optimizer.zero_grad()

            if (epoch + 1) % 10 == 0:
                print(f"epoch: {epoch}, loss: {loss:.4f}")


def main():
    # load dataset
    lg = Preprocessor(datasets.load_breast_cancer())

    # getting dataset and target as input and output
    X, y = lg.get_data_and_target(lg.dataset)

    # samples and features
    _, n_features = lg.get_samples_and_features(X)

    # splitting dataset into training dataset and testing dataset 0.8 training and 0.2 testing
    X_train, X_test, y_train, y_test = lg.split_dataset(X, y, 0.2)

    # applying standard scaller
    sc = StandardScaler()

    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # transforming sets to tensors
    X_train = lg.transform_to_tensor(X_train, np.float32)
    X_test = lg.transform_to_tensor(X_test, np.float32)
    y_train = lg.transform_to_tensor(y_train, np.float32)
    y_test = lg.transform_to_tensor(y_test, np.float32)

    # transposing the output sets
    y_train = y_train.view(y_train.shape[0], 1)
    y_test = y_test.view(y_test.shape[0], 1)

    # instancing LogisticRegression model
    model = LogisticRegression(n_features, learning_rate=0.01)

    # Binary Cross Entropy loss as criterion
    criterion = nn.BCELoss()

    # Stochastic Gradient Descent as optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    # training function
    model.lr_train(150, X_train, y_train, criterion, optimizer)

    # calculating accuracy
    with torch.no_grad():
        y_predicted = model(X_test)
        y_predicted_cls = y_predicted.round()
        mc = MetricsCalculator(y_predicted_cls, y_test)

        mc.get_confusion_matrix_tuple()

        acc = mc.get_accuracy()
        prec = mc.get_precision()
        rec = mc.get_recall()

        print('\nMetrics:')
        print(f"Accuracy: {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall: {rec:.4f}")

if __name__ == "__main__":
    main()
