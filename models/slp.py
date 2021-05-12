from tqdm import tqdm
from torch import nn
import torch


class SLP(nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super(SLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.act = nn.ReLU()
        self.bn = nn.BatchNorm1d(n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_out)
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.bn(x)
        x = self.fc2(x)
        return x

    def inference(self, x):
        x = self.forward(x)
        x = self.sm(x)
        return x


class SLPClassifer:
    def __init__(self, n_in=1, n_hidden=1, n_out=1, epoch=None, batch_size=None):
        self.model = SLP(n_in, n_hidden, n_out)
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1.e-3)
        self.epoch = epoch
        self.batch_size = batch_size

    def fit(self, x, y):
        x = torch.tensor(x).float()
        y = torch.tensor(y)
        self.model.train()
        for _ in range(self.epoch):
            for start_ind in range(0, len(x), self.batch_size):
                self.optimizer.zero_grad()
                x_batch = x[start_ind: start_ind + self.batch_size]
                y_batch = y[start_ind: start_ind + self.batch_size]
                preds = self.model.forward(x_batch)
                loss_value = self.loss(preds, y_batch)
                loss_value.backward()
                self.optimizer.step()

    def predict(self, x):
        x = torch.tensor(x).float()
        self.model.eval()
        with torch.no_grad():
            pred = self.model.inference(x)
        pred = torch.max(pred, 1).indices
        return pred.numpy()
