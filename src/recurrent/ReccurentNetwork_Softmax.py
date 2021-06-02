import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.init import xavier_uniform_


class RecurrentNetSoftmax(nn.Module):
    def __init__(self):
        super(RecurrentNetSoftmax, self).__init__()
        self.hidden_dim = 5
        self.rnn = nn.GRU(101, self.hidden_dim, 1, batch_first=True)
        self.rnn2 = nn.GRU(self.hidden_dim, self.hidden_dim, 1, batch_first=True)
        self.drop = nn.Dropout(0.59)
        self.fc1 = nn.Linear(5532, 2048)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 512)
        self.fc4 = nn.Linear(512, 2)
        xavier_uniform_(self.fc1.weight)
        xavier_uniform_(self.fc2.weight)
        xavier_uniform_(self.fc3.weight)
        xavier_uniform_(self.fc4.weight)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_dim).requires_grad_().cuda()
        x, hidden = self.rnn(x, h0)
        x, hidden = self.rnn2(x, hidden)
        x = x[:, -1, :]
        x = self.drop(x)
        x = self.relu(self.fc1(x))
        x = self.drop(x)
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def predict(self, x):
        x = self.forward(x)
        probs = torch.softmax(x, dim=1)
        return probs

    def activations(self, x):
        # outputs activation this is not necessary just for fun
        z1 = self.conv1(x)
        a1 = torch.relu(z1)
        out = self.pool(a1)

        z2 = self.conv2(out)
        a2 = torch.relu(z2)
        out = self.pool2(a2)
        out = out.view(out.size(0), -1)
        return z1, a1, z2, a2, out