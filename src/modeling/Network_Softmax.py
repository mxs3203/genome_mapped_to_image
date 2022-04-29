import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.init import xavier_uniform_


class ConvNetSoftmax(nn.Module):
    def __init__(self):
        super(ConvNetSoftmax, self).__init__()
        kernel_size = 4
        kernels = 64
        self.conv1 = nn.Conv2d(in_channels=5, out_channels=kernels, kernel_size=(kernel_size, kernel_size),
                               stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=kernels, out_channels=kernels, kernel_size=(kernel_size, kernel_size),
                               stride=1, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size,kernel_size)
        self.pool2 = nn.MaxPool2d(kernel_size,kernel_size)
        self.fc1 = nn.Linear(9216, 2048)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(2048, 2048)
        self.fc3 = nn.Linear(2048, 512)
        self.fc4 = nn.Linear(512, 6)
        self.batchnorm2d = nn.BatchNorm2d(5)
        self.batchnorm1d = nn.BatchNorm1d(9216)
        self.batchnorm1d2 = nn.BatchNorm1d(2048)
        self.batchnorm2d2 = nn.BatchNorm2d(kernels)
        xavier_uniform_(self.conv1.weight)
        xavier_uniform_(self.conv2.weight)
        xavier_uniform_(self.fc1.weight)
        xavier_uniform_(self.fc2.weight)
        xavier_uniform_(self.fc3.weight)
        xavier_uniform_(self.fc4.weight)


    def forward(self, x):
        x = self.batchnorm2d(x)
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.batchnorm2d2(x)
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.batchnorm1d(x)
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc2(x))
        x = self.batchnorm1d2(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        x = self.drop(x)
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def predict(self, x):
        x = self.forward(x)
        #probs = torch.softmax(x, dim=1)
        return x

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