import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()

        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)

        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 16, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool1 = nn.AvgPool2d(2, 2)

        self.conv4 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv5 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)
        self.pool2 = nn.AvgPool2d(2, 2)

        self.conv6 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv7 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv8 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv9 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.pool3 = nn.AvgPool2d(2, 2)

        self.conv10 = nn.Conv2d(64, 128, 5, padding=2)
        self.bn5 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(2, 2)

        self.conv11 = nn.Conv2d(128, 256, 5, padding=2)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool5 = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(256, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):

        x = self.pool1(F.leaky_relu(self.bn1(self.conv1(x))))
        x = self.pool1(F.leaky_relu(self.bn2(self.conv3(F.leaky_relu(self.conv2(x))))))
        x = self.pool2(F.leaky_relu(self.bn3(self.conv5(F.leaky_relu(self.conv4(x))))))
        x = self.pool3(F.leaky_relu(self.bn4(self.conv9(
            F.leaky_relu(self.conv8(
            F.leaky_relu(self.conv7(
            F.leaky_relu(self.conv6(x))))))))))
        x = self.pool4(F.leaky_relu(self.bn5(self.conv10(x))))
        x = self.pool5(F.leaky_relu(self.bn6(self.conv11(x))))

        x = torch.flatten(x, 1)

        x = self.dropout(F.leaky_relu(self.fc1(x)))
        x = self.dropout(F.leaky_relu(self.fc2(x)))

        x = self.fc3(x)

        return torch.sigmoid(x)
