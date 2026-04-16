import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        # ... (Keep your conv/bn/pool layers exactly as they are) ...
        
        # Add this to ensure the output is always 1x1 spatially
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1)) 

        self.fc1 = nn.Linear(256, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # ... (Keep your conv blocks as they are) ...
        x = self.pool5(F.leaky_relu(self.bn6(self.conv11(x))))

        # New step: Force spatial dimensions to 1x1
        x = self.global_pool(x) 
        x = torch.flatten(x, 1) # Now this will be exactly size 256

        x = self.dropout(F.leaky_relu(self.fc1(x)))
        x = self.dropout(F.leaky_relu(self.fc2(x)))
        x = self.fc3(x)

        return torch.sigmoid(x)
