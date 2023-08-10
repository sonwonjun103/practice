import torch.nn as nn
import torch

class ensemble_model(nn.Module):
    def __init__(self, classes=2):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, (3,3), stride=2, padding=1)
        self.conv2 = nn.Conv2d(8, 4, (3, 3), stride=2, padding=1)
        self.conv3 = nn.Conv2d(4, 3, (3,3), stride=2, padding=1)

        self.fc1 = nn.Linear(768*3, 256, bias=True)
        self.fc2 = nn.Linear(256, classes, bias=True)
        # self.fc3 = nn.Linear(64, classes, bias=True)

        self.relu=nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        #find direction axis of concat
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x=x.view(x.shape[0], -1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)  

        # x = self.fc3(x) 
        # x = self.relu(x)
        # x = self.dropout(x)

        x = self.softmax(x)   

        return x