import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch import optim
import numpy as np

NUM_CLASSES = 9

class Classifier1(nn.Module):
    def __init__(self):
        super(Classifier1, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv1_bn = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv2_bn = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64,32,5,padding=4)
        self.conv3_bn = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 64, 3)
        self.conv4_bn = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64, 128, 3)
        self.conv5_bn = nn.BatchNorm2d(128)
        self.conv6 = nn.Conv2d(128, 64, 3,padding=2)
        self.conv6_bn = nn.BatchNorm2d(64)

        self.conv7=nn.Conv2d(64,128,5)
        self.conv7_bn = nn.BatchNorm2d(128)
        self.conv8=nn.Conv2d(128,256,3)
        self.conv8_bn = nn.BatchNorm2d(256)
        self.conv9=nn.Conv2d(256,128,3,padding=2)
        self.conv9_bn = nn.BatchNorm2d(128)
        self.conv10=nn.Conv2d(128,256,5)
        self.conv10_bn = nn.BatchNorm2d(256)
        self.pool_1= nn.MaxPool2d(3,3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(256*10*10, 512)
        #self.fc1_bn = nn.BatchNorm1d(50,120)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256,120)
        #self.fc2_bn = nn.BatchNorm1d(50,84)
        self.fc4 = nn.Linear(120, 84)
        

    def forward(self, x):
        #print(x.shape)
        x = self.pool(F.relu(self.conv1_bn(self.conv1(x))))

        residual_1=x
        #print(x.shape)
        x = F.relu(self.conv2_bn(self.conv2(x)))

        x = F.relu(self.conv3_bn(self.conv3(x)))
        #print(x.shape)
        x=x+residual_1


        x = self.pool(F.relu(self.conv4_bn(self.conv4(x))))
        residual_2=x
        x = F.relu(self.conv5_bn(self.conv5(x)))

        x = F.relu(self.conv6_bn(self.conv6(x)))
        x=x+residual_2
        x = self.pool(F.relu(self.conv7_bn(self.conv7(x))))
        residual_3=x
        x = F.relu(self.conv8_bn(self.conv8(x)))

        x = F.relu(self.conv9_bn(self.conv9(x)))

        x=x+residual_3
        x = self.pool(F.relu(self.conv10_bn(self.conv10(x))))
        #print(x.shape)
        x = x.view(x.size()[0], 256*10*10)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        
        return x
