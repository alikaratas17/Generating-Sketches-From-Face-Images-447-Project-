
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

# Discriminator architectures are implemented following patchgan as described by its paper
# Global discriminator includes additions to it as mentioned in UPDG


class Global_Discriminator(nn.Module):
    def __init__(self,in_channels):
        super(Global_Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels,64,4,stride=2)
        self.conv2 = nn.Conv2d(64,128,4,stride=2)
        self.bnorm2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128,256,4,stride=2)
        self.bnorm3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256,512,4,stride=2)
        self.bnorm4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512,1,2,stride=1)
        self.relu = nn.LeakyReLU(negative_slope= 0.2, inplace=True)
    def forward(self,x):
        x = self.conv1(x)
        self.relu(x)
        x = self.conv2(x)
        x = self.bnorm2(x)
        self.relu(x)
        x = self.conv3(x)
        x = self.bnorm3(x)
        self.relu(x)
        x = self.conv4(x)
        x = self.bnorm4(x)
        self.relu(x)
        x = self.conv5(x)
        x = torch.sigmoid(x).squeeze(3).squeeze(2)
        return x

class Local_Discriminator(nn.Module):
    def __init__(self,in_channels):
        super(Local_Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels,64,4,stride=2)
        self.conv2 = nn.Conv2d(64,128,4,stride=2)
        self.bnorm2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128,256,4,stride=2)
        self.bnorm3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256,512,4,stride=2)
        self.bnorm4 = nn.BatchNorm2d(512)
        self.conv5 = nn.Conv2d(512,1,2,stride=1)
        self.relu = nn.LeakyReLU(negative_slope= 0.2, inplace=True)
    def forward(self,x):
        x = self.conv1(x)
        self.relu(x)
        x = self.conv2(x)
        x = self.bnorm2(x)
        self.relu(x)
        x = self.conv3(x)
        x = self.bnorm3(x)
        self.relu(x)
        x = self.conv4(x)
        x = self.bnorm4(x)
        self.relu(x)
        x = self.conv5(x)
        x = torch.sigmoid(x).squeeze(3).squeeze(2)
        return x


if __name__ == '__main__':
    print(Local_Discriminator(3)(torch.randn(6,3,70,70)).shape)