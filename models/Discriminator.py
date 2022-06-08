
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

# Discriminator architectures are implemented following patchgan as described by its paper with changes following UPDG

class Discriminator(nn.Module):
    def __init__(self,in_channels,image_size=70):
        super(Discriminator, self).__init__()
        self.dim = image_size
        self.relu = nn.LeakyReLU(negative_slope= 0.2, inplace=False)
        if image_size==70:
            self.conv1 = nn.Conv2d(in_channels,64,4,stride=2)
            self.conv2 = nn.Conv2d(64,128,4,stride=2)
            self.bnorm2 = nn.BatchNorm2d(128)
            self.conv3 = nn.Conv2d(128,256,4,stride=2)
            self.bnorm3 = nn.BatchNorm2d(256)
            self.conv4 = nn.Conv2d(256,512,4,stride=2)
            self.bnorm4 = nn.BatchNorm2d(512)
            self.conv5 = nn.Conv2d(512,1,2,stride=1)
        
        if image_size==64:
            self.conv1 = nn.Conv2d(in_channels,64,4,stride=2,padding=1)
            self.conv2 = nn.Conv2d(64,128,4,stride=2,padding=1)
            self.bnorm2 = nn.BatchNorm2d(128)
            self.conv3 = nn.Conv2d(128,256,4,stride=2,padding=1)
            self.bnorm3 = nn.BatchNorm2d(256)
            self.conv4 = nn.Conv2d(256,512,4,stride=2,padding=1)
            self.bnorm4 = nn.BatchNorm2d(512)
            self.conv5 = nn.Conv2d(512,1,4,stride=2)
        
        if image_size==128:
            self.conv1 = nn.Conv2d(in_channels,64,4,stride=2,padding=1)
            self.conv2 = nn.Conv2d(64,128,4,stride=2,padding=1)
            self.bnorm2 = nn.BatchNorm2d(128)
            self.conv3 = nn.Conv2d(128,256,4,stride=2,padding=1)
            self.bnorm3 = nn.BatchNorm2d(256)
            self.conv4 = nn.Conv2d(256,512,4,stride=2,padding=1)
            self.bnorm4 = nn.BatchNorm2d(512)
            self.conv5 = nn.Conv2d(512,512,4,stride=2,padding=1)
            self.bnorm5 = nn.BatchNorm2d(512)
            self.conv6 = nn.Conv2d(512,1,4,stride=2)

    def forward(self,x):
        if self.dim == 70:
            x = self.conv1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.bnorm2(x)
            x = self.relu(x)
            x = self.conv3(x)
            x = self.bnorm3(x)
            x = self.relu(x)
            x = self.conv4(x)
            x = self.bnorm4(x)
            x = self.relu(x)
            x = self.conv5(x)
        
        if self.dim == 64:
            x = self.conv1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.bnorm2(x)
            x = self.relu(x)
            x = self.conv3(x)
            x = self.bnorm3(x)
            x = self.relu(x)
            x = self.conv4(x)
            x = self.bnorm4(x)
            x = self.relu(x)
            x = self.conv5(x)
        
        if self.dim == 128:
            x = self.conv1(x)
            x = self.relu(x)
            x = self.conv2(x)
            x = self.bnorm2(x)
            x = self.relu(x)
            x = self.conv3(x)
            x = self.bnorm3(x)
            x = self.relu(x)
            x = self.conv4(x)
            x = self.bnorm4(x)
            x = self.relu(x)
            x = self.conv5(x)
            x = self.bnorm5(x)
            x = self.relu(x)
            x = self.conv6(x)
        
        x = torch.sigmoid(x)
        return x.mean(dim=(1,2,3))



if __name__ == '__main__':
    print(Discriminator(3,64)(torch.randn(6,3,512,512)).shape)
    print(Discriminator(3,70)(torch.randn(6,3,512,512)).shape)
    print(Discriminator(3,128)(torch.randn(6,3,512,512)).shape)
