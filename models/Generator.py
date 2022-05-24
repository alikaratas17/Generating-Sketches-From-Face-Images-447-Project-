import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

class Generator(nn.Module):
    def __init__(self, input_channel_number, output_channel_number, resnet_block_number=3):
        #input_channel_number and output_channel_number are 3 or 1
        #input: BxCxDxD
        super(Generator, self).__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(input_channel_number, 64, 7, padding = 3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
        )
        
        in_features = 64
        out_features = 128
        
        downsampling_model = []
        for _ in range(2):
            downsampling_model += [nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True)]
            in_features = out_features
            out_features = in_features*2

        self.downsampling = nn.Sequential(*downsampling_model)

        resnet_blocks_model = []
        for _ in range(resnet_block_number):
            resnet_blocks_model += [ResnetBlock(in_features)]

        self.resnet_blocks = nn.Sequential(*resnet_blocks_model)

        upsampling_model = []
        for _ in range(2):
            out_features //= 2
            upsampling_model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True),
            ]
            in_features = out_features
            out_features = in_features//2

        self.upsampling = nn.Sequential(*upsampling_model)

        self.output_layer = nn.Sequential(
            nn.Conv2d(64, output_channel_number, 7, padding = 3),
            nn.Sigmoid()
        )


    def forward(self, x):
        x = self.conv_block(x)
        x = self.downsampling(x)
        x = self.resnet_blocks(x)
        x = self.upsampling(x)
        x = self.output_layer(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, conv_dim):
        super(ResnetBlock, self).__init__()
        self.resnet_block = nn.Sequential(
            nn.Conv2d(conv_dim, conv_dim, kernel_size = (3,3), stride = (1,1), padding = 1),
            nn.InstanceNorm2d(conv_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(conv_dim, conv_dim, kernel_size = (3,3), stride = (1,1), padding = 1),
            nn.InstanceNorm2d(conv_dim),
        )
        
    def forward(self, x):
        return self.resnet_block(x)+x
