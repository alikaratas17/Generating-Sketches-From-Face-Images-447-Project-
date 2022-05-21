#!pip install timm
import torch
import torch.optim as optim
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import trange, tqdm_notebook
import copy
from torch.distributions.uniform import Uniform
from torch.distributions.normal import Normal
import matplotlib.pyplot as plt
from scipy.stats import norm

class Xception(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.model = timm.create_model('xception', pretrained=True, features_only=True)
  def forward(self, x):
    print(x.shape)
    self.model.eval()


    #config = resolve_data_config({}, model=self.model)
    #config["input_size"] = [x.shape[2], x.shape[3]]
    #print(config)
    #transform = create_transform(**config)

    #tensor = transform(x).unsqueeze(0).cuda()
    tensor = x
    with torch.no_grad():
      x0, x1, x2, x3, x4 = self.model(tensor)
    return x3, x4

class SpatialPath(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    kernel_size = 3
    stride = 2
    self.layers = nn.Sequential(
        nn.Conv2d(in_channels, 64, kernel_size, stride, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size, stride, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.Conv2d(64, out_channels, kernel_size, stride, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
        )
    
  def forward(self, x):
    return self.layers(x)

class ARM(nn.Module):
  def __init__(self, in_channels, out_channels):
    """in_channels=out_channels??
    """
    super().__init__()
    stride = 2 #?
    self.layers = nn.Sequential(
        nn.AvgPool2d(kernel_size = 3, stride = 1, padding = 1),
        nn.Conv2d(in_channels, out_channels, 1),
        nn.BatchNorm2d(out_channels),
        nn.Sigmoid()
    )

  def forward(self, x):
    z = self.layers(x)
    return torch.mul(x, z)

class FFM(nn.Module):
  def __init__(self, in_channels, out_channels):
    """
    in_channels=out_channels??
    """
    super().__init__()
    self.convBnReLU = nn.Sequential(
        nn.Conv2d(4827, out_channels, kernel_size = 3, stride = 1, padding = 1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU()
    )
    self.layers = nn.Sequential(
        nn.AvgPool2d(kernel_size = 3, stride = 1, padding = 1),
        nn.Conv2d(out_channels, 64, 1),
        nn.ReLU(),
        nn.Conv2d(64, out_channels, 1),
        nn.Sigmoid()
    )

  def forward(self, x, y):
    print(x.shape)
    print(y.shape)
    z = torch.concat([x, y], dim=1)
    z = self.convBnReLU(z)
    z2 = self.layers(z)
    z3 = torch.mul(z, z2)
    z4 = torch.add(z3, z)
    return z4

class ContextPath(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.xception = Xception(in_channels, out_channels)
    self.arm16 = ARM(728, 728)
    self.arm32 = ARM(2048, 2048)

  def forward(self, x):
    #print(x.device)
    x16, x32 = self.xception(x)
    x16 = F.upsample(x16, scale_factor=2)
    x32 = F.upsample(x32, size=[x16.shape[2], x16.shape[3]])

    #print(x32.shape)
    
    out = F.avg_pool2d(x32, [x32.shape[2], x32.shape[3]])
    #print(out.shape)
    out = x32 + out
    #print(x16.shape)
    #print(out.shape)
    x16 = self.arm16(x16)
    x32 = self.arm32(x32)
    #print("--")
    #print(x16.shape)
    #print(x32.shape)
    out = torch.cat([x16, x32, out], dim=1)
    return out, x16, x32

class BiSeNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.spatialPath = SpatialPath(3, 3)
    self.contextPath = ContextPath(3, 3)
    self.ffm = FFM(3, 19) #out should be number of classes
    #for loss calculation:
    self.conv1 = nn.Conv2d(728, 512, kernel_size = 3, stride = 1, padding = 1) 
    self.conv2 = nn.Conv2d(2048, 512, kernel_size = 3, stride = 1, padding = 1)
    

  def forward(self, x):
    x1 = self.spatialPath(x)
    x2 = self.contextPath(x)[0]
    out = self.ffm(x1, x2)
    out = F.upsample(out, [512, 512])#F.upsample(out, scale_factor = 8)
    return out

  def loss(self, input, label):
    x1 = self.spatialPath(input)
    x2, x16, x32 = self.contextPath(input)
    out = self.ffm(x1, x2)
    out = F.upsample(out, [512, 512])
    x16 = self.conv1(x16)
    x32 = self.conv2(x32)
    x16 = F.upsample(x16, [512, 512])
    x32 = F.upsample(x32, [512, 512])

    print(out.shape)
    print(label.shape)

    loss1 = F.cross_entropy(out, label)
    loss2 = F.cross_entropy(x16, label)
    loss3 = F.cross_entropy(x32, label)

    return loss1 + loss2 + loss3

"""
from torchvision import transforms
import os
import gc


model = BiSeNet()

img = Image.open("yol/0.jpg").convert('RGB')

toTensorTransform = transforms.ToTensor()

tensor = toTensorTransform(img)
inputs = tensor.unsqueeze(0)

label_names =  ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']

img = Image.open("yol/00000_"+label_names[0]+".png")

toTensorTransform = transforms.ToTensor()

tensor = toTensorTransform(img)
my_tensor = tensor.unsqueeze(0)
tensor = torch.where(my_tensor == 0, torch.tensor(18), torch.tensor(0))
labels = tensor[:,0,:,:]

for i in range(1, 18):
  if not os.path.exists("yol/00000_"+label_names[i]+".png"):
    continue
  img = Image.open("yol/00000_"+label_names[i]+".png")
  toTensorTransform = transforms.ToTensor()
  tensor = toTensorTransform(img)
  tensor = tensor.unsqueeze(0)
  tensor = tensor[:,0,:,:]
  labels = torch.where(tensor == 0, labels, torch.tensor(i))


lr = 0.9 #2.5e-2
#lr with poly decay!!
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)

model.train()
for i in range(10):
  optimizer.zero_grad()
  loss = model.loss(inputs, labels)
  print(f"epoch {i} loss {loss}")
  loss.backward()
  optimizer.step()
  gc.collect()

model.train()
for i in range(10):
  optimizer.zero_grad()
  loss = model.loss(inputs, labels)
  print(f"epoch {i} loss {loss}")
  loss.backward()
  optimizer.step()
  gc.collect()

outputs = model(inputs)
for i in range(19):
  output = outputs[0,i,:,:]
  output = output.detach().cpu().numpy()
  plt.matshow(output)
  plt.show()

outputs = model(inputs)
for i in range(19):
  output = outputs[0,i,:,:]
  output = output.detach().cpu().numpy()
  plt.matshow(output)
  plt.show()
  """