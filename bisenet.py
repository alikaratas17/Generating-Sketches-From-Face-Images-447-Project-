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
from scipy.stats import norm
from torchvision import transforms
import os
import gc
from PIL import Image
import timm
import tqdm

class Xception(nn.Module):
  def __init__(self, in_channels, out_channels):
    super().__init__()
    self.model = timm.create_model('xception', pretrained=True, features_only=True)
  def forward(self, x):
    #print(x.shape)
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
    #print(x.shape)
    #print(y.shape)
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
    self.spatialPath = SpatialPath(1, 1)
    self.contextPath = ContextPath(1, 1)
    self.ffm = FFM(1, 19) #out should be number of classes
    #for loss calculation:
    self.conv1 = nn.Conv2d(728, 256, kernel_size = 3, stride = 1, padding = 1) 
    self.conv2 = nn.Conv2d(2048, 256, kernel_size = 3, stride = 1, padding = 1)
    

  def forward(self, x):
    x1 = self.spatialPath(x)
    x2 = self.contextPath(x)[0]
    out = self.ffm(x1, x2)
    out = F.upsample(out, [256, 256])#F.upsample(out, scale_factor = 8)
    return out

  def loss(self, input, label):
    x1 = self.spatialPath(input)
    x2, x16, x32 = self.contextPath(input)
    out = self.ffm(x1, x2)
    out = F.upsample(out, [256, 256])
    x16 = self.conv1(x16)
    x32 = self.conv2(x32)
    x16 = F.upsample(x16, [256, 256])
    x32 = F.upsample(x32, [256, 256])

   # print(out.shape)
    #print(label)

    loss1 = F.cross_entropy(out, label)
    loss2 = F.cross_entropy(x16, label)
    loss3 = F.cross_entropy(x32, label)

    return loss1 + loss2 + loss3

def train(model): 
  """
  inputs = torch.zeros(12000,256,256)
  labels = torch.zeros(12000,256,256)
  toTensorTransform = transforms.ToTensor()
  label_names =  ['skin', 'nose', 'eye_g', 'l_eye', 'r_eye', 'l_brow', 'r_brow', 'l_ear', 'r_ear', 'mouth', 'u_lip', 'l_lip', 'hair', 'hat', 'ear_r', 'neck_l', 'neck', 'cloth']
  #init
  img = Image.open("/datasets/CelebAMask-HQ/CelebA-HQ-img/0.jpg").convert('L').resize((256,256))
  image_tensor = toTensorTransform(img)
  inputs_first = image_tensor
  inputs[0] = inputs_first
  
  label_img = Image.open("/datasets/CelebAMask-HQ/CelebAMask-HQ-mask-anno/0/00000_"+label_names[0]+".png").resize((256,256))
  class_tensor = toTensorTransform(label_img)
  class_tensor = class_tensor.unsqueeze(0)
  tensor = torch.where(class_tensor == 0, torch.tensor(18), torch.tensor(0))
  labels_tensor = tensor[:,0,:,:]
  
  for i in range(1, 18):
    if not os.path.exists("/datasets/CelebAMask-HQ/CelebAMask-HQ-mask-anno/0/00000_"+label_names[i]+".png"):
      continue
    label_img = Image.open("/datasets/CelebAMask-HQ/CelebAMask-HQ-mask-anno/0/00000_"+label_names[i]+".png").resize((256,256))
    class_tensor = toTensorTransform(label_img)
    class_tensor = class_tensor.unsqueeze(0)
    class_tensor = class_tensor[:,0,:,:]
    labels_tensor = torch.where(class_tensor == 0, labels_tensor, torch.tensor(i))
  
  labels[0] = labels_tensor
  #fill
  for i in range(1,12000):
    k = i//2000
    i_str = f'{i:05d}'
    img = Image.open("/datasets/CelebAMask-HQ/CelebA-HQ-img/{}.jpg".format(i)).convert('L').resize((256,256))
    image_tensor = toTensorTransform(img)
    image_tensor = image_tensor
    inputs[i] = image_tensor
    
    
    label_img = Image.open("/datasets/CelebAMask-HQ/CelebAMask-HQ-mask-anno/{}/{}_".format(k,i_str)+label_names[0]+".png").resize((256,256))
    class_tensor = toTensorTransform(label_img)
    class_tensor = class_tensor.unsqueeze(0)
    tensor = torch.where(class_tensor == 0, torch.tensor(18), torch.tensor(0))
    labels_tensor_tmp = tensor[:,0,:,:]
    for j in range(1, 18):
      if not os.path.exists("/datasets/CelebAMask-HQ/CelebAMask-HQ-mask-anno/{}/{}_".format(k,i_str)+label_names[j]+".png"):
        continue
      label_img = Image.open("/datasets/CelebAMask-HQ/CelebAMask-HQ-mask-anno/{}/{}_".format(k,i_str)+label_names[j]+".png").resize((256,256))
      class_tensor = toTensorTransform(label_img)
      class_tensor = class_tensor.unsqueeze(0)
      class_tensor = class_tensor[:,0,:,:]
      labels_tensor_tmp = torch.where(class_tensor == 0, labels_tensor_tmp, torch.tensor(j))
    
    labels[i] = labels_tensor_tmp
    print(i, flush=True)
  print(inputs.shape, flush = True)
  torch.save(inputs, "./inputs.pt")
  torch.save(labels, "./labels.pt")
  """
  B=16
  inputs = torch.Tensor(torch.load("./inputs.pt")).unsqueeze(1) 
  labels = torch.Tensor(torch.load("./labels.pt"))
  train_inputs_loader = data.DataLoader(inputs,batch_size=B,shuffle=True) 
  train_labels_loader = data.DataLoader(labels,batch_size=B,shuffle=True)  
  lr0 = lr_decay(0)
  optimizer = optim.SGD(model.parameters(), lr=lr0)
  scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: lr_decay(step)/lr0)
  print(inputs.shape)
  print(labels.shape)
  for i in range(200):
    train_epochs(i, iter(train_inputs_loader), iter(train_labels_loader), model, optimizer, scheduler)
    if i%50 == 49:
      torch.save(model.state_dict(), "./bisenet.pt")
  """
  outputs = model(inputs.cuda())
  np.save("./outputs", outputs.detach().cpu().numpy())
  for i in range(19):
    output = outputs[0,i,:,:]
    output = output.detach().cpu().numpy()
  """
  torch.save(model.state_dict(), "./bisenet.pt")     

def train_epochs(i, iter_inputs, iter_labels, model, optimizer, scheduler):
  model.train()
  c = min(len(iter_inputs),len(iter_labels))
  for d in range(c):
    inputs = iter_inputs.next().cuda()
    labels = iter_labels.next().long().cuda()
    optimizer.zero_grad()
    loss = model.loss(inputs, labels)
    if d % 100 == 0: 
      print("epoch: {}, batch: {}, loss: {} lr: {}".format(i,d,loss,optimizer.param_groups[0]['lr']),flush=True)
    loss.backward()
    optimizer.step()
    gc.collect()
  scheduler.step()

def lr_decay(global_step,
    init_learning_rate = 2.5e-2,
    min_learning_rate = 1e-5,
    decay_rate = 0.9999):
    lr = ((init_learning_rate - min_learning_rate) *
          pow(decay_rate, global_step) +
          min_learning_rate)
    return lr

def main():
    model = BiSeNet()
    #model.load_state_dict(torch.load("./bisenet.pt"))
    model = model.cuda()
    #sketch_data = (torch.Tensor(np.load("./sketches.pickle", allow_pickle =True)/255.0).repeat(1,3,1,1))
    #inputs = sketch_data[:10]
    #outputs = model(inputs.cuda())
    #np.save("./outputs", outputs.detach().cpu().numpy())  
    #return
    train(model)
    
if __name__ == '__main__':
  print("main", flush=True)
  main()
