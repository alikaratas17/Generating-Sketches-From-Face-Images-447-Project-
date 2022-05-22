import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import torch.utils.data as data
import os
from tqdm import tqdm
from models.Generator import Generator
from models.Discriminator import Discriminator
from FaceParsingNetwork.face_parsing import getParsingNetwork
from util import getMasksFromParsing
import clip
import cv2


def calc_loss(main_gen,other_gen,main_discriminators,other_discriminators,CLIP_model,faceParsingNet,x):
  y = main_gen(x)
  x_hat = other_gen(y)
  parsing_x, features_x = faceParsingNet(x)
  parsing_y, features_y = faceParsingNet(y)
  loss1 = (x - x_hat).mean() # L1 distance cycle consistency
  loss2 = torch.square(CLIP.encode_image(x) - CLIP.encode_image(y)).mean() # L2 distance of CLIP embeddings
  loss3 = torch.square(features_x-features_y).mean() #L2 distance of Face Parsing Net Features/Embeddings
  # For other_discriminators x is real data
  # For main_discriminators y is fake data
  # For main_gen main_discriminators will give adversarial loss
  return lossMainGen,lossMainDisc,lossOtherDisc

def train(genA,genB,discA,discB,iterA,iterB,optimizers):
  genA.train()
  genB.train()
  discA.train()
  discB.train()
  losses = []
  while 1:
    a = next(iterA)




  for x in tqdm(train_loader):
    x = x.cuda()
    loss = model.loss(x,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
  return losses

def eval_model(genA,genB,discA,discB,testA_loader,testB_loader):
  genA.eval()
  genB.eval()
  discA.eval()
  discB.eval()
  lossesA = []
  with torch.no_grad:
    for x in tqdm(testA_loader):
      x = x.cuda()
      loss = calc_loss() # TODO
      lossesA.append(tuple([a.item() for a in loss]))
    for x in tqdm(testB_loader):
      x = x.cuda()
      loss = calc_loss() # TODO
      lossesB.append(tuple([a.item() for a in loss]))
    return lossesA,lossesB

def readDatasets():
  sketch_data = np.load("../sketches.pickle", allow_pickle =True)
  sketch_train, sketch_test = torch.utils.data.random_split(sketch_data, [4000, 1000])
  image_files = os.listdir('/datasets/ffhq/images1024x1024/')
  i=0
  images = []
  for image in tqdm(image_files):
    i+=1
    img = cv2.imread('/datasets/ffhq/images1024x1024/' + image)
    img = cv2.resize(img,(256,256))
    images.append(torch.from_numpy(np.moveaxis(img,2,0)).unsqueeze(0))
    photo_data = torch.cat(images,dim=0).numpy()
    if i==10000:
      break
    
  photo_train, photo__test = torch.utils.data.random_split(sketch_data, [8000, 2000])
  print(torch.Tensor(photo_train).shape)
  return photo_train, photo_test, sketch_train, sketch_test

def getGenerators():
  a = Generator(1,3)
  b = Generator(3,1)
  if "genA.pt" in os.listdir("."):
    a.load_state_dict(torch.load("./genA.pt"))
  if "genB.pt" in os.listdir("."):
    b.load_state_dict(torch.load("./genB.pt"))
  return a,b

def getDiscriminators(num_disc):
  a = [Discriminator(3) for i in range(num_disc)]
  b = [Discriminator(1) for i in range(num_disc)]

  for i in range(discriminator_count):
    if "discA{}.pt".format(i) in os.listdir("."):
      a.load_state_dict(torch.load("./discA{}.pt".format(i)))
    if "discB{}.pt".format(i) in os.listdir("."):
      b.load_state_dict(torch.load("./discB{}.pt".format(i)))
  return a,b

def getFaceParsingNet(x, isRGB = True):
  parsing, features = faceParsingNet(x)
  return getMasksFromParsing(parsing, isRGB), features

def main():
  B = 8
  epochs = 10
  lr = 1e-3

  #Load Datasets
  train_dataA, test_dataA, train_dataB, test_dataB = readDatasets()
  trainAIterator = iter(data.DataLoader(torch.Tensor(train_dataA),batch_size=B,shuffle=True))
  trainBIterator = iter(data.DataLoader(torch.Tensor(train_dataB),batch_size=B,shuffle=True))
  testA_loader = data.DataLoader(torch.Tensor(test_dataA),batch_size=B,shuffle=False)
  testB_loader = data.DataLoader(torch.Tensor(test_dataB),batch_size=B,shuffle=False)

  # Init Models
  genA,genB = getGenerators()
  discriminator_count = 4
  discriminatorsA,discriminatorsB = getDiscriminators(discriminator_count)
  faceParsingNet = getParsingNetwork()
  CLIP,_ = clip.load("ViT-B/32",device="cuda",jit=False)
  clip.model.convert_weights(CLIP) # use CLIP.encode_image() for clip loss

  # Init Optimizers
  optimizerGenA = optim.Adam(genA.parameters(),lr = lr)
  optimizerGenB = optim.Adam(genB.parameters(),lr = lr)
  paramsA = []
  for x in discriminatorsA:
    paramsA += list(x.parameters())
  optimizerDiscA = optim.Adam(paramsA,lr = lr)
  paramsB = []
  for x in discriminatorsB:
    paramsB += list(x.parameters())
  optimizerDiscB = optim.Adam(paramsB,lr = lr)

  # Training Loop
  eval_losses = []
  train_losses = []
  l = eval_model() # TODO
  eval_losses.append(l)
  for i in range(epochs):
    l = train() # TODO
    train_losses.append(l)
    l = eval_model() # TODO
    eval_losses.append(l)

  
  torch.save(genA.state_dict(),"./genA.pt")
  torch.save(genB.state_dict(),"./genB.pt")
  for i in range(discriminator_count):
    torch.save(discriminatorsA[i].state_dict(),"./discA{}.pt".format(i))
    torch.save(discriminatorsB[i].state_dict(),"./discB{}.pt".format(i))




if __name__ == '__main__':
  main()
