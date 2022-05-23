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
import pickle as pkl
import torch.optim as optim

def calc_loss(main_gen,other_gen,main_discriminators,other_discriminators,CLIP_model,faceParsingNet,x):
  y = main_gen(x)
  x_hat = other_gen(y)
  parsing_x, features_x = getFaceParsingOutput(x,faceParsingNet)
  parsing_y, features_y = getFaceParsingOutput(y,faceParsingNet)
  loss1 = (x - x_hat).mean() # L1 distance cycle consistency
  loss2 = torch.square(CLIP.encode_image(x) - CLIP.encode_image(y)).mean() # L2 distance of CLIP embeddings
  loss3 = torch.square(features_x - features_y).mean() #L2 distance of Face Parsing Net Features/Embeddings
  # For other_discriminators x is real data
  loss_other_d = 1 - other_discriminators[0](x)
  for i in range(1,len(other_discriminators)):
    loss_other_d += (1 - other_discriminators[i](x * parsing_x[i-1]))
  
  # For main_discriminators y is fake data
  loss_main_d = main_discriminators[0](y)
  for i in range(1,len(main_discriminators)):
    loss_main_d += main_discriminators[i](y * parsing_y[i-1])
  # For main_gen main_discriminators will give adversarial loss -> use - main_disc loss
  loss_main_g = 1 - loss_main_d
  
  lossMainGen = loss1 + loss2 + loss3 + loss_main_g
  lossMainDisc = loss_main_d
  lossOtherDisc = loss_other_d
  return lossMainGen,lossMainDisc,lossOtherDisc

def train(genA,genB,discA,discB,iterA,iterB,optimizerGenA,optimizerGenB,optimizerDiscA,optimizerDiscB,CLIP_model,faceParsingNet):
  genA.train()
  genB.train()
  (i.train() for i in discA)
  (i.train() for i in discB)
  lossesA = []
  lossesB = []
  while a_cont or b_cont:
    a = next(iterA).cuda()
    if a is None:
      a_cont = False
    if a_cont:
      optimizerGenA.zero_grad()
      optimizerGenB.zero_grad()
      optimizerDiscA.zero_grad()
      optimizerDiscB.zero_grad()
      lossG,lossD_1,lossD_2 = calc_loss(genB,genA,discB,discA,CLIP_model,faceParsingNet,a)
      optimizerGenA.zero_grad()
      lossG.backward()
      optimizerGenB.step()
      optimizerDiscB.zero_grad()
      lossD_1.backward()
      optimizerDiscB.step()
      lossD_2.backward()
      optimizerDiscA.step()
      lossesA.append(tuple(lossG,lossD_1,lossD_2))
    b = next(iterB).cuda()
    if b is None:
      b_cont = False
      continue
    optimizerGenA.zero_grad()
    optimizerGenB.zero_grad()
    optimizerDiscA.zero_grad()
    optimizerDiscB.zero_grad()
    lossG,lossD_1,lossD_2 = calc_loss(genA,genB,discA,discB,CLIP_model,faceParsingNet,b)
    optimizerGenB.zero_grad()
    lossG.backward()
    optimizerGenA.step()
    optimizerDiscA.zero_grad()
    lossD_1.backward()
    optimizerDiscA.step()
    lossD_2.backward()
    optimizerDiscB.step()
    lossesB.append(tuple(lossG,lossD_1,lossD_2))
  return lossesA,lossesB

def eval_model(genA,genB,discA,discB,testA_loader,testB_loader,CLIP_model,faceParsingNet):
  genA.eval()
  genB.eval()
  (i.eval() for i in discA)
  (i.eval() for i in discB)
  lossesA = []
  lossesB = []
  with torch.no_grad():
    for a in tqdm(testA_loader):
      a = a.cuda()
      loss = calc_loss(genB, genA, discB, discA, CLIP_model, faceParsingNet, a)
      lossesA.append(tuple([i.item() for i in loss]))
    for b in tqdm(testB_loader):
      b = b.cuda()
      loss = calc_loss(genA, genB, discA, discB, CLIP_model, faceParsingNet, b)
      lossesB.append(tuple([i.item() for i in loss]))
    return lossesA, lossesB


# Normalize to [0,1]
def readDatasets():
  sketch_data = np.load("../sketches.pickle", allow_pickle =True)
  sketch_train, sketch_test = torch.utils.data.random_split(sketch_data, [4000, 1000]) 
  image_files = os.listdir('/datasets/ffhq/images1024x1024/')
  """
  photo_data = torch.zeros(10000, 3, 256, 256)
  for i in tqdm(range(10000)):
    img = cv2.imread('/datasets/ffhq/images1024x1024/' + image_files[i])
    img = cv2.resize(img,(256,256))
    photo_data[i] = torch.from_numpy(np.moveaxis(img,2,0)).unsqueeze(0)
    if i==10000:
      break
  torch.save(photo_data, "../photos.pt")
  """ 
  photo_data = torch.load("../photos.pt").numpy()  
  photo_train, photo_test = torch.utils.data.random_split(photo_data, [8000, 2000])
  return photo_train, photo_test, sketch_train, sketch_test

def getGenerators():
  a = Generator(1,3)
  b = Generator(3,1)
  if "genA.pt" in os.listdir("."):
    a.load_state_dict(torch.load("./genA.pt"))
  if "genB.pt" in os.listdir("."):
    b.load_state_dict(torch.load("./genB.pt"))
  return a.cuda(),b.cuda()

def getDiscriminators(num_disc):
  a = [Discriminator(3).cuda() for i in range(num_disc)]
  b = [Discriminator(1).cuda() for i in range(num_disc)]

  for i in range(num_disc):
    if "discA{}.pt".format(i) in os.listdir("."):
      a.load_state_dict(torch.load("./discA{}.pt".format(i)))
    if "discB{}.pt".format(i) in os.listdir("."):
      b.load_state_dict(torch.load("./discB{}.pt".format(i)))
  return a,b

def getFaceParsingOutput(x,face_parsing_net):
  if x.shape[1]==1:
    x = x.repeat(1,3,1,1)/3
  parsing, features = face_parsing_net(x)
  return getMasksFromParsing(parsing, x.shape[1]==3), features

def main():
  B = 8
  epochs = 10
  lr = 1e-3

  #Load Datasets
  train_dataA, test_dataA, train_dataB, test_dataB = readDatasets()
  trainA_loader = data.DataLoader(torch.Tensor(train_dataA),batch_size=B,shuffle=True) 
  trainB_loader = data.DataLoader(torch.Tensor(train_dataB),batch_size=B,shuffle=True)
  testA_loader = data.DataLoader(torch.Tensor(test_dataA),batch_size=B,shuffle=False)
  testB_loader = data.DataLoader(torch.Tensor(test_dataB),batch_size=B,shuffle=False)
  
  
  # Init Models
  genA,genB = getGenerators()
  discriminator_count = 4
  discriminatorsA,discriminatorsB = getDiscriminators(discriminator_count)
  faceParsingNet = getParsingNetwork()
  device = "cuda" if torch.cuda.is_available() else "cpu"  
  CLIP,_ = clip.load("ViT-B/32",device=device,jit=False)
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
  l = eval_model(genA,genB,discriminatorsA,discriminatorsB,testA_loader,testB_loader,CLIP,faceParsingNet)
  eval_losses.append(l)
  for i in range(epochs):
    trainAIterator = iter(trainA_loader)
    trainBIterator = iter(trainB_loader)
    l = train(genA,genB,discriminatorsA,discriminatorsB,trainAIterator,trainBIterator,optimizerGenA,optimizerGenB,optimizerDiscA,optimizerDiscB,CLIP,faceParsingNet)
    train_losses.append(l)
    l = eval_model(genA,genB,discriminatorsA,discriminatorsB,testA_loader,testB_loader,CLIP,faceParsingNet)
    eval_losses.append(l)

  
  torch.save(genA.state_dict(),"./genA.pt")
  torch.save(genB.state_dict(),"./genB.pt")
  for i in range(discriminator_count):
    torch.save(discriminatorsA[i].state_dict(),"./discA{}.pt".format(i))
    torch.save(discriminatorsB[i].state_dict(),"./discB{}.pt".format(i))




if __name__ == '__main__':
  main()
