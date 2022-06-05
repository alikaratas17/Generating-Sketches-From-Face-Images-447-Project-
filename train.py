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
from BiSeNet.BiSeNet import BiSeNet
from util import getMasksFromParsing
import clip
import cv2
import pickle as pkl
import torch.optim as optim
from torchvision.transforms import transforms
import sys
from SynergyNet.model_building import SynergyNet
from SynergyNet.FaceBoxes import FaceBoxes
#import argparse
use_synergy_net = False

def calc_loss_train(main_gen,other_gen,main_discriminators,other_discriminators,CLIP_model,faceParsingNet,x, bisenet,SnetModels,loss_num):
  if loss_num == 1:
    y = main_gen(x)
    x_hat = other_gen(y)
    parsing_x, _ = getFaceParsingOutput(x,faceParsingNet)
    parsing_y, _ = getFaceParsingOutput(y,faceParsingNet)
    loss1 = (x - x_hat).mean() # L1 distance cycle consistency
    if x.shape[1]==3:
      image_x = x[:,:,16:-16,16:-16].cuda()
      image_y = y[:,:,16:-16,16:-16].cuda()
      if image_x.shape[1]==1:
        image_x = image_x.repeat(1,3,1,1)
      else:
        image_y = image_y.repeat(1,3,1,1)
      clip_x_embed = CLIP_model.encode_image(image_x)
      clip_y_embed = CLIP_model.encode_image(image_y)
      loss2 = torch.sqrt(torch.square(clip_x_embed - clip_y_embed).view(clip_x_embed.shape[0], -1).mean(dim=1)).mean()  # L2 distance of CLIP embeddings
      if use_synergy_net:
        loss4 = synergynetLoss(SnetModels,x,y)
      else:
        loss4 = 0.0
      bisenet_x = getBisenetOutput(x, bisenet)
      bisenet_y = getBisenetOutput(y.repeat(1,3,1,1), bisenet)
      #loss3 = 0.0
      loss3 = torch.sqrt(torch.square(bisenet_x - bisenet_y).view(bisenet_x.shape[0], -1).mean(dim=1)).mean()  # L2 distance of bisenet embeddings
    else:
      loss2 = 0.0
      loss3 = 0.0
      loss4 = 0.0
    loss_main_g = main_discriminators[0](y)
    for i in range(1,len(main_discriminators)):
      loss_main_g += main_discriminators[i](y * parsing_y[i-1])
    loss_main_g = loss_main_g / len(main_discriminators)
    loss_main_g = - (loss_main_g + 1e-12).log().mean() #burcu: olasi problem (bi ust satira demis)
    loss1 = loss1 * 1e-1 #weight cycle consistency by 1e-2
    loss2 = loss2 * 1.0  #weight CLIP loss by 1e-1
    loss3 = loss3 * 1.0
    loss4 = loss4 * 1.0
    loss_main_g = loss_main_g * 1e1
    lossMainGen = loss1 + loss2 + loss3 + loss4 + loss_main_g
    print(loss1)
    print(loss1.requires_grad)
    print(loss2.requires_grad)
    print(loss3.requires_grad)
    return lossMainGen
    #return loss3
  if loss_num ==2:    
    parsing_x, features_x = getFaceParsingOutput(x,faceParsingNet)
    # For other_discriminators x is real data
    pred_other_d = other_discriminators[0](x) 
    for i in range(1,len(other_discriminators)):
      pred_other_d += other_discriminators[i](x * parsing_x[i-1])
    pred_other_d = pred_other_d / len(other_discriminators)
    loss_other_d = (1 - pred_other_d + 1e-12).log().mean() *1e1
    return loss_other_d
  if loss_num ==3: 
    y = main_gen(x)
    parsing_y, features_y = getFaceParsingOutput(y,faceParsingNet)
    # For main_discriminators y is fake data
    loss_main_d = main_discriminators[0](y)
    for i in range(1,len(main_discriminators)):
      loss_main_d += main_discriminators[i](y * parsing_y[i-1])
    loss_main_d = loss_main_d / len(main_discriminators)
    loss_main_d = (loss_main_d + 1e-12).log().mean() * 1e1
    return loss_main_d


def train(genA,genB,discA,discB,iterA,iterB,optimizerGenA,optimizerGenB,optimizerDiscA,optimizerDiscB,CLIP_model,faceParsingNet,bisenet,SnetModels):
  genA.train()
  genB.train()
  [i.train() for i in discA]
  [i.train() for i in discB]
  lossesA = []
  lossesB = []
  c = min(len(iterA),len(iterB))
  for _ in tqdm(range(c)):
    a = iterA.next()
    a = a.cuda()
    optimizerGenA.zero_grad()
    optimizerGenB.zero_grad()
    optimizerDiscA.zero_grad()
    optimizerDiscB.zero_grad()
    lossG = calc_loss_train(genB,genA,discB,discA,CLIP_model,faceParsingNet,a,bisenet,SnetModels,1)
    lossG.backward()
    optimizerGenB.step()
    optimizerGenA.step()
    optimizerGenA.zero_grad()
    optimizerGenB.zero_grad()
    optimizerDiscA.zero_grad()
    optimizerDiscB.zero_grad()
    lossD_1 = calc_loss_train(genB,genA,discB,discA,CLIP_model,faceParsingNet,a,bisenet,SnetModels,3)
    lossD_1.backward()
    optimizerDiscB.step()
    optimizerGenA.zero_grad()
    optimizerGenB.zero_grad()
    optimizerDiscA.zero_grad()
    optimizerDiscB.zero_grad()
    lossD_2 = calc_loss_train(genB,genA,discB,discA,CLIP_model,faceParsingNet,a,bisenet,SnetModels,2)
    lossD_2.backward()
    optimizerDiscA.step()
    lossesA.append((lossG.item(),lossD_1.item(),lossD_2.item()))

    b = iterB.next()
    b = b.cuda()
    optimizerGenA.zero_grad()
    optimizerGenB.zero_grad()
    optimizerDiscA.zero_grad()
    optimizerDiscB.zero_grad()
    lossG = calc_loss_train(genA,genB,discA,discB,CLIP_model,faceParsingNet,b,bisenet,SnetModels,1)
    lossG.backward()
    optimizerGenA.step()
    optimizerGenB.step()
    optimizerGenA.zero_grad()
    optimizerGenB.zero_grad()
    optimizerDiscA.zero_grad()
    optimizerDiscB.zero_grad()
    lossD_1 = calc_loss_train(genA,genB,discA,discB,CLIP_model,faceParsingNet,b,bisenet,SnetModels,3)
    lossD_1.backward()
    optimizerDiscA.step()
    optimizerGenA.zero_grad()
    optimizerGenB.zero_grad()
    optimizerDiscA.zero_grad()
    optimizerDiscB.zero_grad()
    lossD_2 = calc_loss_train(genA,genB,discA,discB,CLIP_model,faceParsingNet,b,bisenet,SnetModels,2)
    lossD_2.backward()
    optimizerDiscB.step()
    lossesB.append((lossG.item(),lossD_1.item(),lossD_2.item()))
  return lossesA,lossesB

# Normalize to [0,1]
def readDatasets():
  sketch_data =  np.load("./sketches.pickle", allow_pickle =True)/255.0
  sketch_train = sketch_data[:4000]
  #sketch_test = sketch_data[4000:]
  #sketch_train, sketch_test = torch.utils.data.random_split(sketch_data, [4000, 1000]) 
  #image_files = os.listdir('/datasets/ffhq/images1024x1024/')
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
  photo_data = torch.load("./photos.pt").numpy()/255.0
  #print(photo_data.max())
  #print(sketch_data.max())
  photo_train = photo_data[:8000]
  #photo_test = photo_data[8000:]
  #photo_train, photo_test = torch.utils.data.random_split(photo_data, [8000, 2000])
  #print("Dataset shapes: {} | {} | {} | {}".format(photo_train.shape,photo_test.shape,sketch_train.shape,sketch_test.shape))
  return photo_train, None, sketch_train, None

def getGenerators():
  a = Generator(1,3) #sketch->photo
  b = Generator(3,1) #photo->sketch
  if "1genA.pt" in os.listdir("."):
    a.load_state_dict(torch.load("./1genA.pt"))
    print("loaded")
  if "1genB.pt" in os.listdir("."):
    b.load_state_dict(torch.load("./1genB.pt"))
  return a, b

def getDiscriminators(num_disc):
  a = [Discriminator(3) for i in range(num_disc)] #discriminate photo
  b = [Discriminator(1) for i in range(num_disc)] #discriminate sketch

  for i in range(num_disc):
    if "1discA{}.pt".format(i) in os.listdir("."):
      a[i].load_state_dict(torch.load("./1discA{}.pt".format(i)))
    if "1discB{}.pt".format(i) in os.listdir("."):
      b[i].load_state_dict(torch.load("./1discB{}.pt".format(i)))
  #a = [x.cuda() for x in a]
  #b = [x.cuda() for x in b]
  return a,b

def getFaceParsingOutput(x,face_parsing_net):
  x_shape = x.shape[1]
  if x.shape[1]==1:
    x = x.repeat(1,3,1,1)
  parsing, features = face_parsing_net(x)
  return getMasksFromParsing(parsing, x_shape==3), features
def getSynergyNetAndFaceBoxes():
  
  args = {}
  # load pre-tained model
  checkpoint_fp = './SynergyNet/pretrained/best.pth.tar' 
  args['arch'] = 'mobilenet_v2'
  args['devices_id'] = [0]
  args['img_size'] = 120
  checkpoint = torch.load(checkpoint_fp, map_location=lambda storage, loc: storage)['state_dict']
  model = SynergyNet(args)
  model_dict = model.state_dict()
  # because the model is trained by multiple gpus, prefix 'module' should be removed
  for k in checkpoint.keys():
    model_dict[k.replace('module.', '')] = checkpoint[k]
  model.load_state_dict(model_dict, strict=False)
  model = model.cuda()
  # face detector
  face_boxes = FaceBoxes()
  return model,face_boxes
def synergynetLoss(SnetModels,x,y):
  model,face_boxes  = SnetModels
  x_samples = torch.zeros(x.shape[0],x.shape[1],120,120)
  y_samples = torch.zeros(x.shape[0],x.shape[1],120,120)
  for i in range(x.shape[0]):
    x_img = (np.moveaxis(x[i].detach().cpu().numpy(),0,2)*255).astype(np.uint8)
    rectx = face_boxes(x_img)
    #y_img = (np.moveaxis(y[i].detach().numpy(),1,3)*255).astype(np.uint8)
    #recty = face_boxes(y_img)
    rect = rectx[0]
    HCenter = (rect[1] + rect[3])/2
    WCenter = (rect[0] + rect[2])/2
    Hbeginning  = int(HCenter) - 60
    Hend = Hbeginning + 120
    if Hbeginning < 0:
      Hend -= Hbeginning
      Hbeginning= 0
    if Hend > 255:
      Hbeginning -= (Hend - 255)
      Hend = 255
    Wbeginning  = int(WCenter) - 60
    Wend = Wbeginning + 120
    if Wbeginning < 0:
      Wend -= Wbeginning
      Wbeginning= 0
    if Wend > 255:
      Wbeginning -= (Wend - 255)
      Wend = 255
    x_samples[i] = x[i,:,Hbeginning:Hend,Wbeginning:Wend]
    y_samples[i] = x[i,:,Hbeginning:Hend,Wbeginning:Wend]
  x_samples = x_samples.cuda()
  y_samples = y_samples.cuda()
  x_samples = ((x_samples * 255)-127.5)/128
  y_samples = ((y_samples * 255)-127.5)/128
  z_x = model.forward_test(x_samples)
  z_y = model.forward_test(y_samples)
  return torch.sqrt(torch.square(z_x - z_y).view(z_x.shape[0], -1).mean(dim=1)).mean() #L2 distance of synergynet outputs
def getBisenetOutput(x, bisenet):
  return bisenet(x)

def main():
  torch.autograd.set_detect_anomaly(True)
  B=8
  epochs = 10
  lr = 1e-3

  #Load Datasets
  train_dataA, test_dataA, train_dataB, test_dataB = readDatasets()
  #train_dataA = train_dataA[:100]
  #train_dataB = train_dataB[:100]
  #test_dataA = test_dataA[:100]
  #test_dataB= test_dataB[:100]
  
  trainA_loader = data.DataLoader(torch.Tensor(train_dataA),batch_size=B,shuffle=True) 
  trainB_loader = data.DataLoader(torch.Tensor(train_dataB),batch_size=B,shuffle=True)
  #testA_loader = data.DataLoader(torch.Tensor(test_dataA),batch_size=B,shuffle=False)
  #testB_loader = data.DataLoader(torch.Tensor(test_dataB),batch_size=B,shuffle=False)
  
  
  # Init Models
  genA,genB = getGenerators()
  genA = genA.cuda()
  genB = genB.cuda()
  discriminator_count = 4
  discriminatorsA,discriminatorsB = getDiscriminators(discriminator_count)
  for i in range(discriminator_count):
    discriminatorsA[i] = discriminatorsA[i].cuda()
    discriminatorsB[i] = discriminatorsB[i].cuda()
  faceParsingNet = getParsingNetwork().cuda()
  #device = "cuda" if torch.cuda.is_available() else "cpu"  
  CLIP,preprocess = clip.load("ViT-B/32",device="cuda", jit=False)
  clip.model.convert_weights(CLIP) # use CLIP.encode_image() for clip loss
  bisenet = BiSeNet().cuda()
  SnetModels = getSynergyNetAndFaceBoxes()
  #print("genA : {}".format(genA))
  #print("genB : {}".format(genB))
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
  exp_id = int(sys.argv[1])
  # Training Loop
  train_losses = []
  for i in range(epochs):
    print("{}th epoch is starting".format(i))
    l = train(genA,genB,discriminatorsA,discriminatorsB,iter(trainA_loader),iter(trainB_loader),optimizerGenA,optimizerGenB,optimizerDiscA,optimizerDiscB,CLIP,faceParsingNet,bisenet,SnetModels)
    train_losses.append(l)
    print("Train Loss: {}".format([np.array(x).mean(axis=0) for x in l]))

    torch.save(genA.state_dict(),"./gen{}A.pt".format(exp_id))
    torch.save(genB.state_dict(),"./gen{}B.pt".format(exp_id))
    for j in range(discriminator_count):
      torch.save(discriminatorsA[j].state_dict(),"./discA{}{}.pt".format(exp_id, j))
      torch.save(discriminatorsB[j].state_dict(),"./discB{}{}.pt".format(exp_id, j))


if __name__ == '__main__':
  main()
  
