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
from torchvision.transforms import transforms
# Normalize to [0,1]
def readDatasets():
  sketch_data =np.load("./sketches.pickle", allow_pickle =True)/255.0
  sketch_train = sketch_data[:4000]
  sketch_test = sketch_data[4000:]
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
  photo_test = photo_data[8000:]
  #photo_train, photo_test = torch.utils.data.random_split(photo_data, [8000, 2000])
  #print("Dataset shapes: {} | {} | {} | {}".format(photo_train.shape,photo_test.shape,sketch_train.shape,sketch_test.shape))
  return photo_train, photo_test, sketch_train, sketch_test

def getGenerators():
  a = Generator(1,3) #sketch->photo
  b = Generator(3,1) #photo->sketch
  if "genA.pt" in os.listdir("."):
    a.load_state_dict(torch.load("./genA.pt"))
  if "genB.pt" in os.listdir("."):
    b.load_state_dict(torch.load("./genB.pt"))
  return a, b
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
  testA_loader = data.DataLoader(torch.Tensor(test_dataA),batch_size=B,shuffle=False)
  testB_loader = data.DataLoader(torch.Tensor(test_dataB),batch_size=B,shuffle=False)
  
  
  # Init Models
  genA,genB = getGenerators()
  genA = genA.cuda()
  genB = genB.cuda()
  samples = []
  with torch.no_grad():
    for x in testA_loader:
      y = genB(x.cuda())
      samples.append([x,y.cpu()])
      break

    for x in trainA_loader:
      y = genB(x.cuda())
      samples.append([x,y.cpu()])
      break
  x_samples = torch.cat([samples[0][0],samples[1][0]],dim = 0).numpy()
  y_samples = torch.cat([samples[0][1],samples[1][1]],dim = 0).numpy()
  with open("xA_samples.pickle","wb") as f:
    pkl.dump(x_samples,f)
  with open("yA_samples.pickle","wb") as f:
    pkl.dump(y_samples,f)
  samples = []
  with torch.no_grad():
    for x in testB_loader:
      y = genA(x.cuda())
      samples.append([x,y.cpu()])
      break

    for x in trainB_loader:
      y = genA(x.cuda())
      samples.append([x,y.cpu()])
      break
  x_samples = torch.cat([samples[0][0],samples[1][0]],dim = 0).numpy()
  y_samples = torch.cat([samples[0][1],samples[1][1]],dim = 0).numpy()
  with open("xB_samples.pickle","wb") as f:
    pkl.dump(x_samples,f)
  with open("yB_samples.pickle","wb") as f:
    pkl.dump(y_samples,f)

if __name__ == '__main__':
  main()
