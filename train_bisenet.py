import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import torch.utils.data as data
import os
from tqdm import tqdm
from bisenet import BiSeNet

# Returns train and test data
def readDatasets():
  pass

# Returns loss: idea for now is to use cross entropy loss
def bisenet_loss(y, y_pred):
  pass

def train(model,train_loader,optimizer):
  model.train()
  losses = []
  for x,y in tqdm(train_loader):
    x = x.cuda()
    loss = model.loss(x,y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.item())
  return losses

def eval_model(model,test_loader):
  model.eval()
  losses = []
  for x,y in tqdm(train_loader):
    x = x.cuda()
    loss = model.loss(x,y)
    losses.append(loss.item())
  return np.mean(losses)

def main():
  B = 8
  epochs = 10
  lr = 1e-3
  train_data,test_data = readDatasets()
  train_loader = data.DataLoader(torch.Tensor(train_data),batch_size=B,shuffle=True,pin_memory=True)
  test_loader = data.DataLoader(torch.Tensor(test_data),batch_size=B,shuffle=False,pin_memory=True)
  model = BiSeNet()

  # Loading model if was trained before
  if "bisenet.pt" in os.listdir("."):
    model.load_state_dict(torch.load("./bisenet.pt"))
  
  optimizer = optim.Adam(model.parameters(),lr = lr)

  eval_losses = []
  train_losses = []
  eval_losses.append(eval_model(model,test_loader))
  print("Mean Evaluation loss: {}".format(eval_losses[-1]))
  for i in range(epochs):
    train_loss = train(model,train_loader,optimizer)
    print("Mean Train loss: {}".format(np.mean(train_loss)))
    train_losses = train_losses + train_loss
    eval_losses.append(eval_model(model,test_loader))
    print("Mean Evaluation loss: {}".format(eval_losses[-1]))


  torch.save(model.state_dict(),"./bisenet.pt")



if __name__ == '__main__':
  main()