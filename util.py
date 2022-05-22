import torch

def getMasksFromParsing(parsing, isRGB = true):
  max_indices = parsing.argmax(1)
  parsing = torch.zeros (parsing.shape).scatter (1, max_indices.unsqueeze(1), 1.0)

  mask_eyes = parsing[:,2,:,:,]+parsing[:,3,:,:,]+parsing[:,4,:,:,]+parsing[:,5,:,:,]
  mask_nose = parsing[:,6,:,:,]
  mask_lips = parsing[:,7,:,:,]+parsing[:,8,:,:,]+parsing[:,9,:,:,]

  if isRGB:
    mask_eyes = mask_eyes.unsqueeze(1).repeat(1,3,1,1)
    mask_nose = mask_nose.unsqueeze(1).repeat(1,3,1,1)
    mask_lips = mask_lips.unsqueeze(1).repeat(1,3,1,1)
  else:
    mask_eyes = mask_eyes.unsqueeze(1)
    mask_nose = mask_nose.unsqueeze(1)
    mask_lips = mask_lips.unsqueeze(1)

  masks = torch.cat((mask_eyes.unsqueeze(0), mask_nose.unsqueeze(0), mask_lips.unsqueeze(0)), dim=0)

return masks

