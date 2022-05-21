from torchvision import transforms
from PIL import Image
import torch
import os
from tqdm import tqdm

directory = "/datasets/CelebAMask-HQ/CelebA-HQ-img/"

l = os.listdir(directory)
attr_list = ['l_brow', 'l_eye', 'l_lip', 'l_ear', 'mouth', 'neck', 'nose', 'skin', 'u_lip', 'r_eye', 'r_brow', 'cloth', 'r_ear', 'hair', 'ear_r', 'hat', 'eye_g', 'neck_l', 'background']

"""
lines = []
with open("trainlist.txt", "r") as f:
  for x in f:
    lines.append("/datasets/CelebAMask-HQ/CelebA-HQ-img/"+str(int(x.strip()[:-4]))+".jpg")
"""

img_path_list = [directory+x for x in l]

convert_tensor = transforms.ToTensor()
tensor_list = []

for img_path in tqdm(img_path_list):
  tensor_list.append(convert_tensor(Image.open(img_path).resize((512,512))).unsqueeze(0))
  print(tensor_list[0].dtype)
tensor = torch.cat(tensor_list, dim=0)
print(tensor.shape)

