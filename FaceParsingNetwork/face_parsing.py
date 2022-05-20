### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import torch
from torchvision import transforms
from PIL import Image
from torchvision.utils import save_image
from models.networks import define_P

netP = define_P(11, 3, 64, "unet_128", "batch", use_dropout=True, gpu_ids=[])
netP.load_state_dict(torch.load('.\checkpoints\pretrained\latest_net_P.pth'))

img = Image.open('example_photo.png')
img = transforms.Grayscale()(img)
convert_tensor = transforms.ToTensor()
tensor = convert_tensor(img)
tensor = tensor.repeat(3,1,1)/3

new_img, new_img2 = netP(tensor.unsqueeze(0))
print(torch.max(torch.abs(new_img-new_img2)))
max_indices = new_img.argmax(1)
new_img = torch.zeros (new_img.shape).scatter (1, max_indices.unsqueeze (1), 1.0).squeeze(0)
img_eyes = new_img[2].unsqueeze(0)+new_img[3].unsqueeze(0)+new_img[4].unsqueeze(0)+new_img[5].unsqueeze(0)
img_nose = new_img[6].unsqueeze(0)
img_lips = new_img[7].unsqueeze(0)+new_img[8].unsqueeze(0)+new_img[9].unsqueeze(0)

max_indices = new_img2.argmax(1)
new_img2 = torch.zeros (new_img2.shape).scatter (1, max_indices.unsqueeze (1), 1.0).squeeze(0)
img_eyes2 = new_img2[2].unsqueeze(0)+new_img2[3].unsqueeze(0)+new_img2[4].unsqueeze(0)+new_img2[5].unsqueeze(0)
img_nose2 = new_img2[6].unsqueeze(0)
img_lips2 = new_img2[7].unsqueeze(0)+new_img2[8].unsqueeze(0)+new_img2[9].unsqueeze(0)

save_image(img_eyes.unsqueeze(0),'img_eyes.png')
save_image(img_nose.unsqueeze(0),'img_nose.png')
save_image(img_lips.unsqueeze(0),'img_lips.png')

save_image(img_eyes2.unsqueeze(0),'img_eyes2.png')
save_image(img_nose2.unsqueeze(0),'img_nose2.png')
save_image(img_lips2.unsqueeze(0),'img_lips2.png')
print(torch.sum(torch.abs(img_eyes-img_eyes2)))
