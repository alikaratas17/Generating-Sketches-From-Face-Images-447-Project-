### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import torch
from torchvision import transforms
from PIL import Image
from torchvision.utils import save_image
from FaceParsingNetwork. models.networks import define_P

def getParsingNetwork():
    import os
    print(os.listdir("FaceParsingNetwork/checkpoints/pretrained"))
    netP = define_P(11, 3, 64, "unet_128", "batch", use_dropout=True, gpu_ids=[])
    netP.load_state_dict(torch.load('FaceParsingNetwork/checkpoints/pretrained/latest_net_P.pth'))
    return netP
