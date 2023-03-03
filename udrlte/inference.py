from model import BlindSR

import torch
from torchvision import transforms

import numpy as np
import imageio
import cv2
from PIL import Image

import sys

def main():
    img_path = sys.argv[1]
    op_folder = sys.argv[2]

    # Prepare model
    model = BlindSR().cuda()
    model = torch.load("udrlte_x4_176_1.64958.pth", map_location=torch.device("cuda:0"))
    model.eval()

    lr = imageio.imread(img_path)

    # check shape
    if len(lr.shape) == 2:
        lr = cv2.cvtColor(lr, cv2.COLOR_GRAY2RGB)

    lr = Image.fromarray(lr).convert('RGB')

    # Preprocess images
    all_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.4517, 0.4986, 0.5077],
                                        std = [0.1661 , 0.141, 0.1520])
    ])

    # Get a random patch to predict degradation
    get_patch = transforms.RandomCrop(48)

    lr = all_transforms(lr)            
    lr = lr.unsqueeze(0).cuda()

    # inference
    # for i in range(10):
    #     patch += get_patch(lr)
    # patch /= 10

    patch = get_patch(lr)
    fea = model.E(patch, patch)

    # degradation-aware SR
    sr = model.G(lr, fea)            

    invTrans = transforms.Compose([ 
            transforms.Normalize(mean = [ 0., 0., 0. ],
                                std = [ 1/0.1661, 1/0.141, 1/0.1520]),
            transforms.Normalize(mean = [-0.4517, -0.4986, -0.5077],
                                std = [ 1., 1., 1. ]),
                        ])

    sr = invTrans(sr)
    sr = sr.cpu().numpy()
    sr = np.clip(sr, 0., 1.)
    sr = sr.squeeze(0).transpose(1, 2, 0)
    sr = (255 * sr).astype(np.float32)

    # save sr results
    sr = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f'{op_folder}/output.png', sr)

if __name__ == '__main__':
    with torch.no_grad():
        main()