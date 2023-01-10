from model import *
import torch
import numpy as np
import imageio
import argparse
import os
import cv2
from PIL import Image
import glob
from tqdm import tqdm
import math
from skimage.metrics import structural_similarity as ssim

from torchvision import transforms

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='../dataset/test/',
                        help='image directory')
    parser.add_argument('--scale', type=str, default='4',
                        help='super resolution scale')
    return parser.parse_args()

def main():
    args = parse_args()

    # Prepare model
    model = edsr().cuda()
    model = torch.load("../../edsr_x4_154_1.37108.pth", map_location=torch.device("cuda:0"))
    model.eval()

    # process datasets
    datasets = glob.glob(args.data_dir + '*')

    idx = 0
    # Produce benchmark
    for dataset in datasets:
        dataset_name = dataset.split('/')[-1]
        lr_paths = glob.glob(dataset + f"/LR/X{args.scale}/imgs/*")
        hr_paths = glob.glob(dataset + f"/HR/*")
        eval_psnr = 0
        eval_ssim = 0

        for lr_path, hr_path in tqdm(zip(lr_paths, hr_paths),total=len(lr_paths)):

            # read LR and HR image using filename
            lr_path = lr_path.replace('\\', '/')
            hr_path = hr_path.replace('\\', '/')
            lr = imageio.imread(lr_path)
            hr = imageio.imread(hr_path)
            save_lr = lr.copy()

            # check shape
            if len(lr.shape) == 2 and len(hr.shape) == 2:
                lr = cv2.cvtColor(lr, cv2.COLOR_GRAY2RGB)
                hr = cv2.cvtColor(hr, cv2.COLOR_GRAY2RGB)

            lr = Image.fromarray(lr)

            # Preprocess images
            all_transforms = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean = [0.4517, 0.4986, 0.5077],
                                                std = [0.1661 , 0.141, 0.1520])
            ])

            lr = all_transforms(lr)            
            lr = lr.unsqueeze(0).cuda()

            # inference
            sr = model(lr)

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

            hr = hr.astype(np.float32)
            
            # crop border
            w, h = sr.shape[1], sr.shape[0]
            scale = int(args.scale)
            hr = hr[:int(h//scale*scale), :int(w//scale*scale), :]

            # metrics
            eval_psnr += cv2.PSNR(hr,sr)
            eval_ssim += ssim(hr, sr, multichannel=True, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=255)
            
            # save sr results
            if idx % 50 == 0:
                sr = cv2.cvtColor(sr, cv2.COLOR_RGB2BGR)
                hr = cv2.cvtColor(hr, cv2.COLOR_RGB2BGR)
                save_lr = cv2.cvtColor(save_lr, cv2.COLOR_RGB2BGR)
                cv2.imwrite(f'../../edsr_results/{idx}_sr.png', sr)
                cv2.imwrite(f'../../edsr_results/{idx}_hr.png', hr)
                cv2.imwrite(f'../../edsr_results/{idx}_lr.png', save_lr)

            idx += 1

        # print metrics
        print(f'{dataset_name}: PSNR: {eval_psnr / len(lr_paths)}, SSIM: {eval_ssim / len(lr_paths)}')

if __name__ == '__main__':
    with torch.no_grad():
        main()