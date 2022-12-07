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
    parser.add_argument('--data_dir', type=str, default='D:/Research/Viasat/Implementation/Experiment 4/Dataset/testset/',
                        help='image directory')
    parser.add_argument('--scale', type=str, default='4',
                        help='super resolution scale')
    parser.add_argument('--resume', type=int, default=600,
                        help='resume from specific checkpoint')
    parser.add_argument('--blur_type', type=str, default='aniso_gaussian',
                        help='blur types (iso_gaussian | aniso_gaussian)')
    return parser.parse_args()


def crop_border(img_hr, scale):
    b, n, c, h, w = img_hr.size()

    img_hr = img_hr[:, :, :, :int(h//scale*scale), :int(w//scale*scale)]

    return img_hr

def main():
    args = parse_args()

    # path to save sr images
    if args.blur_type == 'iso_gaussian':
        dir = './experiment/blindsr_x' + str(int(args.scale[0])) + '_bicubic_iso'
    elif args.blur_type == 'aniso_gaussian':
        dir = './experiment/blindsr_x' + str(int(args.scale[0])) + '_bicubic_aniso'
    save_dir = dir + '/benchmark/'
    # if not os.path.exists(save_dir):
    #     os.mkdir(save_dir)

    # Prepare model
    DASR = edsr().cuda()
    # DASR.load_state_dict(torch.load("udrl_x4_212_1.83558.pth"), strict=False)
    DASR = torch.load("edsr_x4_238_1.32778.pth", map_location=torch.device("cuda:0"))
    DASR.eval()

    # process datasets
    datasets = glob.glob(args.data_dir + '/*')

    # Produce benchmark
    for dataset in datasets:
        dataset = dataset.replace('\\', '/')
        dataset_name = dataset.split('/')[-1]
        lr_paths = glob.glob(dataset + f"/LR/X{args.scale}/imgs/*")
        hr_paths = glob.glob(dataset + f"/HR/*")
        eval_psnr = 0
        eval_ssim = 0
        # os.mkdir(save_dir + '/' + dataset_name)
        for lr_path, hr_path in tqdm(zip(lr_paths, hr_paths)):

            # read LR and HR image using filename
            lr_path = lr_path.replace('\\', '/')
            hr_path = hr_path.replace('\\', '/')
            lr = imageio.imread(lr_path)
            hr = imageio.imread(hr_path)
            # lr = Image.open(lr_path)

            # check shape
            if len(lr.shape) == 2 and len(hr.shape) == 2:
                lr = cv2.cvtColor(lr, cv2.COLOR_GRAY2RGB)
                hr = cv2.cvtColor(hr, cv2.COLOR_GRAY2RGB)

            # lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)
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
            sr = DASR(lr)

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


            # sr = sr.squeeze(0)
            # sr = sr.cpu().numpy().transpose(1,2,0).astype(np.float32)
            # sr = (255 * sr).astype(np.uint8)

            hr = hr.astype(np.float32)
            
            # print(sr)

            # cv2.imwrite("sr.png", sr)
            # cv2.imwrite("hr.png", hr)


            # crop border
            w, h = sr.shape[1], sr.shape[0]
            scale = int(args.scale)
            hr = hr[:int(h//scale*scale), :int(w//scale*scale), :]

            # metrics
            eval_psnr += cv2.PSNR(hr,sr)
            eval_ssim += ssim(hr, sr, multichannel=True, gaussian_weights=True, sigma=1.5, use_sample_covariance=False, data_range=255)
            
            # # save sr results
            # img_name = lr_path.split('.png')[0].split('/')[-1]
            # cv2.imwrite(save_dir + '/' + dataset_name + '/' + img_name + '_sr.png', sr)

        # print metrics
        print(f'{dataset_name}: PSNR: {eval_psnr / len(lr_paths)}, SSIM: {eval_ssim / len(lr_paths)}')

if __name__ == '__main__':
    with torch.no_grad():
        main()