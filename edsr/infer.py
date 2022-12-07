import torch
from torch import nn

from model import *

import numpy as np
import cv2

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error


# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
scale = 4
hr_size = 512
lr_size = hr_size//4

## Create model
model = torch.load("edsr_x4_1_0.56465.pth", map_location=torch.device(device))
model.eval()

## Read image
img = cv2.imread("../dataset/train/DIV2K_train_HR/0793.png")
img = img.astype(np.float32)
lr = cv2.resize(img, (lr_size,lr_size))
hr = cv2.resize(img, (hr_size,hr_size))

ip = lr/255.0
ip = ip.transpose(2,0,1)
ip = torch.from_numpy(ip).float().unsqueeze(0).to(device)

## Predict
output = model(ip)
output = output.squeeze(0)
output = output.detach().cpu().numpy().transpose(1,2,0).astype(np.float32)
output = output*255.0

## Bicubic interpolation
bi_output = cv2.resize(lr, (hr.shape[1],hr.shape[0]))

print(output.shape)
print(hr.shape)

## PSNR
psnr_model = cv2.PSNR(output, hr)
psnr_bi = cv2.PSNR(bi_output, hr)
print("PSNR Model: ", psnr_model)
print("PSNR Bicubic: ", psnr_bi)

## SSIM
ssim_model = ssim(output, hr, multichannel=True)
ssim_bi = ssim(bi_output, hr, multichannel=True)
print("SSIM Model: ", ssim_model)
print("SSIM Bicubic: ", ssim_bi)

## MSE
mse_model = mean_squared_error(output, hr)
mse_bi = mean_squared_error(bi_output, hr)
print("MSE Model: ", mse_model)
print("MSE Bicubic: ", mse_bi)

## Save
cv2.imwrite("bi_output.png", bi_output)
cv2.imwrite("output.png", output)
cv2.imwrite("input.png", lr)
cv2.imwrite("gt.png", hr)