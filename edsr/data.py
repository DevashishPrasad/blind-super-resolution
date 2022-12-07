import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torchvision.transforms.functional as transfunc

from matplotlib import pyplot as plt
import numpy as np
from PIL import Image

import glob
import random

import degrade

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, label):
        for t in self.transforms:
            img, label = t(img, label)
        return img, label

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, label): # p = (0.5 for flip) + (0.5 for no flip) 
        if random.random() < self.p:
            return transfunc.hflip(img), transfunc.hflip(label)
        return img, label

class RandomRotation(object):
    def __init__(self, p=0.33): # p = (0.33 for 90) + (0.33 for -90) + (0.33 for 0)
        self.p = p

    def __call__(self, img, label):
        if self.p > random.random():
            return transfunc.rotate(img, 90), transfunc.rotate(label, 90)
        elif 2*self.p > random.random():
            return transfunc.rotate(img, -90), transfunc.rotate(label, -90) 
        return img, label

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img, label):
        return (img - self.mean) / self.std, label

class Div2kDataset(Dataset):
    def __init__(self, hr_path, degradation_params, mode='train'):
        self.patch_size = 48
        self.hr_paths = glob.glob(hr_path+'/*')
        self.mode = mode
        self.factor = degradation_params['scale']

        # before degradation transform
        self.get_patch = transforms.RandomCrop(self.patch_size*self.factor)
        
        # select normalization method
        self.normalize = None
        if mode == 'train':
            self.normalize = transforms.Normalize(mean = [0.4517, 0.4986, 0.5077],
                                            std = [0.1661 , 0.141, 0.1520])
        elif mode == 'val':
            self.normalize = transforms.Normalize(mean = [0.4101, 0.4346, 0.4364],
                                        std = [0.2879 , 0.2741, 0.2620])

        # after degradation transform (same for LR and HR)
        self.same_transforms = Compose([
                RandomHorizontalFlip(p=0.5),
                RandomRotation(p=0.33),
        ])

        # Degradation function
        self.degradation = degrade.Degradation(**degradation_params)

    def __getitem__(self, index):
        img = Image.open(self.hr_paths[index])
        img = transfunc.to_tensor(img).to(device)

        # random crop image
        y = self.get_patch(img)
        y = y.unsqueeze(0).unsqueeze(0)
        x, kernels = self.degradation(y)
        
        y = y.squeeze(0).squeeze(0).to(device)
        x = x.squeeze(0).squeeze(0).to(device)

        # normalize image
        y = self.normalize(y)
        x = self.normalize(x)

        # apply same transforms
        x, y = self.same_transforms(x, y)

        return x, y
    
    def __len__(self):
        return len(self.hr_paths)

if __name__ == '__main__':
    # Test the above code
    
    import yaml

    with open('./config.yml', 'r') as f:
        config = yaml.safe_load(f)

    train_hr = config['data_paths']['train_dir']
    valid_hr = config['data_paths']['valid_dir']
    degradation_params = config['degradation_params']

    train_dataset = Div2kDataset(train_hr, degradation_params, 'train')
    val_dataset = Div2kDataset(valid_hr, degradation_params, 'val')

    trainloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    validloader = DataLoader(val_dataset, batch_size=4, shuffle=True)

    def show_img(img, invTrans, title):
        plt.figure(figsize=(6,5))
        # unnormalize
        img = invTrans(img)
        npimg = img.numpy()
        npimg = np.clip(npimg, 0., 1.)
        plt.title(title)
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()
    
    # Test trainloader
    t_invTrans = transforms.Compose([ 
                    transforms.Normalize(mean = [ 0., 0., 0. ],
                                        std = [ 1/0.1661, 1/0.141, 1/0.1520]),
                    transforms.Normalize(mean = [-0.4517, -0.4986, -0.5077],
                                        std = [ 1., 1., 1. ]),
                               ])

    # show images
    for idx,(x_batch,y_batch) in enumerate(trainloader):
        x_batch = x_batch.cpu()
        y_batch = y_batch.cpu()
        show_img(utils.make_grid(x_batch), t_invTrans, 'Train LR')
        show_img(utils.make_grid(y_batch), t_invTrans, 'Train HR')
        if(idx>=3):
            break
    
    # Test validloader
    v_invTrans = transforms.Compose([ 
                    transforms.Normalize(mean = [ 0., 0., 0. ],
                                        std = [1/0.2879 , 1/0.2741, 1/0.2620]),
                    transforms.Normalize(mean = [-0.4101, -0.4346, -0.4364],
                                        std = [ 1., 1., 1. ]),
                               ])

    # show images
    for idx,(x_batch,y_batch) in enumerate(validloader):
        x_batch = x_batch.cpu()
        y_batch = y_batch.cpu()
        show_img(utils.make_grid(x_batch), v_invTrans, 'Val LR')
        show_img(utils.make_grid(y_batch), v_invTrans, 'Val HR')
        if(idx>=3):
            break