import torch
from torch import nn
from torchsummary import summary

from data import *
from model import *
from train import *

import yaml

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

with open('./config.yml', 'r') as f:
    config = yaml.safe_load(f)

# Config parameters
train_hr = config['data_paths']['train_dir']
valid_hr = config['data_paths']['valid_dir']
degradation_params = config['degradation_params']

mini_b_size = config['training_params']['mini_batch_size']
lr = config['training_params']['learning_rate']
epochs = config['training_params']['epochs']
patience = config['training_params']['patience']

train_dataset = Div2kDataset(train_hr, degradation_params, 'train')
val_dataset = Div2kDataset(valid_hr, degradation_params, 'val')

trainloader = DataLoader(train_dataset, batch_size=mini_b_size, shuffle=True, num_workers=2)
validloader = DataLoader(val_dataset, batch_size=mini_b_size, shuffle=True, num_workers=2)

dataloaders = {'train': trainloader, 'val': validloader}

## Create main model
model = BlindSR().to(device)
params_to_update = model.parameters()
# summary(model, (2, 3, 64, 64), batch_size=mini_b_size)

## Initialize optimizers, schedulers and criterion
optimizer_ft = optim.Adam(params_to_update, lr=lr, betas=(0.9, 0.999))
lr_decayer = torch.optim.lr_scheduler.StepLR(optimizer_ft, 40, gamma=0.5)
l1_loss = nn.L1Loss()
contrast_loss = nn.CrossEntropyLoss()
loss = {'l1': l1_loss, 'contrast': contrast_loss}

## Execute Training
train_model(model, dataloaders, loss, optimizer_ft, lr_decayer, num_epochs=epochs)
