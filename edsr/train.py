import torch 
import torch.optim as optim
import torch.nn as nn

from torchvision import transforms

import time

import matplotlib.pyplot as plt
import cv2
import numpy as np

## Training 
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
NAME = "edsr_x4"

t_invTrans = transforms.Compose([ 
                transforms.Normalize(mean = [ 0., 0., 0. ],
                                    std = [ 1/0.1661, 1/0.141, 1/0.1520]),
                transforms.Normalize(mean = [-0.4517, -0.4986, -0.5077],
                                    std = [ 1., 1., 1. ]),
                            ])

def save_training_viz(epoch,train_hist,val_hist,mode):
    plt.figure(figsize=(10,9))
    plt.plot(train_hist,'b', label="train")
    plt.plot(val_hist,'r', label="validation")
    if mode == 'acc':
        plt.ylabel('PSNR')
    if mode == 'loss':
        plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend()
    plt.savefig(f'./training/{NAME}_training_{epoch}_{mode}.png')
    
def log_and_print(text):
    logger = open(f'./training/{NAME}_training_log.txt',"a")
    logger.write(text+"\n")
    print(text)
    logger.close()
    
def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train_model(model, dataloaders, criterion, optimizer, lr_scheduler, num_epochs=25):
    since = time.time()
    val_acc_history = []
    train_acc_history = []
    val_loss_history = []
    train_loss_history = []
    best_acc = 0.0
    mini_batch_no = 0

    for epoch in range(num_epochs):
        log_and_print('=' * 100)
        ep_time = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            
            running_loss = 0.0
            running_psnr = 0

            # Iterate over data.
            for step, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad() # zero the parameter gradients
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs) # Forward prop
                    loss = criterion(outputs, labels) # Get model outputs and calculate loss
                    if phase == 'train':
                        mini_batch_no += 1
                        loss.backward() # Backward prop
                        optimizer.step() 

                # Track training statistics within the epoch
                running_loss += loss.item() * inputs.size(0)
                # Calculate PSNR
                batch_psnr = 0
                for i in range(len(outputs)):
                    img1 = t_invTrans(outputs[i])
                    img1 = img1.detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32)
                    img1 = img1*255.0
                    img2 = t_invTrans(labels[i])
                    img2 = img2.detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32)
                    img2 = img2*255.0
                    batch_psnr += cv2.PSNR(img1, img2)
                running_psnr += (batch_psnr/len(outputs))
                epoch_loss = running_loss / len(dataloaders[phase].dataset)
                epoch_acc = running_psnr / len(dataloaders[phase].dataset)

                # Print training statistics within the epoch
                if(step%20==0 and phase=='train'):
                    lr = get_lr(optimizer)
                    log_and_print(f'{phase}=> Epoch: {epoch}/{num_epochs-1} Step: {mini_batch_no} Acc: {round(epoch_acc,5)} Loss: {round(epoch_loss,5)} LR : {round(lr,8)}')
                if(step%20==0 and phase=='val'):
                    lr = get_lr(optimizer)
                    log_and_print(f'{phase}=> Epoch: {epoch}/{num_epochs-1} Step: {step} Acc: {round(epoch_acc,5)} Loss: {round(epoch_loss,5)} LR : {round(lr,8)}')
            
            # LR scheduler step
            if(phase == 'val'):
                lr_scheduler.step()

            # Print training statistics after the epoch
            log_and_print(f'\n\n***** {phase}=> Epoch: {epoch}/{num_epochs-1} Epoch Acc: {round(epoch_acc,5)} Epoch Loss: {round(epoch_loss,5)} *****\n\n')

            # Save the best model
            if phase == 'val' and epoch_acc > best_acc:
                log_and_print("SAVING THE MODEL")                
                best_acc = epoch_acc
                torch.save(model, f'./training/{NAME}_{epoch}_{round(best_acc,5)}.pth')
            
            # Track training statistics after the epoch
            if phase == 'val':
                val_acc_history.append(epoch_acc)
                val_loss_history.append(epoch_loss)
            if phase == 'train':
                train_acc_history.append(epoch_acc)
                train_loss_history.append(epoch_loss)
        
        # Print time status after both phases
        est_time = ((time.time() - ep_time) / 60) * (num_epochs - epoch)
        log_and_print(" Estimated time remaining : {:.2f}m".format(est_time))
        save_training_viz(epoch,train_acc_history,val_acc_history,'acc')
        save_training_viz(epoch,train_loss_history,val_loss_history,'loss')
        
    # Print total time after training
    time_elapsed = (time.time() - since)
    log_and_print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    log_and_print('Best test acc: {:4f}'.format(best_acc))
    
    return
