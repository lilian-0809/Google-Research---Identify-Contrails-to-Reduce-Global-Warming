#!pip install transformers
#!pip install segmentation-models-pytorch==0.3.3
#!pip install safetensors==0.3.1

from pathlib import Path
import os
import random
import math
from collections import defaultdict
import cv2
import skimage

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch import nn
from torchvision import transforms
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from tqdm.notebook import tqdm
from transformers import get_cosine_schedule_with_warmup

import segmentation_models_pytorch as smp

import safetensors

class Config:
    train = True
    
    num_epochs = 24
    num_classes = 1
    batch_size = 8
    seed = 42
    
    encoder = 'tu-seresnextaa101d_32x8d'
    pretrained = False#True
    weights = None#'imagenet '
    classes = ['contrail']
    activation = None
    in_chans = 3
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    image_size = 768
    warmup = 0
    lr = 1e-4
    pretrain = './pretrain/seresnextaa101d_32x8d.sw_in12k_ft_in1k_288/model.safetensors'
    
class Paths:
    #data_root = '/kaggle/input/google-research-identify-contrails-reduce-global-warming'
    contrails = '../data/contrails/'
    train_path = '../data/train_df.csv'
    valid_path = '../data/valid_df.csv'
    save_dir = './save'
    #pretrain = 'https://huggingface.co/timm/tf_efficientnetv2_xl.in21k_ft_in1k/resolve/main/model.safetensors'
    
def set_seed(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    
os.makedirs(Paths.save_dir, exist_ok=True)
set_seed(9)

# Import dataframes
train_df = pd.read_csv(Paths.train_path)
valid_df = pd.read_csv(Paths.valid_path)

train_df['path'] = Paths.contrails + train_df['record_id'].astype(str) + '.npy'
valid_df['path'] = Paths.contrails + valid_df['record_id'].astype(str) + '.npy'

train_df.shape, valid_df.shape

class ContrailsDataset(torch.utils.data.Dataset):
    def __init__(self, df, image_size=256, train=True):

        self.df = df
        self.trn = train
        self.normalize_image = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        self.image_size = image_size
        if image_size != 256:
            self.resize_image = T.transforms.Resize(image_size)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        con_path = row.path
        con = np.load(str(con_path))

        img = con[..., :-1]
        label = con[..., -1]
        
        if self.trn:
            p = np.random.randint(6)
            if p==1:
                label = np.rot90(label, k=1).copy()
                img = np.rot90(img, k=1).copy()
                
            #if p==2:
            #    label = np.rot90(label, k=2).copy()
            #    img = np.rot90(img, k=2).copy()
                
            #if p==3:
            #    label = np.rot90(label, k=3).copy()
            #    img = np.rot90(img, k=3).copy()
                
                
            if p==2:
                label = label[::-1].copy()
                img = img[::-1].copy()
                
            #if p==5:
            #    label = label[:, ::-1].copy()
            #    img = img[:, ::-1].copy()
                
            if p==3:
                label = np.rot90(label[:, ::-1]).copy()
                img = np.rot90(img[:, ::-1]).copy()
                
            #if p==7:
            #    label = np.rot90(label[:, ::-1], k=3).copy()
            #    img = np.rot90(img[:, ::-1], k=3).copy()
        
        label = torch.tensor(label)

        img = torch.tensor(np.reshape(img, (256, 256, 3))).to(torch.float32).permute(2, 0, 1)

        if self.image_size != 256:
            img = self.resize_image(img)

        img = self.normalize_image(img)

        return img.float(), label.float()

    def __len__(self):
        return len(self.df)

train_ds = ContrailsDataset(
        train_df,
        Config.image_size,
        train=True
    )

valid_ds = ContrailsDataset(
        valid_df,
        Config.image_size,
        train=False
    )

train_dl = DataLoader(train_ds, batch_size=Config.batch_size , shuffle=True, num_workers = 8, drop_last=True)    
valid_dl = DataLoader(valid_ds, batch_size=Config.batch_size, num_workers = 8, drop_last=True)

img, label = next(iter(train_dl))
img.shape, label.shape

img, label = next(iter(valid_dl))
img.shape, label.shape

def dice_coef(y_true, y_pred, thr=0.5, epsilon=0.001):
    y_true = y_true.flatten()
    y_pred = (y_pred>thr).astype(np.float32).flatten()
    inter = (y_true*y_pred).sum()
    den = y_true.sum() + y_pred.sum()
    dice = ((2*inter+epsilon)/(den+epsilon))
    return dice

class UNet(nn.Module):
    def __init__(self, cfg):
        super(UNet, self).__init__()
        
        self.cfg = cfg
        self.training = True
        
        self.model = smp.UnetPlusPlus(
            encoder_name=cfg.encoder, 
            encoder_weights=cfg.weights, 
            decoder_use_batchnorm=True,
            classes=len(cfg.classes), 
            activation=cfg.activation,
        )
        self.model.encoder.model.load_state_dict(safetensors.torch.load_file(cfg.pretrain, device="cpu"), strict=False)
        
        self.loss_fn = smp.losses.FocalLoss(mode='binary', alpha=0.77, gamma=2.1)
    
    def forward(self, imgs, targets):
        
        x = imgs
        y = targets

        logits = self.model(x)
        logits = torch.nn.functional.interpolate(logits, size=256, mode='bilinear')
        loss = self.loss_fn(logits, y)
        
        return {"loss": loss, "logits": logits.sigmoid(), "logits_raw": logits, "target": y}

def train_step(model, dataloader, optimizer, device):
    
    model.train()
    scaler = torch.cuda.amp.GradScaler()
    train_losses = []
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Train ', miniters=50, mininterval=60)
    
    for step, (X, y) in pbar:
        torch.set_grad_enabled(True)
        with torch.cuda.amp.autocast():
            X, y = X.to(device), y.to(device)
            #
        
            output_dict = model(X, y)
        loss = output_dict["loss"]
        train_losses.append(loss.item())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        #loss.backward()
        #optimizer.step()
        scaler.update()
        optimizer.zero_grad()
        
        if scheduler is not None:
            scheduler.step()
    
    train_loss = np.sum(train_losses)
    
    return train_loss

def test_step(model, dataloader, device):
    
    model.eval()
    torch.set_grad_enabled(False)
    
    val_data = defaultdict(list)
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Valid', miniters=50, mininterval=60)
    for step, (X, y) in pbar: 
        X, y = X.to(device), y.to(device)

        output = model(X, y)
        for key, val in output.items():
            val_data[key] += [output[key]]

    for key, val in output.items():
        value = val_data[key]
        if len(value[0].shape) == 0:
            val_data[key] = torch.stack(value)
        else:
            val_data[key] = torch.cat(value, dim=0).cpu().detach().numpy()
    
    val_losses = val_data["loss"].cpu().numpy()
    val_loss = np.sum(val_losses)
    
    val_dice = dice_coef(val_data['target'], val_data['logits'])
    
    return val_loss, val_dice

from tqdm.auto import tqdm


def train(model, train_dataloader, test_dataloader, optimizer, epochs, device):
    results = {'train_loss': [],
              'val_loss': [],
              'val_dice': []}
    best = 0
    es = 3
    k = 0
    for epoch in range(epochs):
        
        set_seed(Config.seed + epoch)
        print("EPOCH:", epoch)
        
        train_loss = train_step(model,
                              train_dataloader,
                              optimizer,
                              device)
        val_loss, val_dice = test_step(model,
                            test_dataloader,
                            device)
        
        train_loss = train_loss / len(train_ds)
        val_loss = val_loss / len(valid_ds)
        
        print(f'Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f}')
        print(f"Learning rate: {optimizer.param_groups[0]['lr']}")
        
        results['train_loss'].append(train_loss)
        results['val_loss'].append(val_loss)
        results['val_dice'].append(val_dice)

        
        if val_dice> best:
            best = val_dice
            PATH = f"{Paths.save_dir}/unetpp_fl_{Config.encoder}_{Config.image_size}_ep{epoch}_dice{val_dice:.4f}_aug2.pth"
            torch.save(model.state_dict(), PATH)
            k = 0
        else:
            k +=1
        if k >=es:
            break
        
    return results
def get_optimizer(lr, params):
    
    model_optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, params), 
            lr=lr,
            weight_decay=1e-6)
    
    return model_optimizer

def get_scheduler(cfg, optimizer, total_steps):
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps= cfg.warmup * (total_steps // cfg.batch_size),
        num_training_steps= cfg.num_epochs * (total_steps // cfg.batch_size)
    )
    return scheduler

NUM_EPOCHS = Config.num_epochs
model = UNet(Config).to(Config.device)

total_steps = len(train_ds)
optimizer = get_optimizer(lr=Config.lr, params=model.parameters())
scheduler = get_scheduler(Config, optimizer, total_steps)

from timeit import default_timer as timer
start_time = timer()

model_results = train(model, train_dl, valid_dl, optimizer, NUM_EPOCHS, Config.device)

end_time = timer()

# run.finish()
print(f'Total Training Time: {end_time-start_time:.3f} seconds')


