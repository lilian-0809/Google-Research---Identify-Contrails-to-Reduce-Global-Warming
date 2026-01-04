# Dataset
import yaml
with open("config_unet_tf_efficientnetv2_xl_512_wd1", "r") as file_obj:
    config = yaml.safe_load(file_obj)
import torch
import numpy as np
import torchvision.transforms as T

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

# Lightning module

import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from torch.optim import AdamW
import torch.nn as nn
from torchmetrics.functional import dice

seg_models = {
    "Unet": smp.Unet,
    "Unet++": smp.UnetPlusPlus,
    "MAnet": smp.MAnet,
    "Linknet": smp.Linknet,
    "FPN": smp.FPN,
    "PSPNet": smp.PSPNet,
    "PAN": smp.PAN,
    "DeepLabV3": smp.DeepLabV3,
    "DeepLabV3+": smp.DeepLabV3Plus,
}


class LightningModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = seg_models[config["seg_model"]](
            encoder_name=config["encoder_name"],
            encoder_weights='noisy-student',
            in_channels=3,
            classes=1,
            activation=None,
        )
        self.loss_module = smp.losses.FocalLoss(mode='binary', alpha=0.77, gamma=2.1)
        self.val_step_outputs = []
        self.val_step_labels = []

    def forward(self, batch):
        imgs = batch
        preds = self.model(imgs)
        return preds

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), **self.config["optimizer_params"])

        if self.config["scheduler"]["name"] == "CosineAnnealingLR":
            scheduler = CosineAnnealingLR(
                optimizer,
                **self.config["scheduler"]["params"]["CosineAnnealingLR"],
            )
            lr_scheduler_dict = {"scheduler": scheduler, "interval": "step"}
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler_dict}
        elif self.config["scheduler"]["name"] == "ReduceLROnPlateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                **self.config["scheduler"]["params"]["ReduceLROnPlateau"],
            )
            lr_scheduler = {"scheduler": scheduler, "monitor": "val_loss"}
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs)
        if self.config["image_size"] != 256:
            preds = torch.nn.functional.interpolate(preds, size=256, mode='bilinear')
        loss = self.loss_module(preds, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=False, batch_size=16)

        for param_group in self.trainer.optimizers[0].param_groups:
            lr = param_group["lr"]
        self.log("lr", lr, on_step=True, on_epoch=False, prog_bar=False)

        return loss

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs)
        if self.config["image_size"] != 256:
            preds = torch.nn.functional.interpolate(preds, size=256, mode='bilinear')
        loss = self.loss_module(preds, labels)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.val_step_outputs.append(preds)
        self.val_step_labels.append(labels)

    def on_validation_epoch_end(self):
        all_preds = torch.cat(self.val_step_outputs)
        all_labels = torch.cat(self.val_step_labels)
        all_preds = torch.sigmoid(all_preds)
        self.val_step_outputs.clear()
        self.val_step_labels.clear()
        val_dice = dice(all_preds, all_labels.long())
        self.log("val_dice", val_dice, on_step=False, on_epoch=True, prog_bar=False)
        if self.trainer.global_rank == 0:
            print(f"\nEpoch: {self.current_epoch}", flush=True)

# Actual training

import warnings

warnings.filterwarnings("ignore")

import os
import torch

import pandas as pd
import pytorch_lightning as pl
from pprint import pprint
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
from torch.utils.data import DataLoader



contrails = os.path.join(config["data_path"], "contrails/")
train_path = os.path.join(config["data_path"], "train_df.csv")
valid_path = os.path.join(config["data_path"], "valid_df.csv")

train_df = pd.read_csv(train_path).head(20512)
valid_df = pd.read_csv(valid_path)#.head(1000)


train_df["path"] = contrails + train_df["record_id"].astype(str) + ".npy"
valid_df["path"] = contrails + valid_df["record_id"].astype(str) + ".npy"

dataset_train = ContrailsDataset(train_df, config["model"]["image_size"], train=True)
dataset_validation = ContrailsDataset(valid_df, config["model"]["image_size"], train=False)

data_loader_train = DataLoader(
    dataset_train,
    batch_size=config["train_bs"],
    shuffle=True,
    num_workers=config["workers"],
)
data_loader_validation = DataLoader(
    dataset_validation,
    batch_size=config["valid_bs"],
    shuffle=False,
    num_workers=config["workers"],
)

checkpoint_callback = ModelCheckpoint(
    save_weights_only=True,
    monitor="val_dice",
    dirpath=config["output_dir"],
    mode="max",
    filename=f"{config['model']['seg_model']}_{config['model']['encoder_name']}_{config['model']['image_size']}",
    save_top_k=1,
    verbose=1,
)

progress_bar_callback = TQDMProgressBar(
    refresh_rate=config["progress_bar_refresh_rate"]
)

early_stop_callback = EarlyStopping(**config["early_stop"])

trainer = pl.Trainer(
    callbacks=[checkpoint_callback, early_stop_callback, progress_bar_callback],
    **config["trainer"],
)

config["model"]["scheduler"]["params"]["CosineAnnealingLR"]["T_max"] *= len(data_loader_train)/config["trainer"]["devices"]
model = LightningModule(config["model"])

trainer.fit(model, data_loader_train, data_loader_validation)
