import subprocess
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from utils.dataset_utils import PromptTrainDataset
from net.model import PromptIR
from utils.schedulers import LinearWarmupCosineAnnealingLR
import numpy as np
import wandb
from options import options as opt
import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger,TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

from utils.val_utils import compute_psnr_ssim


class PromptIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptIR(decoder=True)
        self.loss_fn  = nn.L1Loss()
        self.train_psnr_values = []
        self.train_ssim_values = []
    
    def forward(self,x, psf_patch,e1e2_patch):
        return self.net(x, psf_patch,e1e2_patch)
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        ([clean_name, de_id], degrad_patch, clean_patch, psf_patch, e1e2_patch) = batch

        # Print the batch and psf_patch for debugging
        # print(f"Batch: {batch}")
        # print(f"PSF Patch Size in Training Step: {psf_patch.size()}")

        # Check if psf_patch is None
        if psf_patch is None:
            raise ValueError("psf_patch is None")

        restored = self.net(degrad_patch, psf_patch, e1e2_patch)

        # # 크기 출력
        # print(f"restored size: {restored.size()}, clean_patch size: {clean_patch.size()}")
        # exit()

        loss = self.loss_fn(restored,clean_patch)

        # Calculate PSNR and SSIM
        psnr_value, ssim_value, _ = compute_psnr_ssim(restored, clean_patch)

        self.train_psnr_values.append(torch.tensor(psnr_value))
        self.train_ssim_values.append(torch.tensor(ssim_value))

        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        self.log("train_psnr", psnr_value)
        self.log("train_ssim", ssim_value)

        return loss
    
    def on_train_epoch_end(self):
        avg_psnr = torch.stack(self.train_psnr_values).mean()
        avg_ssim = torch.stack(self.train_ssim_values).mean()

        self.log("avg_train_psnr", avg_psnr, prog_bar=True)
        self.log("avg_train_ssim", avg_ssim, prog_bar=True)

        print(f"Epoch {self.current_epoch} - Average PSNR: {avg_psnr:.4f}, Average SSIM: {avg_ssim:.4f}")

        # Clear the lists for the next epoch
        self.train_psnr_values.clear()
        self.train_ssim_values.clear()
    
    def lr_scheduler_step(self,scheduler,metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,warmup_epochs=15,max_epochs=150)

        return [optimizer],[scheduler]






def main():
    print("Options")
    print(opt)
    if opt.wblogger is not None:
        logger  = WandbLogger(project=opt.wblogger,name="PromptIR-Train")
    else:
        logger = TensorBoardLogger(save_dir = "logs/")

    trainset = PromptTrainDataset(opt)
    checkpoint_callback = ModelCheckpoint(dirpath = opt.ckpt_dir,every_n_epochs = 1,save_top_k=-1)
    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True,
                             drop_last=True, num_workers=opt.num_workers)
    
    model = PromptIRModel()
    
    trainer = pl.Trainer( max_epochs=opt.epochs,accelerator="gpu",devices=opt.num_gpus,strategy="ddp_find_unused_parameters_true",logger=logger,callbacks=[checkpoint_callback])
    trainer.fit(model=model, train_dataloaders=trainloader)


if __name__ == '__main__':
    main()



