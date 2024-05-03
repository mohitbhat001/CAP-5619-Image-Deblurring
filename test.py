import os
import numpy as np
import argparse
import subprocess
from tqdm import tqdm

import torch
import torch.nn as nn 
from torch.utils.data import DataLoader
import lightning.pytorch as pl

from utils.dataset_utils import DeBlurDataset
from utils.val_utils import AverageMeter, compute_psnr_ssim
from utils.image_io import save_image_tensor
from net.model import DeBlurModel
from utils.loss_utils import Stripformer_Loss
import torch.optim as optim

class DeBlurModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = DeBlurModel(decoder=True)
        self.loss_fn  = Stripformer_Loss()
    
    def forward(self,x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        ([clean_name, _], degrad_patch, clean_patch) = batch
        restored = self.net(degrad_patch)

        loss = self.loss_fn(restored,clean_patch)
        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss)
        return loss
    
    def lr_scheduler_step(self,scheduler,metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=0.01, steps_per_epoch=1752, epochs=20)

        return [optimizer],[scheduler]


def test_deblur(net, dataset):
    output_path = testopt.output_path + task + '/'
    subprocess.check_output(['mkdir', '-p', output_path])

    dataset.set_dataset(task)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([degraded_name], degrad_patch, clean_patch) in tqdm(testloader):
            degrad_patch, clean_patch = degrad_patch.cuda(), clean_patch.cuda()

            restored = net(degrad_patch)

            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)
            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)

            save_image_tensor(restored, output_path + degraded_name[0] + '.png')
        print("PSNR: %.2f, SSIM: %.4f" % (psnr.avg, ssim.avg))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Input Parameters
    parser.add_argument('--ckpt_name', type=str, help='checkpoint save path')
    testopt = parser.parse_args()
    
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(testopt.cuda)

    ckpt_path = "ckpt/" + testopt.ckpt_name


    deblur_splits = ["gopro/"]

    base_path = testopt.denoise_path
 

    print("CKPT name : {}".format(ckpt_path))

    net  = DeBlurModel().load_from_checkpoint(ckpt_path).cuda()
    net.eval()

  
    print('Start testing GOPRO...')
    deblur_base_path = testopt.gopro_path
    name = deblur_splits[0]
    testopt.gopro_path = os.path.join(deblur_base_path,name)
    derain_set = DeBlurDataset(testopt,addnoise=False,sigma=15, task='deblur')
    test_deblur(net, derain_set, task="deblur")

    