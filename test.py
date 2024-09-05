import argparse
import subprocess
from tqdm import tqdm
import numpy as np

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import torch.nn as nn 

from utils.dataset_utils import DenoiseTestDataset, DerainDehazeDataset
from utils.schedulers import LinearWarmupCosineAnnealingLR
from utils.val_utils import AverageMeter, compute_psnr_ssim
from utils.image_io import save_image_tensor
from net.model import PromptIR

import lightning.pytorch as pl
import torch.nn.functional as F

class PromptIRModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.net = PromptIR(decoder=True)
        self.loss_fn  = nn.L1Loss()
    
    def forward(self,x, psf_patch, e1e2_patch):
        return self.net(x, psf_patch, e1e2_patch)
    
    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        ([clean_name, de_id], degrad_patch, clean_patch, psf_patch) = batch

        if psf_patch is None:
            raise ValueError("psf_patch is None")
        
        restored = self.net(degrad_patch, psf_patch)


        loss = self.loss_fn(restored,clean_patch)
        # Logging to TensorBoard (if installed) by default
        # Calculate PSNR and SSIM
        psnr_value, ssim_value, _ = compute_psnr_ssim(restored, clean_patch)

        self.train_psnr_values.append(psnr_value)
        self.train_ssim_values.append(ssim_value)

        # Logging to TensorBoard (if installed) by default
        self.log("test_loss", loss)
        self.log("test_psnr", psnr_value)
        self.log("test_ssim", ssim_value)

        return loss
    
    def lr_scheduler_step(self,scheduler,metric):
        scheduler.step(self.current_epoch)
        lr = scheduler.get_lr()
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=2e-4)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer,warmup_epochs=15,max_epochs=150)

        return [optimizer],[scheduler]



def test_Denoise(net, dataset, sigma=15):
    output_path = testopt.output_path + 'test_denoise/' + testopt.resolution + '/'
    subprocess.check_output(['mkdir', '-p', output_path])
    

    # dataset.set_sigma(sigma)
    testloader = DataLoader(dataset, batch_size=1, pin_memory=True, shuffle=False, num_workers=0)

    psnr = AverageMeter()
    ssim = AverageMeter()

    with torch.no_grad():
        for ([clean_name], degrad_patch, clean_patch, psf_patch, e1e2_patch) in tqdm(testloader):
            degrad_patch, clean_patch, psf_patch, e1e2_patch = degrad_patch.cuda(), clean_patch.cuda(), psf_patch.cuda(), e1e2_patch.cuda()

            restored = net(degrad_patch, psf_patch, e1e2_patch)
            temp_psnr, temp_ssim, N = compute_psnr_ssim(restored, clean_patch)

            psnr.update(temp_psnr, N)
            ssim.update(temp_ssim, N)
            save_image_tensor( output_path + clean_name[0], restored)

        print("psnr: %.2f, ssim: %.4f" % (psnr.avg, ssim.avg))



def test_Derain_Dehaze(net, dataset, task="derain"):
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
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--mode', type=int, default=0,
                        help='0 for denoise, 1 for derain, 2 for dehaze, 3 for all-in-one')

    parser.add_argument('--denoise_path', type=str, default="/data4/GalaxySynthesis/Galaxy_Dataset/240703_hst_dataset_fixed_size/f814w/32X32/minmax_ttv/test/gt/", help='save path of test noisy images')
    parser.add_argument('--psf_dir', type=str, default="/data4/GalaxySynthesis/Galaxy_Dataset/240703_hst_dataset_fixed_size/f814w/psf_ttv/test/psf/", help='save path of test psf images')
    parser.add_argument('--derain_path', type=str, default="test/derain/", help='save path of test raining images')
    parser.add_argument('--dehaze_path', type=str, default="test/dehaze/", help='save path of test hazy images')
    parser.add_argument('--output_path', type=str, default="output/test(dircet32)/", help='output save path')
    parser.add_argument('--resolution', type=str, default="32X32", help='resolution')
    parser.add_argument('--ckpt_path', type=str, default="/home2/s20245354/PromptIR/ckpt/Denoise/32X32/train(240825_directlen6)/epoch=9-step=340830.ckpt", help='checkpoint save path')

    parser.add_argument('--e1e2_dir', type=str, default='/data4/GalaxySynthesis/Galaxy_Dataset/240703_hst_dataset_fixed_size/f814w/240813_HST_PSF/e1e2/',
                    help='where training images of e1e2 images saves.')
    testopt = parser.parse_args()
    
    

    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(testopt.cuda)


    ckpt_path = testopt.ckpt_path


    
    # denoise_splits = ["bsd68/"]
    # derain_splits = ["Rain100L/"]

    denoise_tests = []
    derain_tests = []

    # base_path = testopt.denoise_path
    # testopt.denoise_path = os.path.join(base_path,i)
    denoise_testset = DenoiseTestDataset(testopt)
    denoise_tests.append(denoise_testset)


    print("CKPT name : {}".format(ckpt_path))

    net  = PromptIRModel.load_from_checkpoint(ckpt_path).cuda()
    net.eval()

    
    # if testopt.mode == 0:
    #     for testset,name in zip(denoise_tests,denoise_splits) :
    #         print('Start {} testing Sigma=15...'.format(name))
    #         test_Denoise(net, testset, sigma=15)

    #         print('Start {} testing Sigma=25...'.format(name))
    #         test_Denoise(net, testset, sigma=25)

    #         print('Start {} testing Sigma=50...'.format(name))
    #         test_Denoise(net, testset, sigma=50)
    test_Denoise(net, denoise_testset)
    # elif testopt.mode == 1:
    #     print('Start testing rain streak removal...')
    #     derain_base_path = testopt.derain_path
    #     for name in derain_splits:
    #         print('Start testing {} rain streak removal...'.format(name))
    #         testopt.derain_path = os.path.join(derain_base_path,name)
    #         derain_set = DerainDehazeDataset(opt,addnoise=False,sigma=15)
    #         test_Derain_Dehaze(net, derain_set, task="derain")
    # elif testopt.mode == 2:
    #     print('Start testing SOTS...')
    #     derain_base_path = testopt.derain_path
    #     name = derain_splits[0]
    #     testopt.derain_path = os.path.join(derain_base_path,name)
    #     derain_set = DerainDehazeDataset(testopt,addnoise=False,sigma=15)
    #     test_Derain_Dehaze(net, derain_set, task="SOTS_outdoor")
    # elif testopt.mode == 3:
    #     for testset,name in zip(denoise_tests,denoise_splits) :
    #         print('Start {} testing Sigma=15...'.format(name))
    #         test_Denoise(net, testset, sigma=15)

    #         print('Start {} testing Sigma=25...'.format(name))
    #         test_Denoise(net, testset, sigma=25)

    #         print('Start {} testing Sigma=50...'.format(name))
    #         test_Denoise(net, testset, sigma=50)



        # derain_base_path = testopt.derain_path
        # print(derain_splits)
        # for name in derain_splits:

        #     print('Start testing {} rain streak removal...'.format(name))
        #     testopt.derain_path = os.path.join(derain_base_path,name)
        #     derain_set = DerainDehazeDataset(testopt,addnoise=False,sigma=15)
        #     test_Derain_Dehaze(net, derain_set, task="derain")

        # print('Start testing SOTS...')
        # test_Derain_Dehaze(net, derain_set, task="dehaze")