import argparse

parser = argparse.ArgumentParser()

# Input Parameters
parser.add_argument('--cuda', type=int, default=0)

parser.add_argument('--epochs', type=int, default=10, help='maximum number of epochs to train the total model.')
parser.add_argument('--batch_size', type=int,default=16,help="Batch size to use per GPU")
parser.add_argument('--lr', type=float, default=2e-4, help='learning rate of encoder.')

parser.add_argument('--de_type', nargs='+', default=['denoise_15', 'denoise_25', 'denoise_50'],
                    help='which type of degradations is training and testing for.[denoise_15, denoise_25, denoise_50, derain, dehaze]')

parser.add_argument('--patch_size', type=int, default=32, help='patchsize of input.')
parser.add_argument('--num_workers', type=int, default=16, help='number of workers.')

# path
parser.add_argument('--data_file_dir', type=str, default='/data4/GalaxySynthesis/Galaxy_Dataset/240703_hst_dataset_fixed_size/f814w/64X64/minmax_ttv/train/gt/',  help='where clean images of denoising saves.')
parser.add_argument('--denoise_dir', type=str, default='/data4/GalaxySynthesis/Galaxy_Dataset/240703_hst_dataset_fixed_size/f814w/64X64/minmax_ttv/train/gt/',
                    help='where clean images of denoising saves.')
parser.add_argument('--derain_dir', type=str, default='data/Train/Derain/',
                    help='where training images of deraining saves.')
parser.add_argument('--dehaze_dir', type=str, default='data/Train/Dehaze/',
                    help='where training images of dehazing saves.')
parser.add_argument('--psf_dir', type=str, default='/data4/GalaxySynthesis/Galaxy_Dataset/240703_hst_dataset_fixed_size/f814w/psf_ttv/train/psf/',
                    help='where training images of psf images saves.')
parser.add_argument('--e1e2_dir', type=str, default='/data4/GalaxySynthesis/Galaxy_Dataset/240703_hst_dataset_fixed_size/f814w/240813_HST_PSF/e1e2/',
                    help='where training images of e1e2 images saves.')
parser.add_argument('--output_path', type=str, default="/home2/s20245354/PromptIR/output/64X64/train(240825_directlen6)", help='output save path')
parser.add_argument('--ckpt_path', type=str, default="/home2/s20245354/PromptIR/ckpt/Denoise/64X64/train(240825_directlen6)", help='checkpoint save path')
parser.add_argument("--wblogger",type=str,default="promptir(direct64_len6)",help = "Determine to log to wandb or not and the project name")
parser.add_argument("--ckpt_dir",type=str,default="/home2/s20245354/PromptIR/ckpt/Denoise/64X64/train(240825_directlen6)",help = "Name of the Directory where the checkpoint is to be saved")
parser.add_argument("--num_gpus",type=int,default= 1,help = "Number of GPUs to use for training")

options = parser.parse_args()

