# Code Installation Memo. 

> 현재, 이 repo의 코드를 사용하기 위해 초기 설정을 정리해두었습니다.
- Conda environment (Conda를 사용하실 경우)
```shell 
conda create -n promptir python=3.8.11
conda activate promptir 
```
혹은, 아래의 코드로 가상환경에 접속할 수 있습니다.
```shell 
source activate promptir
```

- 만약 conda가 없을 경우, 아래의 코드로 가상환경을 만들 수 있습니다. 
    - Conda 위로 설치하시는게 편합니다!
```shell
python -m venv venv python=3.8.11
source activate venv 
```


- 가상환경이 setting되고, 가상환경에 들어가셨을 경우 라이브러리를 설치하면 됩니다. 
```shell 
pip install -r requirements.txt
```

- 여기까지 오셨을 경우, 기본적인 셋팅은 완료되었습니다.
    - 만약 오류가 나올 경우, cuda version이 다른 경우가 있을 수 있습니다. 이러한 경우 아래의 홈페이지에서 pytorch 맞는 버전을 설치하시면 돌아갈겁니다.
    - https://pytorch.org/get-started/previous-versions/ 

- 오류를 해결하지 못하실 경우 메일로 물어봐주시면 도와드리겠습니다.


<hr>

## Options 설정 (학습을 진행 할 때와 inference 하는 경우 option 설정이 다릅니다.)
- 학습을 진행하실 경우 options.py 파일에서 수정해주시면 됩니다.
    - 해당 파일 내에서 아래 부분을 수정해주시면 됩니다.
    - 현재 학습 한 checkpoint 의 경우 ./ckpt/ 아래 저장됩니다. 아래 option 들 중 ckpt_path 를 수정하시면 그 경로로 저장됩니다.
    - 현재 ./ckpt 아래 32x32, 64x64 psf 이미지를 prompt 형태가 아닌 직접 넣고 학습 시킨 checkpoint 2개가 있습니다.
```shell
parser.add_argument('--data_file_dir', type=str, default='/data4/GalaxySynthesis/Galaxy_Dataset/240703_hst_dataset_fixed_size/f814w/64X64/minmax_ttv/train/gt/',  help='where clean images of denoising saves.') # ground truth 경로
parser.add_argument('--denoise_dir', type=str, default='/data4/GalaxySynthesis/Galaxy_Dataset/240703_hst_dataset_fixed_size/f814w/64X64/minmax_ttv/train/gt/',
                    help='where clean images of denoising saves.') # ground truth 경로
parser.add_argument('--psf_dir', type=str, default='/data4/GalaxySynthesis/Galaxy_Dataset/240703_hst_dataset_fixed_size/f814w/psf_ttv/train/psf/',
                    help='where training images of psf images saves.') # psf 경로
parser.add_argument('--e1e2_dir', type=str, default='/data4/GalaxySynthesis/Galaxy_Dataset/240703_hst_dataset_fixed_size/f814w/240813_HST_PSF/e1e2/',
                    help='where training images of e1e2 images saves.') # e1e2 경로
parser.add_argument('--output_path', type=str, default="/home2/s20245354/PromptIR/output/64X64/train(240825_directlen6)", help='output save path') # training 과정 이미지 저장 경로이나 사용 안함
parser.add_argument('--ckpt_path', type=str, default="/home2/s20245354/PromptIR/ckpt/Denoise/64X64/train(240825_directlen6)", help='checkpoint save path') # checkpoint 저장 경로
parser.add_argument("--wblogger",type=str,default="promptir(direct64_len6)",help = "Determine to log to wandb or not and the project name") # wandb project name 
parser.add_argument("--ckpt_dir",type=str, default="/home2/s20245354/PromptIR/ckpt/Denoise/64X64/train(240825_directlen6)", help = "Name of the Directory where the checkpoint is to be saved") # checkpoint 저장 경로, ckpt path 와 동일하게 지정
parser.add_argument("--num_gpus",type=int,default= 1,help = "Number of GPUs to use for training")
```
- inference 의 경우 test.py 의 if __name__ == '__main__': 아래의 경로들을 수정해주시면 됩니다.

- 물론 cmd 창에서 설정해서 코드를 실행하셔도 무방하십니다.
```shell
python train.py --datafile_dir /data4/..... --denoise_dir /data4/.....
```


## Galaxy Part 
- 위에서 옵션으로 갤럭시 데이터의 경로를 지정해 주시고 train 또는 test 를 진행 하시면 됩니다.  
- utils/data_utils.py 경로에서 dataset을 load하는 코드를 확인해보실 수 있습니다.
- wandb, tensorboard를 통해 성능을 확인하고 있습니다. train 과정에서 현재는 psnr 과 ssim 만 확인 가능하고 학습 후 저장된 checkpoint 로 test.py 에서 inference 를 진행 할 수 있습니다.
- test.py 에서 나온 결과값은 output/test/ 폴더안에 자동으로 저장됩니다. test.py 에서 output_path 를 변경할 경우 변경 된 곳에 저장됩니다.
<hr>

- train 코드
```shell 
python train.py
```

- test 코드
```shell 
python test.py
```


<hr>

- https://github.com/va1shn9v/PromptIR 의 readme를 확인해보시면 실행에 오류가 있을 경우 도움이 되실 수 있습니다. 