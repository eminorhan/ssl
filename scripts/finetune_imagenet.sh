#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gres=gpu:titanrtx:2
#SBATCH --mem=250GB
#SBATCH --time=48:00:00
#SBATCH --array=0
#SBATCH --job-name=finetune_imagenet
#SBATCH --output=finetune_imagenet_%A_%a.out

module purge
module load cuda-11.4

python -u /misc/vlgscratch4/LakeGroup/emin/ssl/finetune_imagenet.py --model 'resnext101_32x8d' --freeze-trunk --n_out 6017 --resume 'TC_SAY_resnext101_32x8d_1.0_1_epoch_7.tar'

### 0.1
#python -u /misc/vlgscratch4/LakeGroup/emin/ssl/finetune_imgnet.py --model 'resnext101_32x8d' --freeze-trunk --n_out 1628 --resume 'TC_SAYAVAKEPICUT_resnext101_32x8d_0.1_1_epoch_11.tar'
#python -u /misc/vlgscratch4/LakeGroup/emin/ssl/finetune_imgnet.py --model 'resnext101_32x8d' --freeze-trunk --n_out 1628 --resume 'TC_SAYAVAKEPICUT_resnext101_32x8d_0.1_2_epoch_11.tar'
#python -u /misc/vlgscratch4/LakeGroup/emin/ssl/finetune_imgnet.py --model 'resnext101_32x8d' --freeze-trunk --n_out 1628 --resume 'TC_SAYAVAKEPICUT_resnext101_32x8d_0.1_3_epoch_11.tar'

### 0.01
#python -u /misc/vlgscratch4/LakeGroup/emin/ssl/finetune_imgnet.py --model 'resnext101_32x8d' --freeze-trunk --n_out 163 --resume 'TC_SAYAVAKEPICUT_resnext101_32x8d_0.01_1_epoch_11.tar'
#python -u /misc/vlgscratch4/LakeGroup/emin/ssl/finetune_imgnet.py --model 'resnext101_32x8d' --freeze-trunk --n_out 163 --resume 'TC_SAYAVAKEPICUT_resnext101_32x8d_0.01_2_epoch_11.tar'
#python -u /misc/vlgscratch4/LakeGroup/emin/ssl/finetune_imgnet.py --model 'resnext101_32x8d' --freeze-trunk --n_out 163 --resume 'TC_SAYAVAKEPICUT_resnext101_32x8d_0.01_3_epoch_11.tar'

### 0.001
#python -u /misc/vlgscratch4/LakeGroup/emin/ssl/finetune_imgnet.py --model 'resnext101_32x8d' --freeze-trunk --n_out 17 --resume 'TC_SAYAVAKEPICUT_resnext101_32x8d_0.001_1_epoch_11.tar'
#python -u /misc/vlgscratch4/LakeGroup/emin/ssl/finetune_imgnet.py --model 'resnext101_32x8d' --freeze-trunk --n_out 17 --resume 'TC_SAYAVAKEPICUT_resnext101_32x8d_0.001_2_epoch_11.tar'
#python -u /misc/vlgscratch4/LakeGroup/emin/ssl/finetune_imgnet.py --model 'resnext101_32x8d' --freeze-trunk --n_out 17 --resume 'TC_SAYAVAKEPICUT_resnext101_32x8d_0.001_3_epoch_11.tar'

echo "Done"
