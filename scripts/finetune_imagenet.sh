#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:v100:2
#SBATCH --mem=320GB
#SBATCH --time=48:00:00
#SBATCH --array=0
#SBATCH --job-name=finetune_imagenet
#SBATCH --output=finetune_imagenet_%A_%a.out

module purge
module load cuda-11.4

#python -u /misc/vlgscratch4/LakeGroup/emin/ssl/finetune_imagenet.py --model 'resnext50_32x4d' --freeze-trunk --model-name 'DINO-SAY-resnext' --seed 1 --frac-retained 0.1

#python -u /misc/vlgscratch4/LakeGroup/emin/ssl/finetune_imagenet.py --model 'resnext50_32x4d' --freeze-trunk --model-name 'TC-SAY-resnext' --num-outs 6269 --seed 1 --frac-retained 0.1

python -u /misc/vlgscratch4/LakeGroup/emin/ssl/finetune_imagenet.py --model 'resnext101_32x8d' --model-name 'TC-SAYAVAKEPICUT' --num-outs 15928 --seed 1 --frac-retained 0.02

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
