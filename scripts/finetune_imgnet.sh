#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gres=gpu:titanrtx:2
#SBATCH --mem=250GB
#SBATCH --time=48:00:00
#SBATCH --array=0
#SBATCH --job-name=finetune_imgnet
#SBATCH --output=finetune_imgnet_%A_%a.out

module purge
module load cuda-10.2

#python -u /misc/vlgscratch4/LakeGroup/emin/baby-vision/imagenet_finetuning.py --frac-retained 0.01 --n_out 6269 --resume 'TC-SAY-resnext.tar'
#python -u /misc/vlgscratch4/LakeGroup/emin/baby-vision/imagenet_finetuning.py --freeze-trunk --n_out 1000 --resume 'ft_IN_TC-SAY-resnext.tar'
#python -u /misc/vlgscratch4/LakeGroup/emin/baby-vision/imagenet_finetuning.py --freeze-trunk --n_out 1000 --resume 'ft_IN_TC_ALL_5fps_288s_resnext50_32x4d_epoch_34.tar'
#python -u /misc/vlgscratch4/LakeGroup/emin/baby-vision/imagenet_finetuning.py --frac-retained 0.05 --n_out 1000 --resume ''

python -u /misc/vlgscratch4/LakeGroup/emin/ssl/finetune_imgnet.py --model 'resnext101_32x8d' --freeze-trunk --n_out 16279 --resume 'TC_SAYAVAKEPICUT_resnext101_32x8d_1.0_epoch_14.tar'

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
