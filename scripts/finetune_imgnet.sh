#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --gres=gpu:titanrtx:4
#SBATCH --mem=320GB
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

#python -u /misc/vlgscratch4/LakeGroup/emin/ssl/finetune_imgnet.py --freeze-trunk --resume 'DINO_SAYAVAKEPICUT_resnext50_32x4d_1.0_epoch_1.pth'
python -u /misc/vlgscratch4/LakeGroup/emin/ssl/finetune_imgnet.py --model 'resnext101_32x8d' --freeze-trunk --n_out 16279 --resume 'TC_SAYAVAKEPICUT_resnext101_32x8d_1.0_epoch_0.tar'
#python -u /misc/vlgscratch4/LakeGroup/emin/ssl/finetune_imgnet.py --freeze-trunk --n_out 1368 --resume 'TC_SAYAVAKEPICUT_resnext101_32x8d_0.1_epoch_9.tar'
#python -u /misc/vlgscratch4/LakeGroup/emin/ssl/finetune_imgnet.py --freeze-trunk --n_out 137 --resume 'TC_SAYAVAKEPICUT_resnext101_32x8d_0.01_epoch_9_run_3.tar'
#python -u /misc/vlgscratch4/LakeGroup/emin/ssl/finetune_imgnet.py --freeze-trunk --n_out 14 --resume 'TC_SAYAVAKEPICUT_resnext101_32x8d_0.001_epoch_9_run_3.tar'

echo "Done"
