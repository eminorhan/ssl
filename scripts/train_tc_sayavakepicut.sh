#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --gres=gpu:titanrtx:4
#SBATCH --mem=320GB
#SBATCH --time=48:00:00
#SBATCH --array=0
#SBATCH --job-name=train_tc_sayavakepicut
#SBATCH --output=train_tc_sayavakepicut_%A_%a.out

module purge
module load cuda-11.4

python -u /misc/vlgscratch4/LakeGroup/emin/ssl/train_tc.py --model 'resnext101_32x8d' --n_out 15928 --subject 'SAYAVAKEPICUT' --resume '' --start-epoch 0 --lr 0.0005 --cache-path '/misc/vlgscratch4/LakeGroup/emin/ssl/cache/SAYAVAKEPICUT_5fps_300s_1.pth' --seed 1 --class-fraction 1.0 --print-freq 1000 --data-dirs '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/5fps_300s/A' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/5fps_300s/ava_test' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/5fps_300s/ava_trainval' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/5fps_300s/kcam' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/5fps_300s/P01' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/5fps_300s/P02' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/5fps_300s/P03' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/5fps_300s/P04' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/5fps_300s/P06' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/5fps_300s/P07' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/5fps_300s/P09' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/5fps_300s/P11' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/5fps_300s/P12' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/5fps_300s/P22' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/5fps_300s/P23' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/5fps_300s/P25' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/5fps_300s/P26' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/5fps_300s/P27' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/5fps_300s/P28' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/5fps_300s/P30' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/5fps_300s/P33' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/5fps_300s/P34' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/5fps_300s/P35' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/5fps_300s/P36' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/5fps_300s/P37' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/5fps_300s/S' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/5fps_300s/UT_ego' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/5fps_300s/Y'

echo "Done"
