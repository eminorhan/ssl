#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gres=gpu:titanrtx:2
#SBATCH --mem=320GB
#SBATCH --time=48:00:00
#SBATCH --array=0
#SBATCH --job-name=train_tc
#SBATCH --output=train_tc_%A_%a.out

module purge
module load cuda-11.4

python -u /misc/vlgscratch4/LakeGroup/emin/ssl/train_tc.py --resume '' --start-epoch 0 --lr 0.0005 --seed 1 --class-fraction 1.0 --print-freq 1000 --model 'resnext101_32x8d' --n_out 6017 --subject 'SAY' --cache-path '/misc/vlgscratch4/LakeGroup/emin/ssl/cache/SAY_5_300_vis.pth' --data-dirs '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/5fps_300s/S' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/5fps_300s/A' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/5fps_300s/Y'

#python -u /misc/vlgscratch4/LakeGroup/emin/ssl/train_tc.py --resume '' --start-epoch 0 --lr 0.0005 --seed 1 --class-fraction 1.0 --print-freq 1000 --model 'resnext101_32x8d' --n_out 2654 --subject 'S' --cache-path '/misc/vlgscratch4/LakeGroup/emin/ssl/cache/S_5_300_vis.pth' --data-dirs '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/5fps_300s/S'

#python -u /misc/vlgscratch4/LakeGroup/emin/ssl/train_tc.py --resume '' --start-epoch 0 --lr 0.0005 --seed 1 --class-fraction 1.0 --print-freq 1000 --model 'resnext101_32x8d' --n_out 1714 --subject 'A' --cache-path '/misc/vlgscratch4/LakeGroup/emin/ssl/cache/A_5_300_vis.pth' --data-dirs '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/5fps_300s/A'

#python -u /misc/vlgscratch4/LakeGroup/emin/ssl/train_tc.py --resume '' --start-epoch 0 --lr 0.0005 --seed 1 --class-fraction 1.0 --print-freq 1000 --model 'resnext101_32x8d' --n_out 1649 --subject 'Y' --cache-path '/misc/vlgscratch4/LakeGroup/emin/ssl/cache/Y_5_300_vis.pth' --data-dirs '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/5fps_300s/Y'

echo "Done"
