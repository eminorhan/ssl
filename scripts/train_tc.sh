#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --gres=gpu:titanrtx:2
#SBATCH --mem=250GB
#SBATCH --time=48:00:00
#SBATCH --array=0
#SBATCH --job-name=train_tc
#SBATCH --output=train_tc_%A_%a.out

module purge
module load cuda-10.2

python -u /misc/vlgscratch4/LakeGroup/emin/ssl/train_tc.py --model 'resnext50_32x4d' --n_out 1649 --subject 'Y' --resume '' --start-epoch 0 --lr 0.0005 --cache-path '/misc/vlgscratch4/LakeGroup/emin/ssl/cache/Y_5_300.pth' --seed 1 --class-fraction 1.0 --print-freq 1000 --data-dirs '/misc/vlgscratch5/LakeGroup/emin/saycam_av/Y_5_300/vid'

echo "Done"
