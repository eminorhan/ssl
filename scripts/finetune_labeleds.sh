#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:titanrtx:1
#SBATCH --mem=200GB
#SBATCH --time=1:00:00
#SBATCH --array=0
#SBATCH --job-name=finetune_labeleds
#SBATCH --output=finetune_labeleds_%A_%a.out

module purge
module load cuda-11.4

#python -u /misc/vlgscratch4/LakeGroup/emin/ssl/finetune_labeleds.py --model-name 'TC-SAY-resnext' --num-outs 6269 --num-classes 26 --print-freq 10

#python -u /misc/vlgscratch4/LakeGroup/emin/ssl/finetune_labeleds.py --model-name 'TC-S-resnext' --num-outs 2765 --num-classes 26 --print-freq 10

#python -u /misc/vlgscratch4/LakeGroup/emin/ssl/finetune_labeleds.py --model-name 'TC-A-resnext' --num-outs 1786 --num-classes 26 --print-freq 10

#python -u /misc/vlgscratch4/LakeGroup/emin/ssl/finetune_labeleds.py --model-name 'TC-Y-resnext' --num-outs 1718 --num-classes 26 --print-freq 10

python -u /misc/vlgscratch4/LakeGroup/emin/ssl/finetune_labeleds.py --model-name 'DINO-SAY-resnext' --num-classes 26 --print-freq 10

#python -u /misc/vlgscratch4/LakeGroup/emin/ssl/finetune_labeleds.py --model-name 'imagenet' --num-classes 26 --print-freq 10

#python -u /misc/vlgscratch4/LakeGroup/emin/ssl/finetune_labeleds.py --model-name 'random' --num-classes 26 --print-freq 10

echo "Done"
