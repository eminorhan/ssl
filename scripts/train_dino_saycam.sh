#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --gres=gpu:titanrtx:4
#SBATCH --mem=320GB
#SBATCH --time=48:00:00
#SBATCH --array=0
#SBATCH --job-name=dino_say
#SBATCH --output=dino_say_%A_%a.out

module purge
module load cuda-10.2

python -u -m torch.distributed.launch --nproc_per_node=4 /misc/vlgscratch4/LakeGroup/emin/ssl/train_dino.py --arch resnext50_32x4d --batch_size_per_gpu 52 --optimizer adamw --weight_decay 0.0 --weight_decay_end 0.0 --global_crops_scale 0.15 1 --local_crops_scale 0.05 0.15 --cache_path '/misc/vlgscratch4/LakeGroup/emin/ssl/cache/S_5fps_288s_dino.pth' --output_dir /misc/vlgscratch4/LakeGroup/emin/ssl/models --data_dirs '/misc/vlgscratch4/LakeGroup/emin/headcam/preprocessing/S_5fps_288s'

echo "Done"
