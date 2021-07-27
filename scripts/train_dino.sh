#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --gres=gpu:titanrtx:4
#SBATCH --mem=320GB
#SBATCH --time=48:00:00
#SBATCH --array=0
#SBATCH --job-name=dino_sayavakepicut
#SBATCH --output=dino_sayavakepicut_%A_%a.out

module purge
module load cuda-10.2

python -u -m torch.distributed.launch --nproc_per_node=4 /misc/vlgscratch4/LakeGroup/emin/ssl/train_dino.py --arch resnext50_32x4d --batch_size_per_gpu 64 --optimizer adamw --weight_decay 0.0 --weight_decay_end 0.0 --global_crops_scale 0.15 1 --local_crops_scale 0.05 0.15 --cache_path '/misc/vlgscratch4/LakeGroup/emin/ssl/cache/SAYAVAKEPICUT_5fps_288s_1_dino.pth' --output_dir /misc/vlgscratch4/LakeGroup/emin/ssl/models --data_dirs '/misc/vlgscratch4/LakeGroup/emin/headcam/preprocessing/SAY_5fps_288s' '/misc/vlgscratch5/LakeGroup/shared_data/ava_v2/trainval_5fps' '/misc/vlgscratch5/LakeGroup/shared_data/ava_v2/test_5fps' '/misc/vlgscratch4/LakeGroup/emin/kcam/frames_5fps_288s' '/misc/vlgscratch4/LakeGroup/emin/epic-kitchens/frames_5fps_288s/P01' '/misc/vlgscratch4/LakeGroup/emin/epic-kitchens/frames_5fps_288s/P02' '/misc/vlgscratch4/LakeGroup/emin/epic-kitchens/frames_5fps_288s/P03' '/misc/vlgscratch4/LakeGroup/emin/epic-kitchens/frames_5fps_288s/P04' '/misc/vlgscratch4/LakeGroup/emin/epic-kitchens/frames_5fps_288s/P06' '/misc/vlgscratch4/LakeGroup/emin/epic-kitchens/frames_5fps_288s/P07' '/misc/vlgscratch4/LakeGroup/emin/epic-kitchens/frames_5fps_288s/P09' '/misc/vlgscratch4/LakeGroup/emin/epic-kitchens/frames_5fps_288s/P11' '/misc/vlgscratch4/LakeGroup/emin/epic-kitchens/frames_5fps_288s/P12' '/misc/vlgscratch4/LakeGroup/emin/epic-kitchens/frames_5fps_288s/P22' '/misc/vlgscratch4/LakeGroup/emin/epic-kitchens/frames_5fps_288s/P23' '/misc/vlgscratch4/LakeGroup/emin/epic-kitchens/frames_5fps_288s/P25' '/misc/vlgscratch4/LakeGroup/emin/epic-kitchens/frames_5fps_288s/P26' '/misc/vlgscratch4/LakeGroup/emin/epic-kitchens/frames_5fps_288s/P27' '/misc/vlgscratch4/LakeGroup/emin/epic-kitchens/frames_5fps_288s/P28' '/misc/vlgscratch4/LakeGroup/emin/epic-kitchens/frames_5fps_288s/P30' '/misc/vlgscratch4/LakeGroup/emin/epic-kitchens/frames_5fps_288s/P33' '/misc/vlgscratch4/LakeGroup/emin/epic-kitchens/frames_5fps_288s/P34' '/misc/vlgscratch4/LakeGroup/emin/epic-kitchens/frames_5fps_288s/P35' '/misc/vlgscratch4/LakeGroup/emin/epic-kitchens/frames_5fps_288s/P36' '/misc/vlgscratch4/LakeGroup/emin/epic-kitchens/frames_5fps_288s/P37' '/misc/vlgscratch4/LakeGroup/emin/UT_ego/frames_5fps_288s'

echo "Done"
