#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --gres=gpu:v100:2
#SBATCH --mem=320GB
#SBATCH --time=48:00:00
#SBATCH --array=0
#SBATCH --job-name=dino_sayavakepicut
#SBATCH --output=dino_sayavakepicut_%A_%a.out

module purge
module load cuda-11.4

python -u -m torch.distributed.launch --nproc_per_node=4 /misc/vlgscratch4/LakeGroup/emin/ssl/train_dino.py --use_fp16 True --arch resnext50_32x4d --batch_size_per_gpu 128 --optimizer adamw --weight_decay 0.0001 --weight_decay_end 0.0001 --clip_grad 0.3 --global_crops_scale 0.15 1 --local_crops_scale 0.05 0.15 --cache_path '/misc/vlgscratch4/LakeGroup/emin/ssl/cache/SAYAVAKEPICUT_5fps_300s_0.1_1.pth' --seed 1 --fraction 0.1 --output_dir /misc/vlgscratch4/LakeGroup/emin/ssl/models --data_dirs '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/5fps_300s/A' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/5fps_300s/ava_test' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/5fps_300s/ava_trainval' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/5fps_300s/kcam' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/5fps_300s/P01' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/5fps_300s/P02' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/5fps_300s/P03' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/5fps_300s/P04' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/5fps_300s/P06' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/5fps_300s/P07' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/5fps_300s/P09' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/5fps_300s/P11' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/5fps_300s/P12' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/5fps_300s/P22' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/5fps_300s/P23' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/5fps_300s/P25' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/5fps_300s/P26' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/5fps_300s/P27' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/5fps_300s/P28' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/5fps_300s/P30' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/5fps_300s/P33' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/5fps_300s/P34' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/5fps_300s/P35' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/5fps_300s/P36' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/5fps_300s/P37' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/5fps_300s/S' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/5fps_300s/UT_ego' '/misc/vlgscratch5/LakeGroup/emin/sayavakepicut/5fps_300s/Y'

echo "Done"
