#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --gres=gpu:titanrtx:4
#SBATCH --mem=320GB
#SBATCH --time=48:00:00
#SBATCH --array=0
#SBATCH --job-name=tc_sayavakepicut
#SBATCH --output=tc_sayavakepicut_%A_%a.out

module purge
module load cuda-10.2

python -u /misc/vlgscratch4/LakeGroup/emin/ssl/train_tc.py --model 'resnext101_32x8d' --n_out 16279 --subject 'SAYAVAKEPICUT' --resume 'TC_SAYAVAKEPICUT_resnext101_32x8d_1.0_epoch_6.tar' --start-epoch 7 --lr 0.0005 --cache-path '/misc/vlgscratch4/LakeGroup/emin/ssl/cache/SAYAVAKEPICUT_5fps_288s_1.pth' --class-fraction 1.0 --print-freq 30000 --data-dirs '/misc/vlgscratch4/LakeGroup/emin/headcam/preprocessing/SAY_5fps_288s' '/misc/vlgscratch5/LakeGroup/shared_data/ava_v2/trainval_5fps' '/misc/vlgscratch5/LakeGroup/shared_data/ava_v2/test_5fps' '/misc/vlgscratch4/LakeGroup/emin/kcam/frames_5fps_288s' '/misc/vlgscratch4/LakeGroup/emin/epic-kitchens/frames_5fps_288s/P01' '/misc/vlgscratch4/LakeGroup/emin/epic-kitchens/frames_5fps_288s/P02' '/misc/vlgscratch4/LakeGroup/emin/epic-kitchens/frames_5fps_288s/P03' '/misc/vlgscratch4/LakeGroup/emin/epic-kitchens/frames_5fps_288s/P04' '/misc/vlgscratch4/LakeGroup/emin/epic-kitchens/frames_5fps_288s/P06' '/misc/vlgscratch4/LakeGroup/emin/epic-kitchens/frames_5fps_288s/P07' '/misc/vlgscratch4/LakeGroup/emin/epic-kitchens/frames_5fps_288s/P09' '/misc/vlgscratch4/LakeGroup/emin/epic-kitchens/frames_5fps_288s/P11' '/misc/vlgscratch4/LakeGroup/emin/epic-kitchens/frames_5fps_288s/P12' '/misc/vlgscratch4/LakeGroup/emin/epic-kitchens/frames_5fps_288s/P22' '/misc/vlgscratch4/LakeGroup/emin/epic-kitchens/frames_5fps_288s/P23' '/misc/vlgscratch4/LakeGroup/emin/epic-kitchens/frames_5fps_288s/P25' '/misc/vlgscratch4/LakeGroup/emin/epic-kitchens/frames_5fps_288s/P26' '/misc/vlgscratch4/LakeGroup/emin/epic-kitchens/frames_5fps_288s/P27' '/misc/vlgscratch4/LakeGroup/emin/epic-kitchens/frames_5fps_288s/P28' '/misc/vlgscratch4/LakeGroup/emin/epic-kitchens/frames_5fps_288s/P30' '/misc/vlgscratch4/LakeGroup/emin/epic-kitchens/frames_5fps_288s/P33' '/misc/vlgscratch4/LakeGroup/emin/epic-kitchens/frames_5fps_288s/P34' '/misc/vlgscratch4/LakeGroup/emin/epic-kitchens/frames_5fps_288s/P35' '/misc/vlgscratch4/LakeGroup/emin/epic-kitchens/frames_5fps_288s/P36' '/misc/vlgscratch4/LakeGroup/emin/epic-kitchens/frames_5fps_288s/P37' '/misc/vlgscratch4/LakeGroup/emin/UT_ego/frames_5fps_288s'

echo "Done"
