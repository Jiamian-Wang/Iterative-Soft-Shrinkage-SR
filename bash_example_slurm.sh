#!/bin/bash -l
# NOTE the -l flag!
#

#SBATCH --job-name=benchmark-SwinIR-L1norm-0.9
#SBATCH --error=/error/err_%j.txt
#SBATCH --output=/out/out_%j.txt
#SBATCH --ntasks 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --time=4-00:00:00
#SBATCH --partition tier3
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=48g

source ~/conda/etc/profile.d/conda.sh
conda activate sr_pruning

srun python SR_pruning/setup.py develop
GLOG_vmodule=MemcachedClient=-1 \
srun python -u ./basicsr/train_L1norm.py  \
--launcher="slurm" --opt  ./options/train/SwinIR/train_SwinIRLight_X2_DF2K_L1norm_pr0.9_lr0.0002.yml \
--prune_method L1 --prune_criterion l1-norm --compare_mode local   \
--wg weight  --stage_pr [0-1000:0.900] --stage_pr_lslen 1000 --stage_pr_lsval 0.9 \
--skip_layers *mean*  *tail* --scale 2  --pick_pruned min_9:10
