#!/bin/bash
#SBATCH -J chenyi
#SBATCH -p q_ai4
#SBATCH -o slurm/job.%j.out
#SBATCH -e slurm/job.%j.err
#SBATCH -n 1
#SBATCH -N 1
#SBATCH -c 1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -t 24:00:00
module load anaconda3
source ~/.bashrc
conda activate /home/liuyunzhe_lab/lichenyi/DATA/.conda/envs/replay
export LD_LIBRARY_PATH=/home/liuyunzhe_lab/lichenyi/DATA/.conda/envs/replay/lib:$LD_LIBRARY_PATH
export DGLBACKEND=tensorflow
python main_batch.py