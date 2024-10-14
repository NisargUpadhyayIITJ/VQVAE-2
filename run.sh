#!/bin/bash
# Job name:
#SBATCH --job-name=test
# Partition:
#SBATCH --partition=small
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.txt
## Processors per task:
#SBATCH --cpus-per-task=2
#
#SBATCH --gres=gpu:1


module load conda
eval "$(conda shell.bash hook)"

conda create --name Nisarg python=3.10 -y
conda activate Nisarg
module load python/3.10.pytorch
pip install -r requirements.txt

python3 main-pixelsnail.py /csehome/b23cs1075/vqvae-2/latent-data/ffhq256_ffhq256-state-dict-0045_2024-10-13_21-42-35_latent /csehome/b23cs1075/vqvae-2/runs/ffhq256-2024-10-13_16-54-23/checkpoints/ffhq256-state-dict-0045.pt 1 --task ffhq256

conda deactivate
