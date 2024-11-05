#!/bin/bash
#SBATCH --output=CONFLUENCE_single_%j.log
#SBATCH --error=CONFLUENCE_single_%j.err
#SBATCH --time=20:00:00
#SBATCH --ntasks=8
#SBATCH --mem-per-cpu=5G


# Activate your Python environment if necessary
module restore confluence_modules


source /home/darri/confluence_env/bin/activate

# Your commands here

python ../CONFLUENCE/CONFLUENCE.py --config /home/darri/code/INDRA/0_config_files/config_active.yaml
