#!/bin/bash
#SBATCH --job-name=amp_array
#SBATCH --output=/home/sagnik/Research/Hierarchical_AMP/Slurm_scripts/pred_nl_comp_singid/slurm_out_%A_%a.out
#SBATCH --error=/home/sagnik/Research/Hierarchical_AMP/Slurm_scripts/pred_nl_comp_singid/slurm_err_%A_%a.err
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --mem=16G
#SBATCH --array=0-4
#SBATCH --partition=caslake
#SBATCH --account=pi-sagnik


# Load Conda (no module needed)
source /home/sagnik/miniconda3/etc/profile.d/conda.sh
conda activate HierarchicalAMP

# Go to your project directory
cd /home/sagnik/Research/Hierarchical_AMP

echo "=== Starting job ID ${SLURM_ARRAY_TASK_ID} on node $(hostname) ==="

# Set test values of n
n_values=(2000 2500 3000 3500 4000)
n=${n_values[$SLURM_ARRAY_TASK_ID]}

echo "Running AMP experiment for n = $n"
python Python_scripts/pred_err_vary_n_nl_singind.py $n