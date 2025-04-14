#!/bin/bash
#SBATCH --job-name=amp_array
#SBATCH --output=Slurm_scripts/vary_n/slurm_out_%A_%a.out
#SBATCH --error=Slurm_scripts/vary_n/slurm_err_%A_%a.err
#SBATCH --time=05:00:00
#SBATCH --ntasks=1
#SBATCH --mem=8G
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
python Python_scripts/cluster_ccoef_n.py $n