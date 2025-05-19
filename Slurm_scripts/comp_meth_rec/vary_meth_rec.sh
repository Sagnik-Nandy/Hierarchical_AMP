#!/bin/bash -l
#SBATCH --job-name=amp_array
#SBATCH --output=/home/sagnik/Research/Hierarchical_AMP/Slurm_scripts/comp_meth_rec/slurm_out_%A_%a.out
#SBATCH --error=/home/sagnik/Research/Hierarchical_AMP/Slurm_scripts/comp_meth_rec/slurm_err_%A_%a.err
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --mem=8G
#SBATCH --array=0-24
#SBATCH --partition=caslake
#SBATCH --account=pi-sagnik

# Load Conda (no module needed)
source /home/sagnik/miniconda3/etc/profile.d/conda.sh
conda activate HierarchicalAMP

# Go to your project directory
cd /home/sagnik/Research/Hierarchical_AMP

echo "=== Starting job ID ${SLURM_ARRAY_TASK_ID} on node $(hostname) ==="

# Define grids for rho and n
rho_values=(0.8 0.85 0.9 0.95 1.0)
n_values=(2000 2500 3000 3500 4000)

idx=$SLURM_ARRAY_TASK_ID
num_n=${#n_values[@]}
rho_idx=$(( idx / num_n ))
n_idx=$(( idx % num_n ))

rho=${rho_values[$rho_idx]}
n=${n_values[$n_idx]}

echo "Running comparison for rho=$rho, n=$n"
python /home/sagnik/Research/Hierarchical_AMP/Python_scripts/compare_rec_perf.py $rho $n