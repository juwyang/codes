#!/bin/bash

set -x;
set -e;

sbatch -A sm86 -C gpu <<EOT
#!/bin/bash
#SBATCH --job-name=lstm_training
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=lstm_training_%j.out
#SBATCH --error=lstm_training_%j.err

# Load necessary modules (adjust these based on your system)
module load daint-gpu PyTorch

# Activate your virtual environment if needed
# source /path/to/your/venv/bin/activate

# Set up the distributed environment
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=12345
export WORLD_SIZE=$(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE))

# Run the training script using Accelerate
srun accelerate launch \
    --num_processes $(($SLURM_NNODES * $SLURM_NTASKS_PER_NODE)) \
    --num_machines $SLURM_NNODES \
    --machine_rank $SLURM_NODEID \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    lstm_training.py