#!/bin/bash
#SBATCH --job-name=complb3d-batch
#SBATCH --partition=meile_p        
#SBATCH --mail-type=ALL 
#SBATCH --mail-user=sa01687@uga.edu
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --mem=64gb
#SBATCH --time=198:00:00
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err
 
cd $SLURM_SUBMIT_DIR



module purge
module load foss/2022a




srun ./complab


