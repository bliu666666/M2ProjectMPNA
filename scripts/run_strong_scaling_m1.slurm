#!/bin/bash
#SBATCH --job-name=MiniHPL_Strong
#SBATCH --account=r250142
#SBATCH --partition=instant
#SBATCH --constraint=x64cpu
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00:15:00
#SBATCH --output=strong_%j.out

module purge
module load openmpi/gnu/4.1.7

cd $SLURM_SUBMIT_DIR

N=1024
NB=64
SEED=1

echo "ranks,N,total_s,pivot_s,swap_s,bcast_s,update_s,solve_s" > strong_${SLURM_JOB_ID}.csv

for P in 1 2 4 8 16; do
  echo "=== Running P=$P N=$N ==="
  srun --ntasks=$P --cpu-bind=cores ./minihpl_mpi_m1 $N $NB $SEED --no-check | tee run_P${P}.log
  grep "^CSV," run_P${P}.log | sed 's/^CSV,//' >> strong_${SLURM_JOB_ID}.csv
done

echo "CSV saved to strong_${SLURM_JOB_ID}.csv"
