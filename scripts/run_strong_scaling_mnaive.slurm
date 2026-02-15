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

module load openmpi/gnu/4.1.7

N=1024
SEED=1

echo "ranks,N,total_s,pivot_s,swap_s,bcast_s,update_s,solve_s" > strong_${SLURM_JOB_ID}.csv

for P in 1 2 4 8 16; do
  echo "=== Running P=$P N=$N ==="
  mpirun -np $P ./minihpl_mpi $N $SEED --no-check | tee run_P${P}.log
  grep "^CSV," run_P${P}.log | sed 's/^CSV,//' >> strong_${SLURM_JOB_ID}.csv
done

echo "CSV saved to strong_${SLURM_JOB_ID}.csv"
