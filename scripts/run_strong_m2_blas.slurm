#!/bin/bash
#SBATCH --job-name=MiniHPL_M2_BLAS_Strong
#SBATCH --account=r250142
#SBATCH --partition=instant
#SBATCH --constraint=x64cpu
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00:15:00
#SBATCH --output=m2_blas_strong_%j.out

module purge
module load aocl-4.2.0
module load openmpi/gnu/4.1.7

cd $SLURM_SUBMIT_DIR

N=1024
NB=64
SEED=1

echo "ranks,N,total_s,diagLU_s,bcastDiagRow_s,bcastDiagCol_s,trsmL_s,trsmU_s,gemm_s" > m2_blas_strong_${SLURM_JOB_ID}.csv

for P in 1 2 4 8 16; do
  echo "=== Running P=$P N=$N ==="
  srun --ntasks=$P --cpu-bind=cores ./minihpl_mpi_m2_blas $N $NB $SEED | tee m2_blas_run_P${P}.log
  grep "^CSV," m2_blas_run_P${P}.log | sed 's/^CSV,//' >> m2_blas_strong_${SLURM_JOB_ID}.csv
done
echo "CSV saved to m2_blas_strong_${SLURM_JOB_ID}.csv"

