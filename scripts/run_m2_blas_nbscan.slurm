#!/bin/bash
#SBATCH --job-name=MiniHPL_NBscan
#SBATCH --account=r250142
#SBATCH --partition=instant
#SBATCH --constraint=x64cpu
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem-per-cpu=4G
#SBATCH --time=01:00:00
#SBATCH --output=nbscan_%j.out

module purge
module load aocl-4.2.0
module load openmpi/gnu/4.1.7

cd $SLURM_SUBMIT_DIR

N=20736
SEED=1

echo "NB,Grid,ranks,N,total_s,GFLOPS,diagLU_s,bcastRow_s,bcastCol_s,trsmL_s,trsmU_s,gemm_s" > nbscan_${SLURM_JOB_ID}.csv

for NB in 64 128 192 256; do
  echo "=== Running NB=$NB ==="
  srun --ntasks=16 --cpu-bind=cores ./minihpl_mpi_m2_blas $N $NB $SEED | tee nb_${NB}.log
  GF=$(grep -oE "GF/s=[0-9.]+" nb_${NB}.log | tail -1 | cut -d= -f2)
  GRID=$(grep -oE "Grid=[0-9]+x[0-9]+" nb_${NB}.log | head -1 | cut -d= -f2)
  grep "^CSV," nb_${NB}.log | sed 's/^CSV,//' | tail -1 | \
    awk -v NB=$NB -v GF=$GF -v GRID=$GRID -F',' \
    '{print NB","GRID","$1","$2","$3","GF","$4","$5","$6","$7","$8","$9}' \
    >> nbscan_${SLURM_JOB_ID}.csv
done

echo "CSV saved to nbscan_${SLURM_JOB_ID}.csv"


