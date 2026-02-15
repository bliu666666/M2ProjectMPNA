#!/bin/bash
#SBATCH --job-name=HPL_sweep
#SBATCH --account=r250142
#SBATCH --partition=instant
#SBATCH --constraint=x64cpu
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --mem-per-cpu=4G
#SBATCH --time=00:25:00
#SBATCH --output=hpl_%j.out

module purge
module load aocl-4.2.0
module load openmpi/gnu/4.1.7

cd $SLURM_SUBMIT_DIR/bin/romeo_gnu_aocl

CSV=$SLURM_SUBMIT_DIR/hpl_nbscan_${SLURM_JOB_ID}.csv
echo "NB,N,P,Q,Time_s,GFLOPS" > $CSV

for NB in 64 128 192 256
do
    echo "============================="
    echo " Running NB = $NB"
    echo "============================="

    cp HPL_nb${NB}.dat HPL.dat

    srun -n 16 ./xhpl > run_nb${NB}.log

    RESULT=$(grep WR00 run_nb${NB}.log | head -n 1)

    N=$(echo $RESULT | awk '{print $2}')
    NBVAL=$(echo $RESULT | awk '{print $3}')
    P=$(echo $RESULT | awk '{print $4}')
    Q=$(echo $RESULT | awk '{print $5}')
    TIME=$(echo $RESULT | awk '{print $6}')
    GFLOPS=$(echo $RESULT | awk '{print $7}')

    echo "$NBVAL,$N,$P,$Q,$TIME,$GFLOPS" >> $CSV

    echo "NB=$NBVAL  Time=$TIME s  GFLOPS=$GFLOPS"
done

echo ""
echo "===================================="
echo "  NB SCAN FINISHED"
echo "  Results saved in:"
echo "  $CSV"
echo "===================================="

