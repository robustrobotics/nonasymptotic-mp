#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task 4
#SBATCH --time=24:00:00

i=1
current_time=$(date +"%Y%m%d-%H%M%S")  # Getting the current time in YYYYMMDD-HHMMSS format
for seed in $(seq 1 200)
do
    for algorithm in "prm"
    do
        if [ $((i)) -eq  $((SLURM_ARRAY_TASK_ID + 0)) ]; then
            save_path="/home/gridsan/acurtis/runs/$algorithm-$seed-$n-$current_time"
            export OMP_NUM_THREADS=4
            python ./kuka_carry/run.py --seed=$seed --save-dir="$save_path" --delta="$1" --collision-buffer="$2" --theta-margin="$3" --adaptive-n
        fi
        i=$((i+1))
    done
done