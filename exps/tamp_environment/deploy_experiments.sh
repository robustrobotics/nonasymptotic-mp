#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task 8
#SBATCH --time=12:00:00

i=1
current_time=$(date +"%Y%m%d-%H%M%S")  # Getting the current time in YYYYMMDD-HHMMSS format
for seed in $(seq 1 40)
do
    for algorithm in "prm"
    do
        for n in "10" "50" "100" "500" "1000" 
        do
            if [ $((i)) -eq  $((SLURM_ARRAY_TASK_ID + 0)) ]; then
                save_path="/home/gridsan/acurtis/runs/$algorithm-$seed-$n-$current_time"
                python run.py --mp-alg="$algorithm" --seed=$seed --save-dir="$save_path" --max-samples="$n"
            fi
            i=$((i+1))
        done
        if [ $((i)) -eq  $((SLURM_ARRAY_TASK_ID + 0)) ]; then
            save_path="/home/gridsan/acurtis/runs/$algorithm-$seed-$n-$current_time"
            python run.py --mp-alg="$algorithm" --adaptive --seed=$seed --save-dir="$save_path" --max-samples="$n"
        fi
        i=$((i+1))
    done
done