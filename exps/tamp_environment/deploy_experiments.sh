#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task 8
#SBATCH --time=24:00:00

i=1
current_time=$(date +"%Y%m%d-%H%M%S")  # Getting the current time in YYYYMMDD-HHMMSS format
for seed in $(seq 1 200)
do
    for algorithm in "prm"
    do
        if [ $((i)) -eq  $((SLURM_ARRAY_TASK_ID + 0)) ]; then
            save_path="/home/gridsan/acurtis/runs/$algorithm-$seed-$n-$current_time"
            if [ "$3" -eq 0 ]; then
                python ./turtlebot_move/run.py --seed=$seed --save-dir="$save_path" --min-samples="$1" --max-samples="$2"
            elif [ "$3" -eq 1 ]; then
                python ./turtlebot_move/run.py --seed=$seed --save-dir="$save_path" --adaptive-n
            fi
        fi
        i=$((i+1))
    done
done