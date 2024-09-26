import subprocess
import time
import random
import os

MAX_TOTAL = 100
NUM_RUNS_PER = 50
import numpy as np

def count_lines_of_command_output():
    try:
        squeue_command = "squeue -u \"`echo $USER`\""
        cancel_command = "scancel -u \"`echo $USER`\""
        # Execute the command and capture its output
        result = subprocess.run(squeue_command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Count the number of lines in the output
        output_lines = result.stdout.strip().split('\n')
        
        failed_output_lines = [ol for ol in output_lines if "held" in ol.lower()]
        print("Squeue returned {} lines".format(str(len(output_lines)-1)))
        print("{} of them were failed".format(str(len(failed_output_lines))))
        if(len(failed_output_lines)==(len(output_lines)-1) and len(output_lines)-1 > 0):
            print("Executing scancel")
            _ = subprocess.run(cancel_command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        return len(output_lines)-1
    
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        return 0
    
def deploy_with_args(delta, collision_buffer, theta_margin, debug=False):
    command_str = f"sbatch --array=1-1 deploy_experiments_kuka_adaptive.sh {delta} {collision_buffer} {theta_margin}"
    print(command_str)
    if(not debug):
        subprocess.run(command_str, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        

if __name__ == "__main__":
    # An arg set is (min_samples, max_samples, adaptive-n bool)
    debug=False
    min_min = 500
    max_max = 30000
    MAX_JOBS = 200
    queue_size = 1
    arg_sets = []

    for delta in [0.02, 0.03, 0.04]:
        for cb in [0.02, 0.03, 0.04]:
            for t in [np.pi/18.0, np.pi/32.0]:
                arg_sets += [(delta, cb, t)] # adaptive bound

    num_points = 20
    i=0
    queue = list(arg_sets)*5
    assert len(queue)< MAX_JOBS
    for experiment in queue:
        deploy_with_args(*experiment, debug=debug)