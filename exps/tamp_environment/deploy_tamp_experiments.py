import subprocess
import numpy as np
import time
import random

def count_lines_of_command_output(command):
    try:
        # Execute the command and capture its output
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        
        # Count the number of lines in the output
        output_lines = result.stdout.strip().split('\n')
        line_count = len(output_lines)
        
        return line_count
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        return 0
    
def deploy_with_args(min_samples, max_samples, adaptive, debug=False):
    command_str = f"sbatch --array=1-100 deploy_experiments.sh {min_samples} {max_samples} {adaptive}"
    if(not debug):
        subprocess.run(command_str, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    else:
        print(command_str)

if __name__ == "__main__":
    # An arg set is (min_samples, max_samples, adaptive-n bool)
    debug=False
    min_min = 100
    max_max = 50000
    MAX_JOBS = 200
    queue_size = 1
    arg_sets = [(0, 0, 1)] # adaptive bound
    num_points = 20
    for i in range(num_points):
        n = random.uniform(min_min+1, max_max)
        arg_sets.append((min_min-1, int(n), 0))
        arg_sets.append((int(n)-1, int(n), 0))

    for i, arg_set in enumerate(arg_sets):
        while(queue_size>0):
            print("Queue size: "+str(queue_size))
            if(not debug):
                queue_size = count_lines_of_command_output("squeue -u \"`echo $USER`\"")-2
                time.sleep(5)
        
        print("deploying {}/{}".format(i, len(arg_sets)))
        deploy_with_args(*arg_set, debug=debug)
        
    
    
    