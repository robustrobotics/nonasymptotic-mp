import subprocess
import time
import random
import os

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
        assert os.path.isfile("./tmux_lock.txt")

        queue_size = count_lines_of_command_output()

        while(queue_size>49):
            print("Queue size: "+str(queue_size))
            if(not debug):
                queue_size = count_lines_of_command_output()
                time.sleep(30)
        
        print("deploying {}/{}".format(i, len(arg_sets)))
        deploy_with_args(*arg_set, debug=debug)
        time.sleep(30)
        
    
    
    