import subprocess
import numpy as np

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
    

if __name__ == "__main__":
    # An arg set is (min_samples, max_samples, adaptive-n bool)
    min_min = 100
    max_max = 30000
    MAX_JOBS = 200
    arg_sets = [(0,0,1), (min_min, max_max, 0)]
    for num_samples in np.linspace(min_min, max_max, 30):
        arg_sets.append((int(num_samples), int(num_samples), 0))
    
    print(arg_sets)
    
    