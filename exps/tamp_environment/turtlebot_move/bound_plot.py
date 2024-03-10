from nonasymptotic.util import compute_numerical_bound
import numpy as np
import matplotlib.pyplot as plt
import os
import re

# Set the path to the directory containing the subfolders
parent_directory = './runs'

# This list will hold all the numbers found
numbers = {}
samples = {}

# Regular expression to find numbers
number_pattern = re.compile(r'\d+')

for root, dirs, files in os.walk(parent_directory):
    for file in files:
        if file.endswith(".log"):  # Checks if the file is a log file
            file_path = os.path.join(root, file)
            with open(file_path, 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    if "Delta:" in line and i+1 < len(lines):
                        numbers[file] = float(lines[i+1].strip())

                    if "Found solution in" in line and i+1 < len(lines):
                        samples[file] = float(lines[i+1].strip())

print(len(samples))
bounds = []
space = list(np.linspace(0.025, 0.2, 50))
for delta in space:
    n, _ = compute_numerical_bound(delta, 0.9, 1, 2, None)
    bounds.append(n)
plt.plot(space, bounds)

# plt.scatter(*zip(*[(numbers[k], v) for k, v in samples.items()]))
plt.show()