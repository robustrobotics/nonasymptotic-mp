import numpy as np
import matplotlib.pyplot as plt
import os
import re
from collections import defaultdict

# Set the path to the directory containing the subfolders
parent_directory = './runs'

# This list will hold all the numbers found
successes = {}
times = {}
adaptive_n = {}
min_samples = {}
max_samples = {}
# Regular expression to find numbers
number_pattern = re.compile(r'\d+')


for root, dirs, files in os.walk(parent_directory):
    for file in files:
        print(file)
        if file.endswith(".log"):  # Checks if the file is a log file
            file_path = os.path.join(root, file)
            with open(file_path, 'r') as f:
                lines = f.readlines()
                for i, line in enumerate(lines):
                    if("Namespace" in line):
                        match = re.search(r"adaptive_n=([^ ,]+),", line)
                        adaptive_n[file] = int(bool(match.group(1)=="True"))
                        match = re.search(r"min_samples=([^ ,]+),", line)
                        min_samples[file] = int(match.group(1))
                        match = re.search(r"max_samples=([^ ,]+),", line)
                        max_samples[file] = int(match.group(1))
                    if line.startswith("Solved:"):
                        successes[file] = int(("True" in lines[i].strip()))
                    if(line.startswith("Time:")):
                        times[file] = float(lines[i].strip().replace("Time: ", ""))

print(len(adaptive_n))
print(len(min_samples))
print(len(max_samples))
print(len(successes))
print(len(times))

plot_points = {}
plot_successes = defaultdict(list)
plot_times = defaultdict(list)
for key in times:
    plot_successes[(min_samples[key], max_samples[key], adaptive_n[key])].append(successes[key])
    plot_times[(min_samples[key], max_samples[key], adaptive_n[key])].append(times[key])


print(plot_successes)
# Calculating means and standard deviations for plotting
groups = plot_successes.keys()
success_means = {group: np.mean(plot_successes[group]) for group in groups}
success_stds = {group: np.std(plot_successes[group])/np.sqrt(len(plot_successes[group])) for group in groups}
time_means = {group: np.mean(plot_times[group]) for group in groups}
time_stds = {group: np.std(plot_times[group])/np.sqrt(len(plot_successes[group])) for group in groups}


# Plotting
fig, ax1 = plt.subplots()

bar_width = 0.35
index = np.arange(len(groups))

# Plot success rate on the primary (left) y-axis
bars1 = ax1.bar(index - bar_width/2, [success_means[group] for group in groups], bar_width,
               yerr=[success_stds[group] for group in groups], color='b', label='Success Rate')

ax1.set_xlabel('Group')
ax1.set_ylabel('Success Rate', color='b')
ax1.tick_params('y', colors='b')
ax1.set_xticks(index)
ax1.set_xticklabels([f'{min_}-{max_}-{adapt}' for min_, max_, adapt in groups])

# Create a secondary y-axis for time
ax2 = ax1.twinx()
bars2 = ax2.bar(index + bar_width/2, [time_means[group] for group in groups], bar_width,
               yerr=[time_stds[group] for group in groups], color='r', label='Time')

ax2.set_ylabel('Time', color='r')
ax2.tick_params('y', colors='r')

fig.tight_layout()
fig.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), ncol=2)

plt.show()
