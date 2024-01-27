from exps.trial_util import straight_line_trial

from datetime import datetime
from itertools import product

import numpy as np
import os, sys, json

now = datetime.now()
save_dir = './results/straight_line%s/' % now.strftime("%m-%d-%Y_%H-%M-%S")

task_id = int(sys.argv[1])
num_tasks = int(sys.argv[2])
exp_config_path = int(sys.argv[3])

with open(exp_config_path, 'r') as f:
    config = json.load(f)

# generate the options and select the ones this task will perform

deltas = np.linspace(0.0, 1.0, num=config['n_deltas'] + 1)[1:]
epsilons = np.linspace(0.0, 1.0, num=config['n_epsilons'] + 1)[1:]

exp_combos = list(product(
    deltas,
    product(epsilons, range(config['n_trials']))
))

# assign tasks to this process
tasks = exp_combos[task_id:len(exp_combos):num_tasks]

# run the experiments!
for _delta, _epsilon, _trial in tasks:
    pass
