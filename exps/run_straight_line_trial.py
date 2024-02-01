from exps.trial_util import straight_line_trial
import numpy as np
import pandas as pd

from datetime import datetime
from itertools import product
import json
import sys
import os

now = datetime.now()

task_id = int(sys.argv[1])
num_tasks = int(sys.argv[2])
exp_config_path = str(sys.argv[3])
exp_save_path = str(sys.argv[4])

with open(exp_config_path, 'r') as f:
    config = json.load(f)

# generate the options and select the ones this task will perform
print('beginning task id %i num_tasks %i' % (task_id, num_tasks))

deltas = np.linspace(0.0, 1.0, num=config['n_deltas'] + 1)[1:]
epsilons = np.linspace(0.0, 1.0, num=config['_epsilons'] + 1)[1:]

exp_combos = list(product(
    deltas,
    epsilons,
    config['dims_to_test'],
    range(config['n_trials'])
))

# assign tasks to this process
tasks = exp_combos[task_id:len(exp_combos):num_tasks]

# trying to obtain reproducible randomness
rng = np.random.default_rng(seed=task_id)

# run the experiments! check to see if an out file already exists and the experiment is resuming.
df_save_path = os.path.join(exp_save_path, 'out%i.csv' % task_id)
if os.path.exists(df_save_path):
    all_tasks_record_df = pd.read_csv(df_save_path, index_col=0)
    tasks = tasks[all_tasks_record_df.shape[0]:]  # skip ahead if we have completed entries
else:
    all_tasks_record_df = None

for _delta, _epsilon, _dim, _trial in tasks:
    print('initiating delta: %f, epsilon: %f, dim: %f, trial%f' % (_delta, _epsilon, _dim, _trial))
    trial_record_df = straight_line_trial(
        _delta, _epsilon, _dim,
        rng_seed=rng.integers(0, 2 ** 32),
        sample_schedule=config['sample_schedule'],
        n_samples_per_ed_check_round=config['n_samples_per_check_round'],
        ed_check_timeout=config['ed_check_timeout'],
        area_tol=config['area_tol'],
        max_k_connection_neighbors=config['prm_max_k_neighbors'],
    )

    trial_record_df['trial'] = _trial

    if all_tasks_record_df is None:
        all_tasks_record_df = trial_record_df
    else:
        all_tasks_record_df = pd.concat([all_tasks_record_df, trial_record_df])

    # we output intermediate results just to see progress.
    all_tasks_record_df.to_csv(df_save_path)
