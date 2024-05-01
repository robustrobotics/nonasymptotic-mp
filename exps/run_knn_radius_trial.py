import time

import numpy as np
import pandas as pd

from itertools import product
import json
import sys
import os

from nonasymptotic.envs import StraightLine
from nonasymptotic.ann import get_ann


def knn_radius_trial(delta_clear, dim, sample_schedule, neighbor_schedule, rng_seed):
    env = StraightLine(dim, delta_clear, rng_seed)
    knn_builder = get_ann("kgraph")
    record = []

    for s in sample_schedule:
        data = np.zeros((s, dim))
        for i_s in range(s):
            data[i_s, :] = env.sample_from_env()

        k_neighbors = np.max([_nn for _nn in neighbor_schedule if _nn < s])

        graph_build_st = time.time()
        _, dist_list = knn_builder.new_graph_from_data(data, k_neighbors)
        graph_build_et = time.time()

        build_time = graph_build_et - graph_build_st

        for nn in neighbor_schedule:
            if nn > s:
                break
            effective_rad = np.min(dist_list[:, nn - 1])
            record.append(
                (dim, s, nn, effective_rad, build_time, delta_clear, rng_seed)
            )

        df_record = pd.DataFrame(record, columns=[
            "dim", "n_samples", "n_neighbors",
            "eff_rad", "build_time",
            "delta_clearance", "seed"
        ])

    return df_record


# ==== TASK SCHEDULING CODE VVVV =======================================
# =======================================================================
task_id = int(sys.argv[1])
num_tasks = int(sys.argv[2])
exp_config_path = str(sys.argv[3])
exp_save_path = str(sys.argv[4])

with open(exp_config_path, 'r') as f:
    config = json.load(f)

# generate the options and select the ones this task will perform
print('beginning task id %i num_tasks %i' % (task_id, num_tasks))

exp_combos = list(product(
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

for _dim, _trial in tasks:
    print('initiating knn trial dim: %f, trial%f' % (_dim, _trial))
    trial_record_df = knn_radius_trial(
        delta_clear=config['delta'],
        dim=_dim,
        rng_seed=rng.integers(0, 2 ** 32),
        sample_schedule=config['sample_schedule'],
        neighbor_schedule=config['neighbor_schedule'],
    )

    trial_record_df['trial'] = _trial

    if all_tasks_record_df is None:
        all_tasks_record_df = trial_record_df
    else:
        all_tasks_record_df = pd.concat([all_tasks_record_df, trial_record_df])

    # we output intermediate results just to see progress.
    all_tasks_record_df.to_csv(df_save_path)
