"""
A simple Monte-Carlo type experiment in environments with varying clearances.
"""

import time

import numpy as np
import pandas as pd

from itertools import product
import json
import sys
import os

from nonasymptotic.envs import NarrowPassage
from nonasymptotic.prm import SimpleFullConnRadiusPRM, SimpleNearestNeighborRadiusPRM


def narrow_passage_clearance(delta_clear, dim, rng_seed,
                             sample_schedule,
                             prm_type,  # can only be 'radius' or 'knn'
                             max_connections  # can be a positive floar for radius or integer for # of neighbors
                             ):
    assert prm_type in ["knn", "radius"], "chosen PRM type must be one of: knn, radius."
    # set up experiment objs
    env = NarrowPassage(dim=dim, clearance=delta_clear, seed=rng_seed)

    if prm_type == 'knn':
        prm = SimpleNearestNeighborRadiusPRM(max_connections,
                                             env.is_motion_valid,
                                             env.sample_from_env,
                                             env.distance_to_path,
                                             truncate_to_eff_rad=False,
                                             seed=rng_seed, verbose=True)
    else:
        prm = SimpleFullConnRadiusPRM(
            max_connections,
            env.is_motion_valid,
            env.sample_from_env,
            seed=rng_seed, verbose=True)

    # we'll save all the data in tuple-record format.
    result_record = []

    # run along sample schedule. Once we are ed-complete, we always will be, so we skip the rest when we are.
    for n_samples in sample_schedule:
        print('Growing prm to %i samples...' % n_samples)
        start_t = time.time()

        # build prm and set up rad search array
        if prm_type == 'knn':
            prm.grow_to_n_samples(n_samples)
            i_conn_lb = 3
            i_conn_ub = max_connections
        else:
            nn_rads = prm.grow_to_n_samples(n_samples)  # nn_rads is ordered ascending
            nn_rads = np.unique(nn_rads)
            i_conn_lb = 1
            i_conn_ub = nn_rads.size - 1  # we're sticking to integers so we can maximize code sharing

        end_t = time.time()
        build_runtime = end_t - start_t

        # binary search down to radius
        query_start = np.array([-1.0] + [0.0] * (dim - 1))
        query_end = np.array([1.0] + [0.0] * (dim - 1))

        start_t = time.time()
        info = ''
        while True:
            if i_conn_lb + 1 == i_conn_ub:
                break
            elif i_conn_lb >= i_conn_ub:
                # again, flag it some weird behavior... but don't shut the exp down since we may get valuable results
                info = 'something went wrong... rad_lb is not supposed to be >= rad_ub'
                break

            i_conn_test = int((i_conn_lb + i_conn_ub) / 2)

            # now, we split based on PRM time again
            if prm_type == 'knn':
                kn_test = i_conn_test
                prm.set_nearest_neighbors(kn_test)
            else:
                rad_test = nn_rads[i_conn_test]
                prm.set_connection_radius(rad_test)

            _, path = prm.query_best_solution(query_start, query_end)
            prm_has_path = len(path)

            if prm_has_path:
                i_conn_ub = i_conn_test
            else:
                i_conn_lb = i_conn_test

        end_t = time.time()
        check_runtime = end_t - start_t

        failed_conn = i_conn_lb if prm_type == 'knn' else nn_rads[i_conn_lb]
        succeed_conn = i_conn_ub if prm_type == 'knn' else nn_rads[i_conn_ub]
        result_record.append(
            (prm_type,
             dim,
             delta_clear,
             n_samples,
             failed_conn,
             succeed_conn,
             build_runtime,
             check_runtime,
             info,
             rng_seed)
        )

    # delete temporary data associated with PRM
    prm.reset()

    # construct dataframe and return
    print('Recording data and returning trial...')
    df_record = pd.DataFrame(result_record,
                             columns=[
                                 'prm_type',
                                 'dim',
                                 'delta_clearance',
                                 'n_samples',
                                 'conn_lb',
                                 'conn_ub',
                                 'build_time',
                                 'check_time',
                                 'info',
                                 'seed'
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
    ['radius', 'knn'],
    config['deltas'],
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

for _prm_type, _clearance, _dim, _trial in tasks:
    print('initiating line trial dim: %f, trial%f' % (_dim, _trial))
    _sample_schedule = config['radius_sample_schedule'] if _prm_type == 'radius' else config['knn_sample_schedule']
    _max_conn = config['radius_prm_max_radius'] if _prm_type == 'radius' else config['knn_prm_max_neighbors']
    trial_record_df = narrow_passage_clearance(
        _clearance,
        _dim,
        rng_seed=rng.integers(0, 2 ** 32),
        sample_schedule=_sample_schedule,
        prm_type=_prm_type,
        max_connections=_max_conn,
    )

    trial_record_df['trial'] = _trial

    if all_tasks_record_df is None:
        all_tasks_record_df = trial_record_df
    else:
        all_tasks_record_df = pd.concat([all_tasks_record_df, trial_record_df])

    # we output intermediate results just to see progress.
    all_tasks_record_df.to_csv(df_save_path)
