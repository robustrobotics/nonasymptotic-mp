# accept a resolution of epsilons we are interested in
# binary search down an ordering of the nearest neighbors until we find the neighbor we lose epsilon-delta optimal
# (or at least be sensitive to timeouts to have a gray zone)
# then report back the epsilon gray-zone between the node that must be included in the next NN dist down as the
# table entry, along with info.


# start with small samples, with constant # of nearest neighbors
# have: lower bound radius is 0.0, upper bound radius is the implicit radius checked.
# sort the nn dists as given by the PRM.
# then do binary search.
# if all goes well, we will find radius_lb which is too stringent, and radius_ub which is not.
# we could probably _further_ understand what the radius exactly needs to be, but that's a bit much.
# we'll just have an 'uknown interval' for now.

# then, we up the sample count. we already know that radius_ub already works... which
# imposes an upper bound on the distances (if helpful)
# if not, then we just have to redo the binary search again.

# it is totally possible that none of the radii (even the full graph), work. then we accept
# that we know that none of the radii represented by the graph are satisfactory.

# this will then offload more work to the post-processing/plotting code to understand
# the data

# if y = (1 + x) / np.sqrt(1 + x**2) => x = (1 - np.sqrt(2*y**2 - y**4)) / (y**2 - 1)


import time

import numpy as np
import pandas as pd

from itertools import product
import json
import sys
import os

from envs import StraightLine
from prm import SimpleNearestNeighborRadiusPRM


def straight_line_trial(delta_clear, dim, rng_seed,
                        sample_schedule=None,
                        n_samples_per_ed_check_round=100,
                        ed_check_timeout=60.0,
                        area_tol=1e-8,
                        max_k_connection_neighbors=254):
    if sample_schedule is None:
        # default 'small experiment' sample schedule
        sample_schedule = [10, 100, 1000]

    # set up experiment objs
    env = StraightLine(dim=dim, delta_clearance=delta_clear, seed=rng_seed)
    prm = SimpleNearestNeighborRadiusPRM(max_k_connection_neighbors,
                                         env.is_motion_valid,
                                         env.sample_from_env,
                                         env.distance_to_path,
                                         seed=rng_seed, verbose=True)

    # lists to save records (we'll form a dataframe afterward)
    radius_lbs = []
    radius_ubs = []
    info_record = []
    build_runtimes = []
    check_runtimes = []

    # run along sample schedule. Once we are ed-complete, we always will be, so we skip the rest when we are.\
    for n_samples in sample_schedule:
        print('Growing prm to %i samples...' % n_samples)
        start_t = time.time()

        # build prm and set up rad search array
        eff_radius_ub, nn_rads = prm.grow_to_n_samples(n_samples)  # nn_rads is ordered ascending
        nn_rads = nn_rads[:np.searchsorted(nn_rads, eff_radius_ub, side='right')]  # truncate up to radius_lb
        i_rad_lb = 1
        i_rad_ub = nn_rads.size - 1

        end_t = time.time()
        build_runtimes.append(end_t - start_t)

        print('build time: %f' % build_runtimes[-1])
        print('Initiating ed checks...')

        # binary search down to radius
        start_t = time.time()
        info = ''
        while True:
            if i_rad_lb + 1 == i_rad_ub:
                break
            elif i_rad_lb >= i_rad_ub:
                # again, flag it some weird behavior... but don't shut the exp down since we may get valuable results
                info = 'something went wrong... rad_lb is not supposed to be >= rad_ub'
                break

            i_rad_test = int((i_rad_lb + i_rad_ub) / 2)
            rad_test = nn_rads[i_rad_test]
            rad_normed = rad_test / 2 / delta_clear

            epsilon_tol = (1 - np.sqrt(2 * rad_normed ** 2 - rad_normed ** 4)) / (rad_normed ** 2 - 1)
            is_ed_complete, ed_info = env.is_prm_epsilon_delta_complete(prm, tol=epsilon_tol,
                                                                        n_samples_per_check=n_samples_per_ed_check_round,
                                                                        timeout=ed_check_timeout,
                                                                        area_tol=area_tol)

            # If we have a timeout, we just assume it's not solvable. But we flag it anyway.
            # Ideally we'd give enough of a timeout/check is fast enough to not worry about this...
            if ed_info == 'timed out':
                info += 'timed out on radius %f ==> eps_tol %f;' % (rad_test, epsilon_tol)

            if is_ed_complete:
                i_rad_ub = i_rad_test
            else:
                i_rad_lb = i_rad_test

        end_t = time.time()
        check_runtimes.append(end_t - start_t)

        print('check time: %f' % check_runtimes[-1])
        radius_lbs.append(nn_rads[i_rad_lb])
        radius_ubs.append(nn_rads[i_rad_ub])
        info_record.append(info)  # anything else we'd like to record from the run

    # delete temporary data associated with PRM
    prm.reset()

    # construct dataframe and return
    print('Recording data and returning trial...')
    df_record = pd.DataFrame(
        {
            'n_samples': sample_schedule,
            'rad_lbs': radius_lbs,
            'rad_ubs': radius_ubs,
            'info': info_record,
            'build_time': build_runtimes,
            'check_time': check_runtimes,
        }
    )

    df_record['dim'] = dim
    df_record['delta_clearance'] = delta_clear
    df_record['seed'] = rng_seed

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
    print('initiating line trial dim: %f, trial%f' % (_dim, _trial))
    trial_record_df = straight_line_trial(
        config['delta'], _dim,
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
