from nonasymptotic.envs import StraightLine
from nonasymptotic.prm import SimpleRadiusPRM

import time
import pandas as pd
import numpy as np


# TODO: modify straight_line_trial to
# accept a resolution of epsilons we are interested in
# binary search down an ordering of the nearest neighbors until we find the neighbor we lose epsilon-delta optimal
# (or at least be sensitive to timeouts to have a gray zone)
# then report back the epsilon gray-zone between the node that must be included in the next NN dist down as the
# table entry, along with info.

def straight_line_trial(delta_clear, epsilon_tol, dim, rng_seed,
                        sample_schedule=None,
                        n_samples_per_ed_check_round=100,
                        ed_check_timeout=60.0,
                        area_tol=1e-8,
                        max_k_connection_neighbors=2048):
    if sample_schedule is None:
        # default 'small experiment' sample schedule
        sample_schedule = [10, 100, 1000]

    # set up experiment objs
    conn_radius = 2 * (1 + epsilon_tol) * delta_clear / np.sqrt(1 + epsilon_tol ** 2)
    env = StraightLine(dim=dim, delta_clearance=delta_clear, seed=rng_seed)
    prm = SimpleRadiusPRM(conn_radius, env.is_motion_valid, env.sample_from_env, env.distance_to_path,
                          max_k_connection_neighbors=max_k_connection_neighbors, seed=rng_seed, verbose=True)

    # lists to save records (we'll form a dataframe afterward)
    ed_complete_record = []
    info_record = []
    build_runtimes = []
    check_runtimes = []
    k_prms = []

    # run along sample schedule. Once we are ed-complete, we always will be, so we skip the rest when we are.\
    is_ed_complete = False
    for n_samples in sample_schedule:

        if not is_ed_complete:
            print('Growing prm to %i samples...' % n_samples)
            start_t = time.process_time()
            prm.grow_to_n_samples(n_samples)
            end_t = time.process_time()
            build_runtimes.append(end_t - start_t)
            print('build time: %f' % build_runtimes[-1])

            print('Initiating ed check...')
            start_t = time.process_time()
            is_ed_complete, info = env.is_prm_epsilon_delta_complete(prm, tol=epsilon_tol,
                                                                     n_samples_per_check=n_samples_per_ed_check_round,
                                                                     timeout=ed_check_timeout,
                                                                     area_tol=area_tol)
            end_t = time.process_time()
            check_runtimes.append(end_t - start_t)
            print('check time: %f' % check_runtimes[-1])

            ed_complete_record.append(is_ed_complete)
            info_record.append(info)
            k_prms.append(prm.k_neighbors)

        else:
            build_runtimes.append(np.NaN)
            check_runtimes.append(np.NaN)
            ed_complete_record.append(True)
            info_record.append('covered with fewer samples')
            k_prms.append(np.NaN)

    # construct dataframe and return
    print('Recording data and returning trial...')
    len_schedule = len(sample_schedule)
    df_record = pd.DataFrame(
        {
            'n_samples': sample_schedule,
            'ed_complete': ed_complete_record,
            'info': info_record,
            'build_time': build_runtimes,
            'check_time': check_runtimes,
            'k_prm': k_prms
        }
    )

    df_record['dim'] = dim
    df_record['delta_clearance'] = delta_clear
    df_record['epsilon_tolerance'] = epsilon_tol
    df_record['seed'] = rng_seed

    return df_record
