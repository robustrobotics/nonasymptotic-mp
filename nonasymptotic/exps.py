from nonasymptotic.envs import StraightLine
from nonasymptotic.prm import SimplePRM

import pandas as pd
import numpy as np


def straight_line_trial(delta_clear, epsilon_tol, dim, rng_seed,
                        sample_schedule=None,
                        n_samples_per_ed_check_round=100,
                        ed_check_timeout=60.0,
                        area_tol=1e-8):
    if sample_schedule is None:
        sample_schedule = [10, 100, 1000, 10000]

    # set up experiment objs
    conn_radius = 2 / np.sqrt(1 + epsilon_tol ** 2) * (epsilon_tol + delta_clear)
    env = StraightLine(dim=dim, delta_clearance=delta_clear, seed=rng_seed)
    prm = SimplePRM(conn_radius, env.is_motion_valid, env.sample_from_env, env.distance_to_path, seed=rng_seed)

    # run along sample schedule. Once we are ed-complete, we always will be, so we skip the rest when we are.
    # TODO: add record keeping mechanism to return
    is_ed_complete = False
    for n_samples in sample_schedule:

        if not is_ed_complete:
            prm.grow_to_n_samples(n_samples)

            # timing data would be useful

            is_ed_complete, info = env.is_prm_epsilon_delta_complete(prm, tol=epsilon_tol,
                                                                     n_samples_per_check=n_samples_per_ed_check_round,
                                                                     timeout=ed_check_timeout,
                                                                     area_tol=area_tol)
        else:
            pass
