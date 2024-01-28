"""
Experiment triple tuner. Since the number of nodes just increases the capacity
over the number of processes we can fire off, we'll tune NPPN and NThreads and
then max out our node allocation on a job request.
"""
import time
import subprocess

import numpy as np

nppns_to_test = [1, 2, 4, 8, 16, 24, 32, 48]
n_threads_to_test = [1, 2, 4, 8, 16, 24, 32, 48]

# tune number of threads first
thread_times = []
for n_threads in n_threads_to_test:
    start_t = time.process_time()
    subprocess.run([
        "python", "submit_and_run_experiment.py",
        "--name triples-tune",
        "--config-path config/triples_tune_run.json",
        "--triples-args [1, 1, %i]" % n_threads
    ])
    end_t = time.process_time()
    thread_times.append(end_t - start_t)

n_threads_best = n_threads_to_test[np.argmin(thread_times)]

nppn_times = []
for nppn in nppns_to_test:
    if nppn * n_threads_best > 48:
        nppn_times.append(float("inf"))
    else:
        start_t = time.process_time()
        subprocess.run([
            "python", "submit_and_run_experiment.py",
            "--name triples-tune",
            "--config-path config/triples_tune_run.json",
            "--triples-args [1, %i, %i]" % (nppn, n_threads_best)
        ])
        end_t = time.process_time()
        nppn_times.append(end_t - start_t)

nppn_best = nppns_to_test[np.argmin(nppn_times)]
print('Optimal NPPN X NTHREAD: [%i, %i], n_threads tuned first' % (nppn_best, n_threads_best))
