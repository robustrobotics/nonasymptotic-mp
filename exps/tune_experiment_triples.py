"""
Experiment triple tuner. Since the number of nodes just increases the capacity
over the number of processes we can fire off, we'll tune NPPN and NThreads and
then max out our node allocation on a job request.
"""
import subprocess
import argparse

nppns_to_test = [1, 2, 4, 8, 16, 24, 32, 48]
n_threads_to_test = [1, 2, 4, 8, 16, 24, 32, 48]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--triple-to-test', choices=['nppn', 'threads'], required=True)
    parser.add_argument('--other-arg-fill-in', type=int, required=True)
    args = parser.parse_args()

    if args.triple_to_test == 'threads':
        for n_threads in n_threads_to_test:
            if n_threads * args.other_arg_fill_in > 48:
                continue
            subprocess.run([
                "python", "submit_and_run_experiment.py",
                "--name triples-tune",
                "--config-path config/triples_tune_run.json",
                "--triples-args [1, %i, %i]" % (args.other_arg_fill_in, n_threads)
            ])

    elif args.triple_to_test == 'nppn':
        for nppn in nppns_to_test:
            if nppn * args.other_arg_fill_in > 48:
                continue
            else:
                subprocess.run([
                    "python", "submit_and_run_experiment.py",
                    "--name triples-tune",
                    "--config-path config/triples_tune_run.json",
                    "--triples-args [1, %i, %i]" % (nppn, args.other_arg_fill_in)
                ])
    else:
        print('Unknown triple-to-test.')
