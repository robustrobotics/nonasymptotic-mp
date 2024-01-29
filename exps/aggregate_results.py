"""
Aggregate experiment results into a single dataframe, and then compute
some statistics about the _run_. Figures will be computed in notebooks.
"""
import pandas as pd
import argparse
import glob
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name-prefix', type=str, required=True)
    parser.add_argument('--df-out-name', type=str, default=None)
    args = parser.parse_args()

    exp_dirs = glob.glob('results/%s*' % args.name_prefix)
    exp_dirs.sort(key=os.path.getmtime)
    latest_exp_dir = exp_dirs[-1]

    mini_out_csvs = glob.glob(latest_exp_dir + '/out*.csv' % args.name_prefix)
    big_out = pd.concat(list(map(pd.read_csv, mini_out_csvs)))

    # time statistics
    mean = big_out['time'].mean()
    std = big_out['time'].std()

    print('Trial exec time mean: %f +/- %f' % (mean, std))

    if args.df_out_name is not None:
        big_out.to_csv(latest_exp_dir + '/' + args.df_out_name + '_agg_out.csv')
    else:
        big_out.to_csv(latest_exp_dir + '/' + args.name_prefix + '_agg_out.csv')
