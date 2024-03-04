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

    mini_out_csvs = glob.glob(latest_exp_dir + '/out*.csv')
    mini_out_dfs = list(map(lambda p: pd.read_csv(p, index_col=0), mini_out_csvs))
    big_out = pd.concat(mini_out_dfs)

    if args.df_out_name is not None:
        big_out.to_csv(latest_exp_dir + '/' + args.df_out_name + '_agg_out.csv')
    else:
        big_out.to_csv(latest_exp_dir + '/' + args.name_prefix + '_agg_out.csv')

    # time statistics (exclude first run since that requires jitting)
    mini_out_dfs_cut = list(map(lambda df: df.tail(df.shape[0] - 1), mini_out_dfs))
    cut_big_out = pd.concat(mini_out_dfs_cut)
    build_mean = cut_big_out['build_time'].mean()
    build_std = cut_big_out['build_time'].std()

    check_mean = cut_big_out['check_time'].mean()
    check_std = cut_big_out['check_time'].std()

    print('Trial build time mean: %f +/- %f' % (build_mean, build_std))
    print('Trial check time mean: %f +/- %f' % (check_mean, check_std))
