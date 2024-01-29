"""
Sets up experiment directory, generates the necessary shell scripts, and then submits the job.
We assume that this script is being run from nonasymptotic-mp/exps/ directory.
"""
from datetime import datetime
import argparse
import shutil
import sys
import os
import subprocess

submit_sh_script_prototype_text = """#!/bin/bash

# Load Anaconda Module
module load anaconda/2022b

echo "My task ID: " $LLSUB_RANK
echo "Number of Tasks: " $LLSUB_SIZE
export PYTHONPATH=$PYTHONPATH:$PWD/../

python run_straight_line_trial.py $LLSUB_RANK $LLSUB_SIZE {0} {1}"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True,
                        help="Name of the experiment.")
    parser.add_argument("--config-path", type=str, required=True,
                        help="Path to the config file with experiment params.")
    parser.add_argument("--triples-args", type=int, nargs=3, required=True,
                        help="The three integers that dictate the SuperCloud triples configuration "
                             "[Nodes, N Proc per node, N threads per proc]")
    args = parser.parse_args()

    # make experiment directory and copy over config
    now = datetime.now()
    now_str = now.strftime('%Y%m%d-%H%M%S')
    save_path = "./results/" + args.name + '_' + now_str

    try:
        os.mkdir(save_path)
    except FileExistsError:
        print('Experiment %s already run. Try a new name?' % args.name)
        sys.exit(1)
    except FileNotFoundError:
        print("Couldn't set up the experiment directory. Did you run from 'nonasymptotic-mp/exps?'")
        sys.exit(1)

    config_path = os.path.join(save_path, 'config.json')
    try:
        shutil.copyfile(args.config_path, config_path)
    except FileNotFoundError:
        print("Could not find the config file '%s'." % args.config_path)

    # generate the submission script (and store in experiment directory)
    submit_script_text = submit_sh_script_prototype_text.format(config_path, save_path)
    submit_script_path = os.path.join(save_path, 'submit_job.sh')
    with open(submit_script_path, 'w') as submit_script_file:
        submit_script_file.write(submit_script_text)

    # submit and queue up the JobArray
    subprocess.run(['chmod', 'u+x', submit_script_path])
    subprocess.run(['LLsub', submit_script_path, '[%i,%i,%i]' % tuple(args.triples_args)])
