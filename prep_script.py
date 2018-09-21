import argparse
import pandas as pd
import numpy as np
import math
import glob
import os.path as op
from itertools import compress
from utilities import files, tools
from scipy.interpolate import interp1d
from scipy import where

"""
Script for data processing and preparation, circular motion joystick task. 
Usage:
python prep_script.py -n 0 -i $INPUT -o $OUTPUT -s 0,1,2,3,4

    -n [index in a list of participants]
    -i [path to the raw data]
    -o [path to the output]
    -s [steps performed e.g. 0,1,2,3,4,5]

export INPUT="~/git/stirring_beh_analysis/data"
export OUTPUT="~/git/stirring_beh_analysis/data"

Author: Maciej Szul, 2018
"""

pd.set_option("mode.chained_assignment", None)

# command line interface
des = "prep script command line"
parser = argparse.ArgumentParser(description=des)

parser.add_argument("-n", type=int, help="index in a list of participants")
parser.add_argument("-i", type=str, help="path to the raw data")
parser.add_argument("-o", type=str, help="path to the output")
parser.add_argument("-s", type=str, help="steps performed")

args = parser.parse_args()
params = vars(args)
pp_ix = params["n"]
input_path = params["i"]
output_path = params["o"]
steps = params["s"]
steps = eval("["+ steps +"]")

# variables for the script
f_types = [
    "arrow_training",
    "stir_training_50",
    "stir_training_main",
    "stir_main_LAB",
    "stir_main_MEG"
]

d_types = [
    "buffer_",
    "joystick_",
    "trial_"
]

time = np.linspace(0.0, 4.0, num=1000)

# functions

def items_cont_str(input_list, string, sort=False):
    """
    returns a list of items which contain a given string
    optionally sorted
    """
    output_list = [string in i for i in input_list]
    output_list = list(compress(input_list, output_list))
    if sort:
        output_list.sort()
    return output_list


def resamp_interp(x, y, new_x):
    """
    returns resampled an interpolated data
    """
    resamp = interp1d(x, y, kind='slinear', fill_value='extrapolate')
    new_data = resamp(new_x)
    return new_data


def to_polar(x, y):
    radius = []
    angle = []
    xy = zip(x, y)
    for x, y in xy:
        rad, theta = tools.cart2polar(x, y)
        theta = math.degrees(theta)
        angle.append(theta)
        radius.append(rad)
    del xy
    radius = np.array(radius)
    angle = np.array(angle)
    return [angle, radius]


def nan_cleaner(arr):
    """
    clears nan values and interpolates the missing value
    """
    mask = np.isnan(arr)
    arr[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), arr[~mask])
    return arr


def calculate_degs(angle, radius):
    degs = np.diff(angle)
    degs = np.insert(degs, 0, 0)
    degs[np.abs(degs) > 300] = np.nan
    degs = nan_cleaner(degs)
    degs = degs * radius
    return degs


print(steps)

if 0 in steps:
    pp_paths = files.get_folders_files(input_path, wp=True)[0]
    pp_paths.sort()
    pp_path = pp_paths[pp_ix]
    pp_name = pp_path[-3:]
    print(pp_name, pp_path)
    glob_path = op.join(pp_path, "**", "*.csv")
    all_csvs = [i for i in glob.iglob(glob_path, recursive=True)]
    f_ses = {i: items_cont_str(all_csvs, i, sort=True) for i in f_types}
    files_all = dict()
    for tp in f_types:
       files_all[tp] = {d: items_cont_str(f_ses[tp], d, sort=True) for d in d_types}


if 1 in steps:
    # iteration over files_all here
    trial_csv = "/home/maciek/git/stirring_beh_analysis/data/042/ses_000/trial_sub042_ses000_run000_trial000_arrow_training_2017_May_17_1346.csv"
    buffer_csv = "/home/maciek/git/stirring_beh_analysis/data/042/ses_000/buffer_sub042_ses000_run000_trial000_arrow_training_2017_May_17_1346.csv"
    joystick_csv = "/home/maciek/git/stirring_beh_analysis/data/042/ses_000/joystick_sub042_ses000_run000_trial000_arrow_training_2017_May_17_1346.csv"
    trial_data = pd.read_csv(trial_csv, index_col=False)
    buffer_data = pd.read_csv(buffer_csv, index_col=False)
    joystick_data = pd.read_csv(joystick_csv, index_col=False)
    
    joystick = pd.concat([buffer_data, joystick_data], axis=0, ignore_index=True)
    joystick = joystick[["t", "x", "y"]]
    joystick.sort_values(by=["t"], inplace=True,)
    joystick.reset_index(level=0, drop=True, inplace=True)

    t = np.array(joystick.t) - trial_data.stim_onset.values[0]
    x = np.array(joystick.x)
    y = np.array(joystick.y)

    del joystick, buffer_data, joystick_data
    
    x = resamp_interp(t, x, time)
    y = resamp_interp(t, y, time)
    
    angle, radius = to_polar(x,y)

    degs = calculate_degs(angle, radius)

    cols = [
        'change_start', 'change_stop', 'clockwise', 'conditions', 'cue_onset',
        'iti', 'run', 'run_start', 'session', 'stim_onset', 'subject', 'trial',
        'trial_end'
    ]

    trial_data = trial_data[cols]
    for i in["x", "y", "angle", "degs"]:
        trial_data[i] = None
        trial_data[i] = trial_data[i].astype(object)
        trial_data[i].loc[0] = eval(i)


