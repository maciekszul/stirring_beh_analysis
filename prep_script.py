import argparse
import pandas as pd
import numpy as np
import math
import glob
import os.path as op
from itertools import compress
from utilities import files, tools
import scipy.ndimage.filters as nd
from scipy.interpolate import interp1d
from scipy import where

"""
Script for data processing and preparation, circular motion joystick task. 
Usage:
python prep_script.py -n 0 -i $INPUT -s 0,1,2,3,4

    -n [index in a list of participants]
    -i [path to the raw data]
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
parser.add_argument("-s", type=str, help="steps performed")

args = parser.parse_args()
params = vars(args)
pp_ix = params["n"]
input_path = params["i"]
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

high = 'high'
low = 'low'
conditions = {0: (high, 0, high, 0),
              1: (high, 0, high, 180),
              2: (high, 0, low, 0),
              3: (high, 0, low, 180),
              4: (high, 180, high, 0),
              5: (high, 180, high, 180),
              6: (high, 180, low, 0),
              7: (high, 180, low, 180)
              }

bin_cond = {0: ((0, 5), 'no change of coherence and direction'),
            1: ((1, 4), 'change of direction only'),
            2: ((2, 7), 'change of coherence but no direction'),
            3: ((3, 6), 'change of both coherence and direction')
            }

cw = [{0: -1, 180: 1}, {0: 1, 180: -1}]

time = np.linspace(0.0, 4.0, num=1000)

blink_ix = (487, 512) # index of 1.95 and 2.05 timepoints

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


def change_dir(degs):
    all_changes = where(degs[:-1] * degs[1:] < 0)[0]
    changes = all_changes[all_changes > blink_ix[1]]
    if len(changes) >= 1:
        ch = changes[0]
    elif len(changes) == 0:
        ch = blink_ix[1]
    return ch


def acc_both_phases(signs, clockwise, condition):
    phase1 = signs[:blink_ix[1]]
    phase2 = signs[blink_ix[1]:]
    a, cond_ph1, a, cond_ph2 = conditions[condition]
    del a
    p1_bool = phase1 == cw[clockwise][cond_ph1]
    p2_bool = phase2 == cw[clockwise][cond_ph2]
    p1_acc = p1_bool.sum() / len(phase1)
    p2_acc = p2_bool.sum() / len(phase2)
    return (p1_acc, p2_acc)


if 0 in steps:
    pp_paths = files.get_folders_files(input_path, wp=True)[0]
    pp_paths.sort()
    pp_path = pp_paths[pp_ix]
    pp_name = pp_path[-3:]
    glob_path = op.join(pp_path, "**", "*.csv")
    all_csvs = [i for i in glob.iglob(glob_path, recursive=True)]
    f_ses = {i: items_cont_str(all_csvs, i, sort=True) for i in f_types}
    files_all = dict()
    for tp in f_types:
       files_all[tp] = {d: items_cont_str(f_ses[tp], d, sort=True) for d in d_types}


if 1 in steps:
    pp_trials = list()

    for exp_type in files_all.keys():
        files_len = len(files_all[exp_type]["joystick_"])
        for trial in range(files_len):
            trial_csv = files_all[exp_type]["trial_"][trial]
            buffer_csv = files_all[exp_type]["buffer_"][trial]
            joystick_csv = files_all[exp_type]["joystick_"][trial]
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

            degs = nd.gaussian_filter1d(degs, sigma=1.5)

            cols = [
                'clockwise', 'conditions', 'run', 'session', 'subject', 'trial'
            ]

            engage_ix = np.argmax(radius[:blink_ix[1]] >= 0.2)

            change_ix = change_dir(degs)

            signs = np.sign(degs)

            time_phase1, time_phase2 = acc_both_phases(signs, trial_data.clockwise.values[0], trial_data.conditions.values[0])
            
            del signs

            p_split = trial_csv.split(sep="/")

            filename = p_split[-1]

            trial_data = trial_data[cols]

            for i in["engage_ix", "change_ix", "time_phase1", "time_phase2", "filename", "exp_type"]:
                trial_data[i] = None
                trial_data[i].loc[0] = eval(i)

            for i in["x", "y", "angle", "degs"]:
                trial_data[i] = None
                trial_data[i] = trial_data[i].astype(object)
                trial_data[i].loc[0] = eval(i)

            pp_trials.append(trial_data)
            
    path = ["/"] + p_split[1:-2] + ["{0}_all_data.pkl".format(p_split[-3])]
    path = op.join(*path)

    pp_trials = pd.concat(pp_trials, axis=0, ignore_index=True)

    pp_trials.to_pickle(path)

    path = ["/"] + p_split[1:-2] + ["{0}_results_data.csv".format(p_split[-3])]
    path = op.join(*path)

    selection = [
        'clockwise', 'conditions', 'run', 'session', 'subject', 'trial', 
        "engage_ix", "change_ix", "time_phase1", "time_phase2", "filename", 
        "exp_type"
    ]
    pp_trials[selection].to_csv(path, index_label=False)

    print("END")

