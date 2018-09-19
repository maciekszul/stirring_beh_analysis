import argparse
import pandas as pd
import numpy as np
import math
from utilities import files
from scipy.interpolate import interp1d
from scipy import where

"""
Script for data processing and preparation, circular motion joystick task. 
Usage:
python prep_script.py -n 0 -p $INPUT -o $OUTPUT -s 0,1,2,3,4

    -n [index in a list of participants]
    -p [path to the raw data]
    -o [path to the output]
    -s [steps performed e.g. 0,1,2,3,4,5]

export INPUT="~/git/stirring_beh_analysis/data"
export OUTPUT="~/git/stirring_beh_analysis/data"

Author: Maciej Szul, 2018
"""

# command line interface
des = "prep script command line"
parser = argparse.ArgumentParser(description=des)

parser.add_argument("-n", type=int, help="index in a list of participants")
parser.add_argument("-p", type=str, help="path to the raw data")
parser.add_argument("-o", type=str, help="path to the output")
parser.add_argument("-s", type=str, help="steps performed")

args = parser.parse_args()
params = vars(args)
n = params["n"]
p = params["p"]
o = params["o"]
s = params["s"]
s = eval("["+ s +"]")

print(n, p, o, s)

if 0 in s:
    print("step 0")