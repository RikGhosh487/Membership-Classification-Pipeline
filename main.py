#!/usr/bin/env python

"""
[summary]
"""

# imports
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import configparser
import numpy as np
import math

# custom module imports
from modules import two_tail_err as tte
from modules import gamma_err as ger
from modules import visualize as vis
from modules import mixmod as gmod
from modules import db

"""
[area for file details]
"""

def main():
    """
    [method summary]
    """
    # constants
    CONFIG_FILE = r"inputs.ini"
    SECTION = r"Pipeline Inputs"
    TO_REMOVE = r"dr2_radial_velocity"
    GMM_LIST = ["ra", "dec", "pmra", "pmdec", "parallax"]
    BINSIZE = 100

    # reading data from INI file
    read_config = configparser.ConfigParser()
    read_config.read(CONFIG_FILE)

    filename = read_config.get(SECTION, "filename")
    centerx = float(read_config.get(SECTION, "x-center"))
    centery = float(read_config.get(SECTION, "y-center"))
    verbosity = int(read_config.get(SECTION, "verbosity"))
    gamma_thresh = float(read_config.get(SECTION, "gamma-thresh"))
    two_tail_factor = float(read_config.get(SECTION, "two-tail-factor"))
    
    # reading file for specified location
    dataframe = pd.read_csv(filename)
    print(f"Raw datafile: {len(dataframe)} datapoints") # debug
    
    # check output to consider DR2 data removal
    if TO_REMOVE in dataframe.columns:
        temp = dataframe.dropna()
        print(f"DROPNA with radial velocity: {len(temp)} datapoints")    # debug
        dataframe.drop([TO_REMOVE, "dr2_radial_velocity_error"], axis="columns", inplace=True)
    
    # remove rows with missing data
    dataframe.dropna(inplace=True)
    print(f"DROPNA: {len(dataframe)} datapoints")   # debug

    dataframe = ger.remove_error(dataframe, "ra_error", BINSIZE, gamma_thresh, verbosity)
    dataframe = ger.remove_error(dataframe, "dec_error", BINSIZE, gamma_thresh, verbosity)
    dataframe = ger.remove_error(dataframe, "pmra_error", BINSIZE, gamma_thresh, verbosity)
    dataframe = ger.remove_error(dataframe, "pmdec_error", BINSIZE, gamma_thresh, verbosity)
    dataframe = ger.remove_error(dataframe, "parallax_error", BINSIZE, gamma_thresh, verbosity)
    print(f"Gamma Error reduction: {len(dataframe)} datapoints")   # debug

    # Chopping Parallax to prepare for GMM
    dataframe = dataframe[dataframe["parallax"] >= 0]
    dataframe = dataframe[(dataframe["pmra"] >= -30) & (dataframe["pmra"] <= 30)]
    dataframe = dataframe[(dataframe["pmdec"] >= -30) & (dataframe["pmdec"] <= 30)]
    print(f"Parallax Reduction: {len(dataframe)}")  # debug

    # double scratching GMMs
    dataframe = gmod.get_test_dataset(dataframe, GMM_LIST, centerx, centery)
    print(f"Gaussian Mixture Model Reduction #1: {len(dataframe)}")
    dataframe = db.dbscan_reduction(dataframe, "pmra", "pmdec", 20, verbose=1)
    print(f"DBSCAN Reduction: {len(dataframe)}")
    dataframe = gmod.get_test_dataset(dataframe, GMM_LIST, centerx, centery)
    print(f"Gaussian Mixture Model Reduction #2: {len(dataframe)}")

    dataframe = tte.chop_tails(dataframe, "pmra", BINSIZE//4, two_tail_factor, verbosity)
    dataframe = tte.chop_tails(dataframe, "pmdec", BINSIZE//4, two_tail_factor, verbosity)
    dataframe = tte.chop_tails(dataframe, "parallax", BINSIZE//4, two_tail_factor, verbosity)
    print(f"Two-tail Error Reduction: {len(dataframe)} datapoints")   # debug
    
    vis.generate_plots(dataframe)

if __name__=="__main__":
    main()
