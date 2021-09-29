#!/usr/bin/env python

"""
[summary]
"""

# imports
import pandas as pd
import configparser

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
    ASTRO_ERR_LIST = [x + "_error" for x in GMM_LIST]
    BINSIZE = 100

    # reading data from INI file
    read_config = configparser.ConfigParser()
    read_config.read(CONFIG_FILE)

    filename = read_config.get(SECTION, "filename")
    centerx = float(read_config.get(SECTION, "x-center"))
    centery = float(read_config.get(SECTION, "y-center"))
    distance = float(read_config.get(SECTION, "distance"))
    verbosity = int(read_config.get(SECTION, "verbosity"))
    gamma_thresh = float(read_config.get(SECTION, "gamma-thresh"))
    two_tail_factor = float(read_config.get(SECTION, "two-tail-factor"))
    
    # reading file for specified location
    dataframe = pd.read_csv(filename)
    if verbosity > 0:
        print(f"Raw datafile: {len(dataframe)} datapoints")
    
    # check output to consider DR2 data removal
    if TO_REMOVE in dataframe.columns:
        temp = dataframe.dropna()
        if verbosity != 0:
            print(f"DROPNA with radial velocity: {len(temp)} datapoints")    # debug
        dataframe.drop([TO_REMOVE, "dr2_radial_velocity_error"], axis="columns", inplace=True)
    
    # remove rows with missing data
    dataframe.dropna(inplace=True)
    if verbosity != 0:
        print(f"DROPNA: {len(dataframe)} datapoints")   # debug

    for err_name in ASTRO_ERR_LIST:
        dataframe = ger.remove_error(dataframe, err_name, BINSIZE, gamma_thresh, verbosity)
    if verbosity != 0:
        print(f"Gamma Error reduction: {len(dataframe)} datapoints")   # debug

    # Chopping Parallax to prepare for GMM
    dataframe = dataframe[(dataframe["pmra"] >= -40) & (dataframe["pmra"] <= 40)]
    dataframe = dataframe[(dataframe["pmdec"] >= -40) & (dataframe["pmdec"] <= 40)]
    dataframe = dataframe[dataframe["parallax"] >= 0]
    if verbosity != 0:
        print(f"Proper Motion Reduction: {len(dataframe)} datapoints")  # debug

    # double scratching GMMs
    dataframe = gmod.get_test_dataset(dataframe, GMM_LIST, centerx, centery, distance, verbosity)
    if verbosity != 0:
        print(f"Gaussian Mixture Model Reduction: {len(dataframe)} datapoints")
    
    # determine optimal min_samples parameter
    min_samples = 25 if len(dataframe) > 1000 else int(pow(len(dataframe), 0.4))
    if verbosity == 2:
        print(f"DBSCAN min_samples used: {min_samples}")
    dataframe = db.dbscan_reduction(dataframe, "pmra", "pmdec", min_samples, verbosity)
    if verbosity != 0:
        print(f"DBSCAN Reduction: {len(dataframe)} datapoints")


    for astro in GMM_LIST:
        dataframe = tte.chop_tails(dataframe, astro, BINSIZE, two_tail_factor, verbosity)
    if verbosity != 0:
        print(f"Two-tail Error Reduction: {len(dataframe)} datapoints")   # debug
    
    vis.generate_plots(dataframe)

if __name__=="__main__":
    main()
