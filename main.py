#!/usr/bin/env python

"""
A generalized open stellar cluster membership classification algorithm that distinguishes the members of a cluster (if present) from the non-member background and foreground field objects.
"""

# imports
import pandas as pd
import configparser

# custom module imports
from modules import two_tail_err as tte
from modules import msf_dist as dist
from modules import gamma_err as ger
from modules import visualize as vis
from modules import mixmod as gmod
from modules import db

__author__ = "Rik Ghosh"
__copyright__ = "Copyright 2021, The University of Texas at Austin"
__credits__ = ["Soham Saha", "Larissa Franco"]
__license__ = "MIT"
__version__ = "2.4.1"
__maintainer__ = "Rik Ghosh"
__email__ = "rikghosh487@gmail.com"
__status__ = "Production"

def main():
    """
    The main function that follows the membership classification pipeline to process a data set supplied by the user with some preliminary information about the cluster being studied. The function first extracts the user supplied information from the `inputs.ini` file and then proceeds with the algorithm.\n
    Raises:\n
        `ValueError` → Incorrect input in the `inputs.ini` file
    """
    # constants
    CONFIG_FILE = r"inputs.ini"
    SECTION = r"Pipeline Inputs"
    TO_REMOVE = r"dr2_radial_velocity"
    GMM_LIST = ["ra", "dec", "pmra", "pmdec", "parallax"]
    ASTRO_ERR_LIST = [x + "_error" for x in GMM_LIST]
    BINSIZE = 100
    ACCEPTABLE_VERBOSE = [0, 1, 2]

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

    # prerequirement checks
    if verbosity not in ACCEPTABLE_VERBOSE:
        raise ValueError(f"Verbosity can only be 0, 1, or 2.\nValue provided: {verbosity}")
    if gamma_thresh < 0 or gamma_thresh > 1:
        raise ValueError("Gamma Threshold is a probability and must be in the range 0 <= gamma_thresh <= 1."
                + f"\nValue provided: {gamma_thresh}")
    if two_tail_factor < 0:
        raise ValueError(f"Two tail factor must be positive.\nValue provided: {two_tail_factor}")
    
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

    # Gamma reductions for 5 astrometric error parameters
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

    # GMM processing
    dataframe = gmod.compute_gmm(dataframe, GMM_LIST, centerx, centery, distance, verbosity)
    if verbosity != 0:
        print(f"Gaussian Mixture Model Reduction: {len(dataframe)} datapoints")
    
    # determine optimal min_samples parameter
    min_samples = 25 if len(dataframe) > 1000 else int(pow(len(dataframe), 0.4))
    if verbosity == 2:
        print(f"DBSCAN min_samples used: {min_samples}")
    dataframe = db.dbscan_reduction(dataframe, "pmra", "pmdec", min_samples, verbosity) # DBSCAN
    if verbosity != 0:
        print(f"DBSCAN Reduction: {len(dataframe)} datapoints")

    # Two tail error reduction of 5 astrometric parameters
    for astro in GMM_LIST:
        dataframe = tte.chop_tails(dataframe, astro, BINSIZE, two_tail_factor, verbosity)
    if verbosity != 0:
        print(f"Two-tail Error Reduction: {len(dataframe)} datapoints")   # debug
    
    if verbosity != 0:
        vis.generate_plots(dataframe)

    # output file
    dist.msf(dataframe, 0)
    dataframe.to_csv(f"{filename[:-4]}_finalized.csv", index=False)

if __name__=="__main__":
    main()
