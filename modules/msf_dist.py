#!/usr/bin/env python

"""
Applies Main Sequence Fitting on the target cluster after processing through the classification pipeline. Uses the Hyades Cluster as a standard cluster to apply Main Sequence Fitting, and computes the distance to the target cluster from Earth in parsecs using the computed distance modulus.
"""

# imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

__author__ = "Rik Ghosh"
__copyright__ = "Copyright 2021, The University of Texas at Austin"
__credits__ = ["Soham Saha", "Larissa Franco"]
__license__ = "MIT"
__version__ = "1.0.1"
__maintainer__ = "Rik Ghosh"
__email__ = "rikghosh487@gmail.com"
__status__ = "Production"

def msf(dataframe, verbose=0):
    # scope variables
    standard_filename = "hyades_absolute.csv"

    # dataframe = __main_sequence_isolator(dataframe)         # isolate the main sequence line
    df = pd.read_csv(standard_filename)

    # shift to the turn off point of the standard
    dataframe["g_rp"] = dataframe["g_rp"] + (min(df["b_v"]) - min(dataframe["g_rp"]))

    if verbose != 0:
        _, bins, _ = plt.hist(df["b_v"], histtype="step", bins=23)
        plt.show()
    else:
        _, bins = np.histogram(df["b_v"], bins=23)

    plt.gca().invert_yaxis()
    plt.scatter(dataframe["g_rp"], dataframe["phot_g_mean_mag"], marker=".")
    plt.scatter(df["b_v"], df["v_abs"], marker=".")
    plt.show()

    dist = list()
    for i in range(len(bins) - 1):
        temp1 = dataframe[(dataframe["g_rp"] >= bins[i]) & (dataframe["g_rp"] < bins[i + 1])]
        temp2 = df[(df["b_v"] >= bins[i]) & (df["b_v"] < bins[i + 1])]
        if(len(temp1) != 0 and len(temp2) != 0):
            dist.append(__get_mod(temp1, temp2))

    dist.sort()
    dist.pop(1)
    dist.pop(1)
    dist.pop(1)
    print(f"Average Distance = {sum(dist) / len(dist)}")

def __main_sequence_isolator(dataframe):
    dataframe = dataframe[dataframe["phot_g_mean_mag"] <= 20]
    refined = dataframe[["phot_g_mean_mag", "g_rp"]]             # only take the photometric information
    hist, xbin, ybin, _ = plt.hist2d(refined["g_rp"], refined["phot_g_mean_mag"], bins=50,
            density=True, cmin=0.45)
    plt.gca().invert_yaxis()
    plt.show()

    # only select the fields that have non-nan values
    reduced = list()
    for i in range(len(hist)):
        for j in range(len(hist[i])):
            if(not np.isnan(hist[i][j])):
                reduced.append([i, j])
    
    dfs = list()
    for elem in reduced:
        x, y = elem[0], elem[1]
        temp = refined[(refined["phot_g_mean_mag"] > ybin[y]) & (refined["phot_g_mean_mag"] < ybin[y + 1])]
        temp = temp[(temp["g_rp"] > xbin[x]) & (temp["g_rp"] < xbin[x + 1])]
        dfs.append(temp)            # filter dataframes to map only composite boxes

    df = pd.concat(dfs)             # combine dataframes
    return df

def __get_mod(target_df, standard_df):
    m1 = np.median(target_df["phot_g_mean_mag"])
    m2 = np.median(standard_df["v_abs"])
    return math.pow(10, 0.2 * (m1 - m2) + 1)
