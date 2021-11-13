#!/usr/bin/env python

"""
Uses Gaussian Mixture Models to separate members of clusters from non members. A series of data reductions are conducted on the supplied dataset to prepare it for Gaussian Mixture Model processing. Subordinate functions take care of clean up operations.
"""

# imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture

__author__ = "Rik Ghosh"
__copyright__ = "Copyright 2021, The University of Texas at Austin"
__credits__ = ["Soham Saha", "Larissa Franco"]
__license__ = "MIT"
__version__ = "3.1.1"
__maintainer__ = "Rik Ghosh"
__email__ = "rikghosh487@gmail.com"
__status__ = "Production"

# globals
ATTR1 = "dist_frm_cent"
ATTR2 = "probability"
ATTR3 = "type"

def compute_gmm(df, plist, center_x, center_y, dist_est, verbose=0):
    """
    Parent function handling all the subordinate functions that compute individual pieces of the GMM process. A series of data reductions are performed before the data set is deemed to be usable for GMM processing\n
    Args:
    ► `df` (pandas.core.frame.DataFrame): The pandas dataframe that contains the dataset\n
    ► `plist` (list): list of attributes to be used to form a subset of the `df` dataframe for GMM processing\n
    ► `center_x` (float): estimated x coordinate of the center of the studying cluster\n
    ► `center_y` (float): esitmated y coordinate of the center of the studying cluster\n
    ► `dist_est` ([type]): distance estimate (in parsecs) of how far the cluster is located\n
    ► `verbose` (int, optional): Verbosity for the algorithm. Must be in the range [0,2]. Defaults to `0`\n
        `0` → No plots or debug statements, data returned post processeing\n
        `1` → plots displayed and data returned post processing\n
        `2` → plots displayed, debug statements printed, and data returned\n
    Returns:\n
    ► df (pandas.core.frame.DataFrame): the same dataframe `df` after the Gaussian Mixture Model has finished processing along with the preparatory reductions enforced on the dataframe before the processing\n
    Raises:\n
        `KeyError` → plist contains paramter not present in the dataframe `df`\n
        `ValueError` → `center_x < 0` or `center_x >= 360`\n
        `ValueError` → `center_y < -90` or `center_y > 90`\n
        `ValueError` → `dist_est < 0`\n
        `ValueError` → `verbosity != 0` or `1` or `2`\n
    """
    #scope constants
    acceptable_verbose = [0, 1, 2]
    max_slope = 1.392
    min_slope = 0.719
    max_intercept = -84.5
    min_intercept = 9.37

    try:
        for elem in plist:
            df[elem]
    except KeyError:
        print("plist contains paramters that are not present in the dataframe. Please provide a valid plist")
    if center_x < 0 or center_x >= 360:
        raise ValueError(f"center_x must be in the range 0 <= center_x < 360.\nValue provided: {center_x}")
    if center_y < -90 or center_y > 90:
        raise ValueError(f"center_y must be in the range -90 <= center_y <= 90.\n Value provided: {center_y}")
    if dist_est < 0:
        raise ValueError(f"Distance cannot be negative.\nValue provided: {dist_est}")
    if verbose not in acceptable_verbose:
        raise ValueError(f"Verbosity can only be 0, 1, or 2.\nValue provided: {verbose}")
    
    # linear interpolation for GMM bounds
    min_dist = min_slope * dist_est + min_intercept
    max_dist = max_slope * dist_est + max_intercept

    # obtain distance information
    df = __get_distance(df, plist[0], plist[1], center_x, center_y)
    plist.append(ATTR1)
    
    # make required dataset
    test_frame = df[plist]
    test_frame = test_frame[test_frame[ATTR1] <= 1]     # distance from center <= 1 deg
    test_frame = test_frame[1000 / test_frame[plist[-2]] <= max_dist]   # max distance bound
    test_frame = test_frame[1000 / test_frame[plist[-2]] >= min_dist]   # min distance bound
    test_frame.drop([ATTR1], axis="columns", inplace=True)
    
    # GMM
    test_frame = __fit_gmm(test_frame, verbose)
    plist.pop()         # removing distance from center for final test set
    test_ra = np.array(test_frame["ra"])
    df = df[df["ra"].isin(test_ra)]
    return df

def __get_distance(df, param1, param2, center_x, center_y):
    """
    Private function that calculates the Eucledian distance of every data point to the supplied center_x and center_y and appends it to the supplied dataframe to be used later\n
    Args:
    ► `df` (pandas.core.frame.DataFrame): the pandas dataframe containing the data \n
    ► `param1` (str): attribute storing the equivalent x coordinate of the dataset\n
    ► `param2` (str): attribute storing the equivalent y coordinate of the dataset\n
    ► `center_x` (float): estimated x coordinate of the center of the studying cluster\n
    ► `center_y` (float): esitmated y coordinate of the center of the studying cluster\n
    Returns:\n
    ► df (pandas.core.frame.DataFrame): the same dataframe `df` with the Eucledian distance of each point to the supplied center x and y coordinates appended\n
    """
    distance = list()
    x = np.array(df[param1])
    y = np.array(df[param2])
    for i in range(len(df)):
        distance.append(math.dist([x[i], y[i]], [center_x, center_y]))  # compute eucledian distance
    df[ATTR1] = distance
    return df
    
def __fit_gmm(test_frame, verbose=0):
    """
    Private function that applies Gaussian Mixture Model processing on the supplied 5-parameter dataset. The data is standardized and then processed through a 2 component Gaussian Mixture to group cluster members and non cluster members. The probability of each data point is calculated. Only data points with 0.99+ probability get selected as members.\n
    Args:
    ► `test_frame` (pandas.core.frame.DataFrame): the pandas dataframe containing the data\n
    ► `verbose` (int, optional): Verbosity for the algorithm. Must be in the range [0,2]. Defaults to `0`\n
        `0` → No histogram or debug statements, data returned post processeing\n
        `1` → Histogram displayed and data returned post processing\n
        `2` → Histogram displayed, debug statements printed, and data returned\n
    Returns:\n
    ► test_frame (pandas.core.frame.DataFrame): the same dataframe `test_frame` that has gone through Gaussian Mixture Model processing and has retained only member datapoints with 0.99+ probability of being a cluster member\n
    """
    # Normalize the data frame
    x = test_frame.values
    x_scaled = MinMaxScaler().fit_transform(x)
    X = pd.DataFrame(x_scaled)

    gmm = GaussianMixture(n_components=2).fit(X)
    labels = gmm.predict(X)
    count = 0
    for elem in labels:
        count += 1 if elem == 0 else 0
    if verbose != 0:
        plt.scatter(X[2], X[3], marker=".", c=labels, cmap="viridis")
        plt.xlabel("Proper Motion in Right Ascension " + r"$\mu_{\alpha*}$")
        plt.ylabel("Proper Motion in Declination " + r"$\mu_{\delta}$")
        plt.title("Scatterplot separation based on Membership")
        plt.show()
    if verbose == 2:
        print("Number of 0s:", count)

    probs = gmm.predict_proba(X)
    l1 = probs[:, 0]
    l2 = probs[:, 1]

    mem1 = list(filter(lambda prob: prob >= 0.5, l1))
    mem2 = list(filter(lambda prob: prob >= 0.5, l2))
    cache = len(list(filter(lambda prob: 0.4 <= prob <= 0.6, l1)))
    count1, count2 = len(mem1), len(mem2)

    hist1, _ = np.histogram(l1, 100)
    hist2, _ = np.histogram(l2, 100)
    stdev1 = np.std(hist1[-5:])
    stdev2 = np.std(hist2[-5:])

    if verbose == 2:
        print("Probability >= 0.5 for set 1:", count1)
        print("Standard Deviation for set 1:", stdev1)
        print("Probability >= 0.5 for set 2:", count2)
        print("Standard Deviation for set 2:", stdev2)
    if cache < .05 * len(l1) or hist1[0] < .1 * hist2[0] or hist2[0] < .1 * hist1[0]:
        probs = l2 if count1 > count2 else l1
    else:
        probs = l2 if stdev1 < stdev2 else l1
    test_frame[ATTR2] = probs
    if verbose != 0:
        _, bins, _ = plt.hist(probs, 100, histtype="step")
        plt.xlabel("Probability of being a Cluster Member " + r"$N_{mem}$")
        plt.ylabel("Counts")
        plt.title("Distribution of Cluster Membership probability")
        plt.show()
    else:
        _, bins = np.histogram(probs, 100)

    # debug stats
    test_frame = test_frame[test_frame[ATTR2] >= bins[-2]]
    test_frame = test_frame.drop([ATTR2], axis="columns")
    return test_frame