#!/usr/bin/env python

"""[summary]
"""

# imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture

# globals
ATTR1 = "dist_frm_cent"
ATTR2 = "probability"
ATTR3 = "type"

def get_test_dataset(df, plist, center_x, center_y, dist_est, verbose=0):
    # constants
    MAX_SLOPE = 1.392
    MIN_SLOPE = 0.719
    MAX_INTERCEPT = -84.5
    MIN_INTERCEPT = 9.37
    
    # linear interpolation for GMM bounds
    min_dist = MIN_SLOPE * dist_est + MIN_INTERCEPT
    max_dist = MAX_SLOPE * dist_est + MAX_INTERCEPT

    # obtain distance information
    df = __get_distance(df, plist[0], plist[1], center_x, center_y)
    plist.append(ATTR1)
    
    # make required dataset
    test_frame = df[plist]
    test_frame = test_frame[test_frame[ATTR1] <= 1]
    test_frame = test_frame[1000 / test_frame[plist[-2]] <= max_dist]
    test_frame = test_frame[1000 / test_frame[plist[-2]] >= min_dist]
    test_frame.drop([ATTR1], axis="columns", inplace=True)
    
    # GMM
    test_frame = __fit_gmm(test_frame, verbose)
    plist.pop()         # removing distance from center for final test set
    test_ra = np.array(test_frame["ra"])
    df = df[df["ra"].isin(test_ra)]
    return df

def __get_distance(df, param1, param2, center_x, center_y):
    distance = list()
    x = np.array(df[param1])
    y = np.array(df[param2])
    for i in range(len(df)):
        distance.append(math.dist([x[i], y[i]], [center_x, center_y]))
    df[ATTR1] = distance
    return df
    
def __fit_gmm(test_frame, verbose=0):
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
