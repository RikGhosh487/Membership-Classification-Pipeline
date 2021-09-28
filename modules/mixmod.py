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
    # MIN_DIST = 0.362 * dist_est + 174
    MIN_DIST = 600
    MAX_DIST = 800
    # MAX_DIST = 2.06 * dist_est - 257

    # obtain distance information
    df = __get_distance(df, plist[0], plist[1], center_x, center_y)
    plist.append(ATTR1)
    
    # make required dataset
    test_frame = df[plist]
    test_frame = test_frame[test_frame[ATTR1] <= 1]
    test_frame = test_frame[1000 / test_frame[plist[-2]] <= MAX_DIST]
    test_frame = test_frame[1000 / test_frame[plist[-2]] >= MIN_DIST]
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
        print("Number of 0s:", count)
        plt.scatter(X[2], X[3], marker=".", c=labels, cmap="viridis")
        plt.xlabel("Proper Motion in Right Ascension " + r"$\mu_{\alpha*}$")
        plt.ylabel("Proper Motion in Declination " + r"$\mu_{\delta}$")
        plt.title("Scatterplot separation based on Membership")
        plt.show()

    probs = gmm.predict_proba(X)
    l1 = probs[:, 0]
    l2 = probs[:, 1]

    mem1 = list(filter(lambda prob: prob >= 0.5, l1))
    mem2 = list(filter(lambda prob: prob >= 0.5, l2))
    count1, count2 = len(mem1), len(mem2)

    hist1, _ = np.histogram(l1, 100)
    hist2, _ = np.histogram(l2, 100)    
    stdev1 = np.std(hist1[-5:])
    stdev2 = np.std(hist2[-5:])

    if verbose != 0:
        print("Probability >= 0.5 for set 1:", count1)
        print("Probability >= 0.5 for set 2:", count2)
    if abs(count1 - count2) > 1000:
        probs = l2 if count1 > count2 else l1
    else:
        probs = l2 if count1 > count2 and stdev1 < stdev2 else l1
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
