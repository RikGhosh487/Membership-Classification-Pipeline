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

def get_test_dataset(df, plist, center_x, center_y):
    # constants
    MIN_DIST = 500
    MAX_DIST = 1600

    # obtain distance information
    df = __get_distance(df, plist[0], plist[1], center_x, center_y)
    plist.append(ATTR1)
    
    # make required dataset
    test_frame = df[plist]
    test_frame = test_frame[1000 / test_frame[plist[-2]] <= MAX_DIST]
    test_frame = test_frame[1000 / test_frame[plist[-2]] >= MIN_DIST]
    test_frame.drop([ATTR1], axis="columns", inplace=True)
    
    # GMM
    test_frame = __fit_gmm(test_frame)
    plist.pop()     # removing distance from center for final test set
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
    
def __fit_gmm(test_frame):
    # Normalize the data frame
    x = test_frame.values
    x_scaled = MinMaxScaler().fit_transform(x)
    X = pd.DataFrame(x_scaled)

    gmm = GaussianMixture(n_components=2).fit(X)
    labels = gmm.predict(X)

    plt.scatter(X[2], X[3], marker=".", c=labels, cmap="viridis")
    plt.legend(["member", "field"])
    plt.xlabel("Proper Motion in Right Ascension " + r"$\mu_{\alpha*}$")
    plt.ylabel("Proper Motion in Declination " + r"$\mu_{\delta}$")
    plt.title("Scatterplot separation based on Membership")
    plt.show()

    probs = gmm.predict_proba(X)[:, 0]
    test_frame[ATTR2] = probs
    hist, bins, _ = plt.hist(probs, 100, histtype="step")
    plt.xlabel("Probability of being a Cluster Member " + r"$N_{mem}$")
    plt.ylabel("Counts")
    plt.title("Distribution of Cluster Membership probability")
    plt.show()

    # debug stats
    if hist[1] > hist[-2]:
        test_frame = test_frame[test_frame[ATTR2] >= bins[-2]]
    else:
        test_frame = test_frame[test_frame[ATTR2] <= bins[1]]
    test_frame = test_frame.drop([ATTR2], axis="columns")
    return test_frame
