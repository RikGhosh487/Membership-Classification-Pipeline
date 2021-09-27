#!/usr/bin/env python

"""
    Uses Density Based Spatial Clustering for Applications with Noise (DBSCAN) algorithm to pinpoint
    largest cluster based on the epsilon and min samples supplied. If there are min samples data points
    within an epsilon radius from a given datapoint, that group is taken to be a cluster. The algorithm
    proceeds recursively until it has picked out all the clusters it could find. Any remaining points are
    treated as noise in the data, and get rejected.
"""

# imports
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from kneed import KneeLocator

__author__ = "Rik Ghosh"
__copyright__ = "Copyright 2021, The University of Texas at Austin"
__credits__ = ["Soham Saha", "Larissa Franco"]
__license__ = "MIT"
__version__ = "1.1.5"
__maintainer__ = "Rik Ghosh"
__email__ = "rikghosh487@gmail.com"
__status__ = "Production"


def dbscan_reduction(df, param1, param2, n_members, verbose=0):
    """
    Selects the specified fields from the supplied pandas DataFrame and standardizes the data to prepare for    elbow determination, followed by DBSCAN. Depending on the verbosity, it can also create a matplotlib.pyplot scatterplot for the result of the DBSCAN reduction compared to the original supplied data. The elbow determination algorithm is used to find the most optimal radius (eps) for the DBSCAN algorithm, making the entire function dependent on only a single parameter (n_members) instead of two. Depending on the verbosity, the algorithm can also print plots for elbow determination, and specify debugging information for both DBSCAN and elbow determination steps\n
    Args:
    ► `df` (pandas.core.frame.DataFrame): The pandas dataframe that contains the dataset\n
    ► `param1` (str): An attribute present in the `data` parameter that will be processed in the algorithm. This `param1` needs to be the x-axis component for a bivariate dataset\n
    ► `param2` (str): An attribute present in the `data` parameter that will be processed in the algorithm. This `param2` needs to be the y-axis component for a bivariate dataset\n
    ► `n_members` (int): The minimum number of datapoints required for a group to be classified as a cluster by the DBSCAN algorithm. Also used by the elbow estimator to find the optimal eps value. This `n_members` needs to be `n_members > 1`
    ► `verbose` (int, optional): Verbosity for the algorithm. Must be in the range [0,2]. Defaults to `0`\n
        `0` → No scatterplots or debug statements, data returned post processing\n
        `1` → Plots displayed and data returned post processing\n
        `2` → Plots displayed, debug statements printed, and data returned\n
    Returns:
    ► data (pandas.core.frame.DataFrame): the same dataframe `data` after the required restrictions have been enforced via DBSCAN reduction\n
    Raises:
        `KeyError` → invalid parameters `param1` or `param2`\n
        `ValueError` → `n_members <= 1`\n
        `ValueError` → `verbosity != 0` or `1` or `2`\n
    """
    # scope constants
    x, y = None, None
    acceptable_verbose = [0, 1, 2]

    # prerequirement checks
    try:
        x = np.array(df[param1])    # obtain data for specified field name
        y = np.array(df[param2])    # obtain data for specified field name
    except KeyError:
        print(f"The parameters {param1} and {param2} must be present in the dataframe. Please provide valid parameters")
    if n_members <= 1:
        raise ValueError(f"Must have at least 2 n_members.\nValue provided: {n_members}")
    if verbose not in acceptable_verbose:
        raise ValueError(f"Verbosity can only be 0, 1, or 2.\nValue provided: {verbose}")
    
    xy = np.vstack([x, y]).T
    X = StandardScaler().fit_transform(xy)          # standardize data
    eps = __elbow(X, n_members, verbose=verbose)    # esimate best epsilon value
    db = DBSCAN(eps=eps, min_samples=n_members).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    c_labels = db.labels_
    if verbose != 0:
        plt.scatter(x, y, c="indigo", marker=".", label="Original Data")
        plt.xlabel(r"$\mu_{\alpha*} cos(\delta)$" + " (mas/yr)")
        plt.ylabel(r"$\mu_{\delta}$" + " (mas/yr)")
        plt.title("Vector Point Diagram with DBSCAN selection")
        plt.scatter(x[c_labels != -1], y[c_labels != -1], marker=".", c="salmon", label="DBSCAN selection")
        plt.legend(loc="best")
        plt.show()
        if verbose == 2:
            n_clusters = len(set(c_labels)) - (1 if -1 in c_labels else 0)
            n_noise = list(c_labels).count(-1)
            print(f"Estimated number of clusters: {n_clusters}")
            print(f"Estimated number of noise data points: {n_noise}")
    new_param1 = x[c_labels != -1]      # ignore noise data points
    df = df[df[param1].isin(new_param1)]
    if verbose == 2:
        print(f"Reduced dataframe size: {len(df)}")
    return df

def __elbow(x, sample_size, verbose=0):
    """
    Private method used to calculate the most optimal epsilon value for the radius parameter in the DBSCAN algorithm. Using a Knee/Elbow locator, the nearest k neighbors supplied from sample_size are used to
    determine the optimal radius. Depending on verbosity, plots and debug statements may also be produced\n 
    Args:
    ► `x` (np.ndarray): A 2 dimensional standardized numpy ndarray\n 
    ► `sample_size` (int): number of data points required to be considered a cluster\n
    ► `verbose` (int, optional): Verbosity for the algorithm. Must be in the range [0,2]. Defaults to `0`\n
        `0` → No plots or debug statements, data returned post processing\n
        `1` → Plots displayed and data returned post processing\n
        `2` → Plots displayed, debug statements printed, and data returned\n
    Returns:\n
    ► Y value (float) for the location of the elbow computed by the algorithm 
    """
    
    nei = NearestNeighbors(n_neighbors=sample_size) # compute nearest neighbors
    nei_fit = nei.fit(x)
    distances, _ = nei_fit.kneighbors(x)
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]
    i = np.arange(len(distances))
    kneedle = KneeLocator(i, distances, S=1, curve="convex", direction="increasing")    # calculate elbow
    if verbose != 0:
        plt.plot(distances)
        plt.axvline(kneedle.knee, color="crimson", linestyle="--", label="Elbow")
        plt.legend(loc="best")
        plt.xlabel("Datapoints")
        plt.ylabel(r"$\epsilon$")
        plt.title("Elbow Estimation for DBSCAN")
        plt.show()
        if verbose == 2:
            print(f"Best EPS for a sample size of {sample_size} = {kneedle.knee_y}")
    return kneedle.knee_y
