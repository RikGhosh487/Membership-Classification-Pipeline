#!/usr/bin/env python

"""[summary]
"""

# imports
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from kneed import KneeLocator

def dbscan_reduction(df, param1, param2, n_members, verbose=0):
    x = np.array(df[param1])
    y = np.array(df[param2])
    xy = np.vstack([x, y]).T
    X = StandardScaler().fit_transform(xy)
    eps = __elbow(X, n_members, verbose=verbose)
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
    new_param1 = x[c_labels != -1]
    df = df[df[param1].isin(new_param1)]
    if verbose == 2:
        print(len(df))
    return df

def __elbow(x, sample_size, verbose=0):
    nei = NearestNeighbors(n_neighbors=sample_size)
    nei_fit = nei.fit(x)
    distances, _ = nei_fit.kneighbors(x)
    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]
    i = np.arange(len(distances))
    kneedle = KneeLocator(i, distances, S=1, curve="convex", direction="increasing")
    if verbose != 0:
        plt.plot(distances)
        plt.axvline(kneedle.knee, color="crimson", linestyle="--", label="Elbow")
        plt.legend(loc="best")
        plt.xlabel("Datapoints")
        plt.ylabel(r"$\epsilon$")
        plt.title("Elbow Estimation for DBSCAN")
        plt.show()
    return kneedle.knee_y