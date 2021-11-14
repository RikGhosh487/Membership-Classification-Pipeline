#!/usr/bin/env python

# imports
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import gaussian_kde

ATTR1 = "dist"
XLABEL = r"Right Ascension ($\alpha$ deg)"
YLABEL = r"Declination ($\delta$ deg)"
DATA1 = "SDSS data"
DATA2 = "GAIA data"

def photo_metallicity(data, sdss_csv, verbose=0):
    sdss_dataframe = pd.read_csv(sdss_csv)
    if verbose == 2:
        print(f"SDSS datafile: {len(sdss_dataframe)} datapoints")
    
    sdss_dataframe.dropna(inplace=True)
    if verbose == 2:
        print(f"SDSS DROPNA: {len(sdss_dataframe)}")
    
    sdss_dataframe = __reduce_sdss(data, sdss_dataframe, verbose)
    sdss_dataframe.drop(["ra", "dec"], axis="columns", inplace=True)

    segue = pd.read_csv("segue.csv")
    segue.dropna()
    feh = np.array(segue["feh"])
    segue.drop(["logg", "teff", "feh"], axis="columns", inplace=True)

    # photometric estimates
    x_train, x_test, y_train, y_test = train_test_split(segue, feh, test_size=0.2)
    model = RandomForestRegressor(n_estimators=163)
    model.fit(x_train, y_train)         # training the model
    y_pred = model.predict(x_test)      # predicting with test data

    # density based coloring
    if verbose != 0:
        xy = np.vstack([y_test, y_pred])
        z = gaussian_kde(xy)(xy)

        # plotting
        plt.scatter(y_test, y_pred, c=z, marker='.')
        plt.plot(y_test, y_test, 'r-', label="One-to-one Regression Line")
        plt.text(min(y_test), max(y_pred) - 0.25, f"RMSE: {round(mean_squared_error(y_test, y_pred), 4)}") # RMSE
        y_pls = [0.75 + x for x in y_test]     # CPE lines
        plt.plot(y_test, y_pls, 'b--', label='-0.75 dex line')
        plt.plot(y_pls, y_test, 'b--', label='+0.75 dex line')
        diff = abs(y_pred - y_test)
        count = 0
        for elem in diff:
            if elem > 0.7:
                count += 1
        plt.text(min(y_test), max(y_pred) - 2 * 0.25, f'CPER: {round(count / len(y_test), 4)}') # CPER
        plt.xlabel(r'$[Fe/H]_{SSPP}$')
        plt.ylabel(r'$[Fe/H]_{RF}$')
        plt.legend(loc='best')
        plt.title("Machine Learning Truth-to-Prediction Plot")
        plt.show()

    value = model.predict(sdss_dataframe)
    print(f"Cluster Metallicity: {np.mean(value)}")

def __reduce_sdss(data, sdss_frame, verbose):
    if verbose != 0:
        plt.scatter(sdss_frame["ra"], sdss_frame["dec"], marker=".", label=DATA1)
        plt.scatter(data["ra"], data["dec"], marker=".", label=DATA2)
        plt.plot(np.mean(data["ra"]), np.mean(data["dec"]), "r*", label="Mean")
        plt.xlabel(XLABEL)
        plt.ylabel(YLABEL)
        plt.legend(loc="best")
        plt.show()

    xlist = np.array(data["ra"])
    ylist = np.array(data["dec"])
    xcenter, ycenter = np.mean(data["ra"]), np.mean(data["dec"])
    limit = __get_max_dist(xlist, ylist, xcenter, ycenter)

    xs, ys = np.array(sdss_frame["ra"]), np.array(sdss_frame["dec"])

    dists = list()
    for i in range(len(xs)):
        dists.append(__get_dist(xs[i], xcenter, ys[i], ycenter))
        
    sdss_frame[ATTR1] = dists
    sdss_frame = sdss_frame[sdss_frame[ATTR1] <= limit]         # only allow data point smaller than limit
    sdss_frame.drop([ATTR1], axis="columns", inplace=True)

    if verbose != 0:
        plt.scatter(sdss_frame["ra"], sdss_frame["dec"], marker=".", label=DATA1)
        plt.scatter(data["ra"], data["dec"], marker=".", label=DATA2)
        plt.xlabel(XLABEL)
        plt.ylabel(YLABEL)
        plt.legend(loc="best")
        plt.show()

    df_list = list()
    for i in range(len(xlist)):
        temp = sdss_frame[(sdss_frame["ra"] < xlist[i] + 0.01) & (sdss_frame["ra"] > xlist[i] - 0.01)]
        if len(temp) != 0:
            temp = temp[(temp["dec"] < ylist[i] + 0.01) & (temp["dec"] > ylist[i] - 0.01)]
        if len(temp) != 0:
            df_list.append(temp)

    sdss_frame = pd.concat(df_list)

    if verbose != 0:
        plt.scatter(sdss_frame["ra"], sdss_frame["dec"], marker=".", label=DATA1)
        plt.scatter(data["ra"], data["dec"], marker=".", label=DATA2)
        plt.xlabel(XLABEL)
        plt.ylabel(YLABEL)
        plt.legend(loc="best")
        plt.show()

    return sdss_frame
    
def __get_max_dist(xlist, ylist, xcenter, ycenter):
    mdist = -1
    for i in range(len(xlist)):
        dist = __get_dist(xlist[i], xcenter, ylist[i], ycenter)
        if dist > mdist:
            mdist = dist
    
    return mdist


def __get_dist(x1, x2, y1, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

