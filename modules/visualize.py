#!/usr/bin/env python

"""
Helps visualize all the resulting data from the pipeline processing. Several plots are generated
from the finalized data set that display all the important attributes.
"""

# imports
import matplotlib.pyplot as plt
import seaborn as sns

__author__ = "Rik Ghosh"
__copyright__ = "Copyright 2021, The University of Texas at Austin"
__credits__ = ["Soham Saha", "Larissa Franco"]
__license__ = "MIT"
__version__ = "1.9.2"
__maintainer__ = "Rik Ghosh"
__email__ = "rikghosh487@gmail.com"
__status__ = "Production"

def generate_plots(df):
    """
    Produces Pair plots, Histograms and Scatterplots for particular subsets of data from the passed in
    pandas dataframe. This module is used to visualize the final dataframe that is generated post processing
    in the pipeline\n 
    Args:
    ► `df` (pandas.core.frame.DataFrame): The pandas dataframe that contains the subset data which will be used to generate pair plots, scatterplots and histograms\n
    Raises:\n
        `KeyError` → absent parameter in `df`\n
    """
    # scope constants
    ra_dec, pmra_pmdec, parallax_pm, colors = None, None, None, None

    # organizing components of dataframe
    try:
        ra_dec = df[["ra", "dec"]]
        pmra_pmdec = df[["pmra", "pmdec"]]
        parallax_pm = df[["parallax", "pm"]]
        colors = df[["phot_g_mean_mag", "bp_g", "bp_rp", "g_rp"]]
    except KeyError:
        print("Dataframe must contain the following fields:\nra, dec, pmra, pmdec, parallax, pm," 
                + " phot_g_mean_mag, bp_g, bp_rp, g_rp")
    
    __grid_plots(ra_dec, ["royalblue", "steelblue", "deepskyblue"])
    __grid_plots(pmra_pmdec, ["sienna", "chocolate", "sandybrown"])

    fig, ax = plt.subplots(1, 2)
    fig.suptitle("Parallax and Proper Motion")
    sns.histplot(data=parallax_pm, x="parallax", element="step", fill=False, ax=ax[0], color="yellowgreen")
    sns.histplot(data=parallax_pm, x="pm", element="step", fill=False, ax=ax[1], color="slateblue")
    plt.show()

    fig, ax = plt.subplots(1, 3, sharey=True)
    fig.suptitle("Color Magnitude Diagrams")
    plt.gca().invert_yaxis()
    # CMD for BP - G
    ax[0].scatter(data=colors, x="bp_g", y="phot_g_mean_mag", marker=".", c="indianred")
    ax[0].set_title(r"$G_{mag}$ vs $B_P - G$ CMD")
    ax[0].set_xlabel(r"$B_P - G$")
    ax[0].set_ylabel(r"$G_{mag}$")
    # CMD for BP - RP
    ax[1].scatter(data=colors, x="bp_rp", y="phot_g_mean_mag", marker=".", c="plum")
    ax[1].set_title(r"$G_{mag}$ vs $B_P - R_P$ CMD")
    ax[1].set_xlabel(r"$B_P - R_P$")
    # CMD for G - RP
    ax[2].scatter(data=colors, x="g_rp", y="phot_g_mean_mag", marker=".", c="peru")
    ax[2].set_title(r"$G_{mag}$ vs $G - R_P$ CMD")
    ax[2].set_xlabel(r"$G - R_P$")
    plt.show()

def __grid_plots(dataframe, color):
    """
    Private method used to generate grid plots using the `seaborn` module. The generated grid pair plot must be from a bivariate data set. The major diagonals are composed of individual histograms, while the bottom left is occupied by a scatterplot of the bivariate data, and the top right is used for a kernel density plot.\n
    Args:
    ► `dataframe` (pandas.core.frame.DataFrame): the pandas dataframe containing the subset of bivariate data to be used in the pair grid plot.\n
    ► `color` (list): list of matplotlib colors to be used for each subplot in the grid plot. The list must have a length of 3, with the following indices corresponding to the following colors:
        `0` → Kernel Density Plot\n
        `1` → Scatterplot\n
        `2` → Histograms\n
    Raises:\n
        `ValueError` → incorrect list size for `color`\n
    """
    # prerequirement check
    if len(color) != 3:
        raise ValueError(f"List must have exactly 3 elements.\nSupplied list has length = {len(color)}")

    grid = sns.PairGrid(dataframe)
    grid.map_upper(sns.kdeplot, color=color[0])
    grid.map_lower(plt.scatter, marker=".", color=color[1])
    grid.map_diag(sns.histplot, element="step", fill=False, color=color[2])
    plt.show()
