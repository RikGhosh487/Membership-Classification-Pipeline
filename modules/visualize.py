#!/usr/bin/env python

# imports
import matplotlib.pyplot as plt
import seaborn as sns

"""
[summary]
"""

def generate_plots(df):
    """
    

    Args:
        df (): [description]
    """
    # scope constants
    ra_dec, pmra_pmdec, corrs, parallax_pm = None, None, None, None

    # organizing components of dataframe
    try:
        ra_dec = df[["ra", "dec"]]
        pmra_pmdec = df[["pmra", "pmdec"]]
        corrs = df[["parallax_pmra_corr", "parallax_pmdec_corr", "pmra_pmdec_corr"]]
        parallax_pm = df[["parallax", "pm"]]
    except KeyError:
        print("Dataframe must contain the following fields: ra, dec, pmra, pmdec, parallax_pmra_corr," 
                + " parallax_pmdec_corr, pmra_pmdec_corr, parallax, pm")
    
    __grid_plots(ra_dec)
    __grid_plots(pmra_pmdec)

    fig, ax = plt.subplots(1, 2)
    fig.suptitle("Parallax and Proper Motion")
    sns.histplot(data=parallax_pm, x="parallax", element="step", fill=False, ax=ax[0])
    sns.histplot(data=parallax_pm, x="pm", element="step", fill=False, ax=ax[1])
    plt.show()

    sns.histplot(data=corrs, x="parallax_pmra_corr", element="step", fill=False, legend="parallax pmra")
    sns.histplot(data=corrs, x="parallax_pmdec_corr", element="step", fill=False, legend="parallax pmdec")
    sns.histplot(data=corrs, x="pmra_pmdec_corr", element="step", fill=False, legend="pmra pmdec")
    plt.xlabel("Correlations")
    plt.show()

    plt.gca().invert_yaxis()
    plt.scatter(df["g_rp"], df["phot_g_mean_mag"], marker=".")
    plt.show()

def __grid_plots(dataframe):
    """
    [summary]

    Args:
        dataframe ([type]): [description]
    """
    grid = sns.PairGrid(dataframe)
    grid.map_upper(plt.scatter, marker=".")
    grid.map_lower(sns.kdeplot)
    grid.map_diag(sns.histplot, element="step", fill=False, bins=100)
    plt.show()
