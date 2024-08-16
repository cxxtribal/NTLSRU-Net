import os
import numpy as np
import pandas as pd
import gc
from sklearn import metrics
import math
import pickle
import argparse
from sklearn.cluster import KMeans

import seaborn as sns
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from scipy import stats
from matplotlib.colors import Normalize
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import math

ROOTPATH = "."
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 14

def plot_profiles(profilespath, long_or_lat,title):
    df = pd.read_csv(profilespath)
    df = df.sort_values('pid')

    pidlst = df[long_or_lat].values
    xlabel_min = pidlst.min() - 0.05
    xlabel_max = pidlst.max() + 0.05
    fig = plt.figure(figsize=(6,4))
    ax = plt.subplot(1,1,1)
    ax.plot(df[long_or_lat], df['DMSP'], label='DMSP', color='grey', linewidth=1, linestyle="dotted")
    ax.plot(df[long_or_lat], df['ENL'], label='ChenVNL', color='green', linewidth=1)
    ax.plot(df[long_or_lat], df['VNL'], label='VIIRS', color='darkorange', linewidth=1)
    ax.plot(df[long_or_lat], df['srDNL'], label='SVNL', color='darkblue', linewidth=2)

    ylim = 120
    if "Los" in profilespath:
        ylim = 250
    ax.set_ylim(-10, ylim)
    ax.set_xlim(xlabel_min, xlabel_max)

    if long_or_lat == "long":
        xlabel = "Longitude (\u00b0)"
    else:
        xlabel = "Latitude (\u00b0)"
    ax.set_xlabel(xlabel)
    ax.set_ylabel('NTL intensity')

    plt.legend(loc='upper right',ncol=1, fontsize='14',prop={'family': 'Arial'})
    fig.tight_layout()

    if "Beijing" in profilespath:
        title = "BJ" + title
    else:
        title = "LA" + title
    plt.savefig(ROOTPATH+'\\'+title+".png", dpi=300, bbox_inches='tight')
    plt.close()


plot_profiles(ROOTPATH+"/data-profile/Beijing/2012_l2.csv", "long",'L2') #   Beijing
plot_profiles(ROOTPATH+"/data-profile/Beijing/2012_l1.csv", "lat",'L1')
plot_profiles(ROOTPATH+"/data-profile/LosAngeles/2012_l2.csv", "long",'L4') #   LosAngeles
plot_profiles(ROOTPATH+"/data-profile/LosAngeles/2012_l1.csv", "lat",'L3')