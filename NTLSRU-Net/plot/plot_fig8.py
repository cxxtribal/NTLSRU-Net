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

from matplotlib import rcParams

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 14

ROOTPATH = r'D:\jianguoyun\paper\论文\03NPP-VIIRS-like\plot\pic8'

def plot_lat():
    lat_path = ROOTPATH+r'\latitude.csv'
    data = pd.read_csv(lat_path)
    data['lat'] = data['lat']*-1
    selected_data = data[['lat', '1992', '2002', '2012', '2022']]
    colors = ['#BABABA', '#0001A1', '#037F77', '#C5272D']
    fig, ax = plt.subplots(figsize=(4, 1.5))
    for i, column in enumerate(selected_data.columns[1:]):
        ax.plot(selected_data['lat'], selected_data[column], label=column,color=colors[i])
    ax.axis('off')
    fig.tight_layout()
    outpath = ROOTPATH+r'\lat1.png'
    plt.savefig(outpath,dpi=300,bbox_inches='tight')
    plt.close()


def plot_lon():
    lon_path = ROOTPATH+r'\longitude.csv'
    data = pd.read_csv(lon_path)
    selected_data = data[['lon', '1992', '2002', '2012', '2022']]
    colors = ['#BABABA', '#0001A1', '#037F77', '#C5272D']
    fig, ax = plt.subplots(figsize=(9.7, 1.5))
    for i, column in enumerate(selected_data.columns[1:]):
        ax.plot(selected_data['lon'], selected_data[column], label=column,color=colors[i])
    ax.axis('off')
    fig.tight_layout()
    outpath = ROOTPATH+r'\lon.png'
    plt.savefig(outpath,dpi=300,bbox_inches='tight')
    plt.close()

plot_lat()
plot_lon()