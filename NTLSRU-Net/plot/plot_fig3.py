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

plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 14

ROOTPATH = "."

def plot_pointHist():
    num1 = 10
    num2 = 60
    viirsname = 'VIIRS'
    oursname = 'SVNL'
    color_palette = sns.color_palette("colorblind")
    datacols = ['VNL','UNet_labelNoProcess_12_13_1571']
    labelcols = [viirsname,oursname]
    texts = ['(a1)','(a2)','(a3)','(b1)','(b2)','(b3)']
    fig,axs = plt.subplots(2,3,figsize=(12,8))
    for i in range(0,2):
        if i == 0:
            year = 2012
        else:
            year = 2013
        path = ROOTPATH + r'\data\\'+str(year) + '_point_prj_adjusted.csv'
        # path = r'D:\04study\00Paper\Eassy\02\analysis\diffRangeAssessment\\' + str(year) + '_point_prj_adjusted.csv'
        df = pd.read_csv(path)
        subdf = df[datacols]
        subdf.columns = labelcols
        for j in range(0,3):
            if j==0:
                snum,enum = 0,num1
            elif j==1:
                snum,enum = num1,num2
            else:
                snum,enum = num2,np.inf
            ax = axs[i,j]
            for col in labelcols:
                data = subdf[col]
                data1 = data[(data < enum) & (data >= snum)]
                if viirsname in col:
                    ax.hist(data1, bins=60, label=str(year)+' '+col,color=color_palette[0], histtype="stepfilled")
                elif oursname in col:
                    ax.hist(data1, bins=60, label=str(year)+' '+col,color=color_palette[1], histtype="step", linewidth=2)
            ax.legend()
            ax.set_xlabel(r'NTL radiance $(\times 10^{-9} W cm^{-2} sr^{-1})$')
            ax.set_ylabel('Count')
            if j==2:
                bins = [i for i in range(60, 550, 120)]
                ax.set_xticks(bins)
                ax.set_xticklabels(bins)
            elif j == 0:
                bins = [i for i in range(0, 11, 2)]
                ax.set_xticks(bins)
                ax.set_xticklabels(bins)
            elif j==1:
                bins = [i for i in range(10, 61, 10)]
                ax.set_xticks(bins)
                ax.set_xticklabels(bins)
            ax.text(0.1, 0.95, texts[i*3+j], transform=ax.transAxes, va='top', ha='left',family='Arial', size=14,weight='bold')
    plt.subplots_adjust(left=0.1, bottom=0.1, top=0.95,right=0.95, hspace=0.3, wspace=0.33)
    outputFullPath = ROOTPATH + r'\data\fig3-2.png'
    plt.savefig(outputFullPath, dpi=300, bbox_inches='tight')

plot_pointHist()