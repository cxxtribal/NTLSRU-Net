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
plt.rcParams['font.size'] = 16

def plot_2_14_VNL_DNL_annualTimeSeries():
    plt.figure(figsize=(8, 5))
    #VNL
    path = ROOTPATH + r'\data\VNL_set1.txt'
    file_list = tuple(open(path, "r"))
    yearlst = [int(content.strip().split(',')[1]) for content in file_list]
    tnllst = [float(content.strip().split(',')[2]) for content in file_list]
    # plt.plot(yearlst, tnllst,linestyle=':', marker='|', label='VNL')
    plt.plot(yearlst, tnllst, marker='|', label='VIIRS')
    #raw DNL
    path = ROOTPATH + r'\data\raw_DNL_tnl.txt'
    file_list = tuple(open(path, "r"))
    sensorlst = [content.strip().split(',')[0][:3] for content in file_list]
    yearlst = [int(content.strip().split(',')[0][3:]) for content in file_list]
    tnllst = [float(content.strip().split(',')[1]) for content in file_list]
    df = pd.DataFrame(data={'sensor':sensorlst,'year':yearlst,'tnl':tnllst})
    group = df.groupby('sensor')
    for key,value in group:
        # plt.plot(value['year'],value['tnl'],marker='o',c='darkgrey',label='DMSP '+key)
        plt.plot(value['year'],value['tnl'],marker='o',label='DMSP '+key)
    xticklabellst = [str(i) for i in range(1992, 2025,4)]
    xticklst = [i for i in range(1992, 2025, 4)]
    plt.xticks(xticklst,xticklabellst)
    # plt.ylabel('NTL intensity' + r"$(nano-Wcm^{-2}sr^{-1})$")
    # plt.xlabel('Year')
    # plt.ylabel('夜间灯光强度' + r"$(nano-Wcm^{-2}sr^{-1})$")
    plt.ylabel('Total NTL intensity')
    plt.xlabel('Year')
    plt.ylim(1.5e8, 5.5e8)
    # plt.legend(fontsize='medium')
    #plt.title('（a）一致性校正前全球夜间灯光总强度时间序列',y=-0.28)
    # plt.tick_params(labelsize=14)
    plt.subplots_adjust(top=0.950, bottom=0.110, left=0.093, right=0.983)
    plt.legend(loc='upper left', bbox_to_anchor=(0.1, 0.99), ncol=3, fontsize='16',frameon=False)
    outputFullPath = ROOTPATH + r'\VNL_DNL_set1.png'
    plt.savefig(outputFullPath, dpi=300, bbox_inches='tight')
    plt.close()

def plot_2_14_VNL_CNL_srDNL_DNL_HDNL_annualTimeSeries():
    plt.figure(figsize=(8, 5))
    #VNL
    path = ROOTPATH + r'\data\VNL_set1.txt'
    file_list = tuple(open(path, "r"))
    yearlst = [int(content.strip().split(',')[1]) for content in file_list]
    tnllst = [float(content.strip().split(',')[2]) for content in file_list]
    # plt.plot(yearlst, tnllst, marker='o', label='VNL')
    plt.plot(yearlst,tnllst, marker='|', label='VIIRS')
    #srDNL
    path = ROOTPATH + r'\data\UNet_labelNoProcess_12_13_1571_world.txt'
    file_list = tuple(open(path, "r"))
    yearlst = [int(content.strip().split(',')[0][0:4]) for content in file_list]
    tnllst = [float(content.strip().split(',')[1]) for content in file_list]
    plt.plot(yearlst, tnllst, marker='o', label='SVNL')
    #ENL
    path = ROOTPATH + r'\data\CNL.txt'
    file_list = tuple(open(path, "r"))
    yearlst = [int(content.strip().split(',')[0][3:]) for content in file_list]
    tnllst = [float(content.strip().split(',')[1]) for content in file_list]
    plt.plot(yearlst, tnllst,linestyle='--', marker='o', label='ChenVNL')
    #harmonized
    path = ROOTPATH + r'\data\Harmonized_DN_NTL_tnl.txt'
    file_list = tuple(open(path, "r"))
    yearlst = [int(content.strip().split(',')[0][18:22]) for content in file_list]
    tnllst = [float(content.strip().split(',')[1]) for content in file_list]
    plt.plot(yearlst, tnllst, linestyle='--',marker='o', label='LiDNL')
    xticklabellst = [str(i) for i in range(1992, 2025,4)]
    xticklst = [i for i in range(1992, 2025,4)]
    plt.xticks(xticklst,xticklabellst)
    plt.ylim(0,1.2e9)
    # plt.ylabel('夜间灯光强度' + r"$(nano-Wcm^{-2}sr^{-1})$")
    plt.ylabel('Total NTL intensity')
    plt.xlabel('Year')
    # plt.legend(fontsize='medium')
    plt.legend()
    # plt.tick_params(labelsize=14)
    plt.subplots_adjust(top=0.950, bottom=0.110, left=0.093, right=0.983)
    plt.legend(loc='upper left', bbox_to_anchor=(0.1, 0.99), ncol=2, fontsize='16',frameon=False)
    outputFullPath = ROOTPATH + r'\srNL_VNL_ENL_HNL.png'
    plt.savefig(outputFullPath, dpi=300, bbox_inches='tight')
    plt.close()

plot_2_14_VNL_DNL_annualTimeSeries()
plot_2_14_VNL_CNL_srDNL_DNL_HDNL_annualTimeSeries()