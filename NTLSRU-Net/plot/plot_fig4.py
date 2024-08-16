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

def stats_Param(df, xcol, ycol):
    '''
    计算统计量
    :param df:
    :param xcol:
    :param ycol:
    :return:
    '''
    y_true = df[ycol]
    y_pred = df[xcol]
    rmse = math.sqrt(metrics.mean_squared_error(y_true, y_pred))
    mae = metrics.mean_absolute_error(y_true, y_pred)
    evs = metrics.explained_variance_score(y_true, y_pred)  # 可解释性方差
    r2 = metrics.r2_score(y_true, y_pred)  # r2
    stat_dict = {'RMSE': rmse, 'MAE': mae, 'explained_variance_score': evs, 'R2': r2}
    return stat_dict

class plot_2_7_ScalesScatter_density():
    def func_pixelLevel_figure_allScalesScatter_density_2012_2013(self,year,xcol,ycol,xlabel,ylabel,title):
        dir = ROOTPATH+r'\data'
        path = os.path.join(dir,str(year)+'_point_prj_adjusted.csv')
        df = pd.read_csv(path)
        stat_dict = stats_Param(df, xcol, ycol)
        # ===========Calculate the point density==========
        x = df[xcol]
        y = df[ycol]
        xy = np.vstack([x, y])
        z = stats.gaussian_kde(xy)(xy)
        # ===========Sort the points by density, so that the densest points are plotted last===========
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
        #绘图
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.set_xlim(-5, 400)
        ax.set_ylim(-5,400)
        ax.plot([-5, 400], [-5, 400], 'black', lw=1)  # 画的1:1线，线的颜色为black，线宽为0.8
        h = plt.scatter(x, y, c=z, s=2, edgecolor='', cmap='viridis')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        # 设置文本
        RMSE = stat_dict['RMSE']
        MAE = stat_dict['MAE']
        R2 = stat_dict['R2']
        ax.text(200, 90, '$RMSE=%.0f$' % RMSE,family='Arial', size=14)
        ax.text(200, 55, '$MAE=%.0f$' % MAE,family='Arial', size=14)
        ax.text(200, 20, '$R^2=%.3f$' % R2,family='Arial', size=14)
        # colorbar
        cb = fig.colorbar(h)
        cb.set_ticks([])
        # 设置颜色带刻度
        cb.set_ticks([np.min(z), np.max(z)])
        cb.ax.yaxis.set_ticklabels(['Low', 'High'], fontsize=12,family='Arial')
        cb.ax.set_title('Density', pad=10,x=1.5,fontsize=14,family='Arial')
        plt.subplots_adjust(left=0.180, right=0.975, top=0.900, bottom=0.180, hspace=0.2, wspace=0.2)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        # ax.text(0.1, 0.95, title, transform=ax.transAxes, va='top', ha='left', family='Arial', size=14)
    def func_cityLevel_figure_allScalesScatter_density_2012_2013(self,year,xcol,ycol,xlabel,ylabel,title):
        dir = ROOTPATH+r'\data'
        path = os.path.join(dir,str(year)+'_city_prj_adjusted.csv')
        df = pd.read_csv(path)
        stat_dict = stats_Param(df, xcol, ycol)
        # ===========Calculate the point density==========
        x = df[xcol]
        y = df[ycol]
        xy = np.vstack([x, y])
        z = stats.gaussian_kde(xy)(xy)
        # ===========Sort the points by density, so that the densest points are plotted last===========
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
        fig, ax = plt.subplots(figsize=(4, 3))
        #绘图
        ax.set_xlim(0,300000)
        ax.set_ylim(0,300000)
        ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
        ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='x')
        ax.plot([0, 8e5], [0, 8e5], 'black', lw=1)  # 画的1:1线，线的颜色为black，线宽为0.8
        h = plt.scatter(x, y, c=z, s=2, edgecolor='', cmap='viridis')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        # 设置文本
        RMSE = stat_dict['RMSE']
        MAE = stat_dict['MAE']
        R2 = stat_dict['R2']
        ax.text(1.3e5, 7e4, '$RMSE=%.0f$' % RMSE, family='Arial', size=14)
        ax.text(1.3e5, 4e4, '$MAE=%.0f$' % MAE,family='Arial', size=14)
        ax.text(1.3e5, 1e4, '$R^2=%.3f$' % R2,family='Arial', size=14)
        # colorbar
        cb = fig.colorbar(h)
        cb.set_ticks([])
        # 设置颜色带刻度
        cb.set_ticks([np.min(z), np.max(z)])
        cb.ax.yaxis.set_ticklabels(['Low', 'High'], fontsize=12,family='Arial')
        cb.ax.set_title('Density', pad=10,x=1.5,fontsize=14,family='Arial')
        plt.subplots_adjust(left=0.180, right=0.975, top=0.900, bottom=0.180, hspace=0.2, wspace=0.2)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        # ax.text(0.1, 0.95, title, transform=ax.transAxes, va='top', ha='left', family='Arial', size=14)
    def func_provinceLevel_figure_allScalesScatter_density_2012_2013(self,year,xcol,ycol,xlabel,ylabel,title):
        dir = ROOTPATH+r'\data'
        path = os.path.join(dir,str(year)+'_province_prj_adjusted.csv')
        df = pd.read_csv(path)
        stat_dict = stats_Param(df, xcol, ycol)
        # ===========Calculate the point density==========
        x = df[xcol]
        y = df[ycol]
        xy = np.vstack([x, y])
        z = stats.gaussian_kde(xy)(xy)
        # ===========Sort the points by density, so that the densest points are plotted last===========
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
        #绘图
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.set_xlim(0,1.5e6)
        ax.set_ylim(0,1.5e6)
        ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
        ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='x')
        ax.plot([0, 3.5e6], [0, 3.5e6], 'black', lw=1)  # 画的1:1线，线的颜色为black，线宽为0.8
        h = plt.scatter(x, y, c=z, s=2, edgecolor='', cmap='viridis')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        # 设置文本
        RMSE = stat_dict['RMSE']
        MAE = stat_dict['MAE']
        R2 = stat_dict['R2']
        ax.text(6e5, 3.3e5, '$RMSE=%.0f$' % RMSE,family='Arial', size=14)
        ax.text(6e5, 2e5, '$MAE=%.0f$' % MAE,family='Arial', size=14)
        ax.text(6e5, 0.7e5, '$R^2=%.3f$' % R2,family='Arial', size=14)
        # # colorbar
        cb = fig.colorbar(h)
        cb.set_ticks([])
        # 设置颜色带刻度
        cb.set_ticks([np.min(z), np.max(z)])
        cb.ax.yaxis.set_ticklabels(['Low', 'High'], fontsize=12,family='Arial')
        cb.ax.set_title('Density', pad=10,x=1.5,fontsize=14,family='Arial')
        plt.subplots_adjust(left=0.180, right=0.975, top=0.900, bottom=0.180, hspace=0.2, wspace=0.2)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        # ax.text(0.1, 0.95, title, transform=ax.transAxes, va='top', ha='left', family='Arial', size=14)
    def func_nationLevel_figure_allScalesScatter_density_2012_2013(self,year,xcol,ycol,xlabel,ylabel,title):
        dir = ROOTPATH+r'\data'
        path = os.path.join(dir,str(year)+'_nation_prj_adjusted.csv')
        df = pd.read_csv(path)
        stat_dict = stats_Param(df, xcol, ycol)
        # ===========Calculate the point density==========
        x = df[xcol]
        y = df[ycol]
        xy = np.vstack([x, y])
        z = stats.gaussian_kde(xy)(xy)
        # ===========Sort the points by density, so that the densest points are plotted last===========
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
        #绘图
        fig, ax = plt.subplots(figsize=(4, 3))
        ax.set_xlim(0,1e7)
        ax.set_ylim(0,1e7)
        ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
        ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='x')
        ax.plot([0, 3e7], [0, 3e7], 'black', lw=1)  # 画的1:1线，线的颜色为black，线宽为0.8
        h = plt.scatter(x, y, c=z, s=2, edgecolor='', cmap='viridis')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        # 设置文本
        RMSE = stat_dict['RMSE']
        MAE = stat_dict['MAE']
        R2 = stat_dict['R2']
        ax.text(3.8e6, 2.3e6, '$RMSE=%.0f$' %RMSE, family='Arial', size=14)
        ax.text(3.8e6, 1.4e6, '$MAE=%.0f$' % MAE,family='Arial', size=14)
        ax.text(3.8e6, 5e5, '$R^2=%.3f$' % R2,family='Arial', size=14)
        # # colorbar
        cb = fig.colorbar(h)
        cb.set_ticks([])
        # 设置颜色带刻度
        cb.set_ticks([np.min(z), np.max(z)])
        cb.ax.yaxis.set_ticklabels(['Low', 'High'], fontsize=12, family='Arial')
        cb.ax.set_title('Density', pad=10, x=1.5, fontsize=14, family='Arial')
        plt.subplots_adjust(left=0.180, right=0.975, top=0.900, bottom=0.180, hspace=0.2, wspace=0.2)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        # ax.text(0.1, 0.95, title, transform=ax.transAxes, va='top', ha='left', family='Arial', size=14)
    #返回一张图，所有子图出在一张图上
    def figure_allScalesScatter_density_2012_2013(self):
        fig = plt.figure(figsize=(12,12))
        xlabel = 'simulated VIIRS NTL intensity'
        ylabel = 'original VIIRS NTL intensity'
        '''pixel 2012 srDNL~VNL'''
        ax = plt.subplot(4,3,1)
        # ax.set_title('pixel level')
        self.func_pixelLevel_figure_allScalesScatter_density_2012_2013(ax,2012,'UNet_labelNoProcess_12_13_1571','VNL',xlabel,ylabel,'(a) 2012 SVNL and VIIRS')
        '''pixel 2013 srDNL~VNL'''
        ax = plt.subplot(4,3,2)
        self.func_pixelLevel_figure_allScalesScatter_density_2012_2013(ax, 2013, 'UNet_labelNoProcess_12_13_1571','VNL',xlabel,ylabel,'(b) 2013 SVNL and VIIRS')
        '''pixel 2012 ENL~VNL'''
        ax = plt.subplot(4,3,3)
        self.func_pixelLevel_figure_allScalesScatter_density_2012_2013(ax, 2012, 'CNL','VNL',xlabel,ylabel,'(c) 2012 ChenVNL and VIIRS')

        '''city 2012 srDNL~VNL'''
        ax = plt.subplot(4, 3, 4)
        # ax.set_title('city level')
        self.func_cityLevel_figure_allScalesScatter_density_2012_2013(ax,2012,'UNet_labelNoProcess_12_13_1571','VNL',xlabel,ylabel,'(a) 2012 SVNL and VIIRS')
        ax = plt.subplot(4, 3, 5)
        self.func_cityLevel_figure_allScalesScatter_density_2012_2013(ax,2013,'UNet_labelNoProcess_12_13_1571','VNL',xlabel,ylabel,'(b) 2013 SVNL and VIIRS')
        ax = plt.subplot(4, 3, 6)
        self.func_cityLevel_figure_allScalesScatter_density_2012_2013(ax, 2012, 'CNL', 'VNL',xlabel,ylabel,'(c) 2012 ChenVNL and VIIRS')

        '''province 2012 srDNL~VNL'''
        ax = plt.subplot(4, 3, 7)
        # ax.set_title('province level')
        self.func_provinceLevel_figure_allScalesScatter_density_2012_2013(ax,2012,'UNet_labelNoProcess_12_13_1571','VNL',xlabel,ylabel,'(a) 2012 SVNL and VIIRS')
        ax = plt.subplot(4, 3, 8)
        self.func_provinceLevel_figure_allScalesScatter_density_2012_2013(ax,2013,'UNet_labelNoProcess_12_13_1571','VNL',xlabel,ylabel,'(b) 2013 SVNL and VIIRS')
        ax = plt.subplot(4, 3, 9)
        self.func_provinceLevel_figure_allScalesScatter_density_2012_2013(ax, 2012, 'CNL', 'VNL',xlabel,ylabel,'(c) 2012 ChenVNL and VIIRS')

        '''nation 2012 srDNL~VNL'''
        ax = plt.subplot(4, 3, 10)
        # ax.set_title('nation level')
        self.func_nationLevel_figure_allScalesScatter_density_2012_2013(ax,2012,'UNet_labelNoProcess_12_13_1571','VNL',xlabel,ylabel,'(a) 2012 SVNL and VIIRS')
        ax = plt.subplot(4, 3, 11)
        self.func_nationLevel_figure_allScalesScatter_density_2012_2013(ax,2013,'UNet_labelNoProcess_12_13_1571','VNL',xlabel,ylabel,'(b) 2013 SVNL and VIIRS')
        ax = plt.subplot(4, 3, 12)
        self.func_nationLevel_figure_allScalesScatter_density_2012_2013(ax, 2012, 'CNL', 'VNL',xlabel,ylabel,'(c) 2012 ChenVNL and VIIRS')
        # fig.tight_layout()
        # plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.05, hspace=0.2)#调整子图间间距
        # # plt.savefig(savepath)
        # # plt.close()
        # outputFullPath = ROOTPATH+r'\allscalesScatterDensity2_7.tif'
        # plt.savefig(outputFullPath, dpi=300, bbox_inches='tight')
    #每张子图出一幅图，后面手动拼接成一整幅图
    def figure_each_allScalesScatter_density_2012_2013(self):
        xlabel = 'Simulated VIIRS NTL'
        ylabel = 'Real VIIRS NTL'
        # '''pixel 2012 srDNL~VNL'''
        self.func_pixelLevel_figure_allScalesScatter_density_2012_2013(2012,'UNet_labelNoProcess_12_13_1571','VNL',xlabel,ylabel,'(a) 2012 SVNL and VIIRS')
        outputFullPath = ROOTPATH + r'\subplot1\a1.tif'
        plt.savefig(outputFullPath, dpi=300, bbox_inches='tight')
        '''pixel 2013 srDNL~VNL'''
        self.func_pixelLevel_figure_allScalesScatter_density_2012_2013(2013, 'UNet_labelNoProcess_12_13_1571','VNL',xlabel,ylabel,'(b) 2013 SVNL and VIIRS')
        outputFullPath = ROOTPATH + r'\subplot1\b1.tif'
        plt.savefig(outputFullPath, dpi=300, bbox_inches='tight')
        '''pixel 2012 ENL~VNL'''
        self.func_pixelLevel_figure_allScalesScatter_density_2012_2013(2012, 'CNL','VNL',xlabel,ylabel,'(c) 2012 ChenVNL and VIIRS')
        outputFullPath = ROOTPATH + r'\subplot1\c1.tif'
        plt.savefig(outputFullPath, dpi=300, bbox_inches='tight')
        #
        # '''city 2012 srDNL~VNL'''
        # self.func_cityLevel_figure_allScalesScatter_density_2012_2013(2012,'UNet_labelNoProcess_12_13_1571','VNL',xlabel,ylabel,'(d) 2012 SVNL and VIIRS')
        # outputFullPath = ROOTPATH + r'\subplot1\a2.tif'
        # plt.savefig(outputFullPath, dpi=300, bbox_inches='tight')
        # self.func_cityLevel_figure_allScalesScatter_density_2012_2013(2013,'UNet_labelNoProcess_12_13_1571','VNL',xlabel,ylabel,'(e) 2013 SVNL and VIIRS')
        # outputFullPath = ROOTPATH + r'\subplot1\b2.tif'
        # plt.savefig(outputFullPath, dpi=300, bbox_inches='tight')
        # self.func_cityLevel_figure_allScalesScatter_density_2012_2013(2012, 'CNL', 'VNL',xlabel,ylabel,'(f) 2012 ChenVNL and VIIRS')
        # outputFullPath = ROOTPATH + r'\subplot1\c2.tif'
        # plt.savefig(outputFullPath, dpi=300, bbox_inches='tight')

        # '''province 2012 srDNL~VNL'''
        # self.func_provinceLevel_figure_allScalesScatter_density_2012_2013(2012,'UNet_labelNoProcess_12_13_1571','VNL',xlabel,ylabel,'(g) 2012 SVNL and VIIRS')
        # outputFullPath = ROOTPATH + r'\subplot1\a3.tif'
        # plt.savefig(outputFullPath, dpi=300, bbox_inches='tight')
        # self.func_provinceLevel_figure_allScalesScatter_density_2012_2013(2013,'UNet_labelNoProcess_12_13_1571','VNL',xlabel,ylabel,'(h) 2013 SVNL and VIIRS')
        # outputFullPath = ROOTPATH + r'\subplot1\b3.tif'
        # plt.savefig(outputFullPath, dpi=300, bbox_inches='tight')
        # self.func_provinceLevel_figure_allScalesScatter_density_2012_2013(2012, 'CNL', 'VNL',xlabel,ylabel,'(i) 2012 ChenVNL and VIIRS')
        # outputFullPath = ROOTPATH + r'\subplot1\c3.tif'
        # plt.savefig(outputFullPath, dpi=300, bbox_inches='tight')

        # '''nation 2012 srDNL~VNL'''
        # self.func_nationLevel_figure_allScalesScatter_density_2012_2013(2012,'UNet_labelNoProcess_12_13_1571','VNL',xlabel,ylabel,'(j) 2012 SVNL and VIIRS')
        # outputFullPath = ROOTPATH + r'\subplot1\a4.tif'
        # plt.savefig(outputFullPath, dpi=300, bbox_inches='tight')
        # self.func_nationLevel_figure_allScalesScatter_density_2012_2013(2013,'UNet_labelNoProcess_12_13_1571','VNL',xlabel,ylabel,'(k) 2013 SVNL and VIIRS')
        # outputFullPath = ROOTPATH + r'\subplot1\b4.tif'
        # plt.savefig(outputFullPath, dpi=300, bbox_inches='tight')
        # self.func_nationLevel_figure_allScalesScatter_density_2012_2013(2012, 'CNL', 'VNL',xlabel,ylabel,'(l) 2012 ChenVNL and VIIRS')
        # outputFullPath = ROOTPATH + r'\subplot1\c4.tif'
        # plt.savefig(outputFullPath, dpi=300, bbox_inches='tight')

plt_2_7 = plot_2_7_ScalesScatter_density()
plt_2_7.figure_each_allScalesScatter_density_2012_2013()