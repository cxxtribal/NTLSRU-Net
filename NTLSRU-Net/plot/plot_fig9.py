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

ROOTPATH = r'D:\jianguoyun\paper\论文\03NPP-VIIRS-like\plot\pic9'


def df_log(df, xcol, ycol):
    for index, row in df.iterrows():
        row[xcol] = math.log(row[xcol], math.e)
        row[ycol] = math.log(row[ycol], math.e)
        df.iloc[index] = row
    return df

'''合并各国1992-2020年夜间灯光统计表'''
def ntl_nation_merge_1992_2020():
    # # 加载数据
    path_countrycode = r'D:\jianguoyun\paper\论文\03NPP-VIIRS-like\plot\pic9\data\TNL1992-2020\nationID_join_PopCountryCode_SOC.csv'
    key_field_countrycode = 'ELEMID' #与国家shp的LEMID对应
    df_countrycode = pd.read_csv(path_countrycode,encoding='gb2312')
    colnamelst = df_countrycode.columns.tolist()
    # 加载srNTL 国家夜光总强度
    df = df_countrycode[[key_field_countrycode,'Country_or_area','ISO3_Alpha']]
    df.columns = ['ID','Country_or_area','ISO3']
    # srNTL_Dir = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\world\analysis\zonalStat\csv'
    srNTL_Dir = r'D:\jianguoyun\paper\论文\03NPP-VIIRS-like\plot\pic9\data\TNL1992-2020'
    key_field_NTL = 'ID'
    subcols = ['ID', 'UNet_labelNoProcess_12_13_1571']
    for year in range(1992, 2012):
        path_NTL = os.path.join(srNTL_Dir, 'nation_UNet_labelNoProcess_12_13_1571_' + str(year) + '_prj.csv')
        df_NTL = pd.read_csv(path_NTL)
        df_NTL = df_NTL[subcols]
        df_NTL.rename(columns={'UNet_labelNoProcess_12_13_1571': str(year)}, inplace=True)
        df = pd.merge(df, df_NTL, left_on=key_field_NTL, right_on=key_field_NTL)
        print(year, len(df))
    # 加载vnl 国家夜光总强度
    subcols = ['ID', 'VNL']
    for year in range(2012, 2021):
        path_NTL = os.path.join(srNTL_Dir, 'nation_VNL_' + str(year) + '_prj.csv')
        df_NTL = pd.read_csv(path_NTL)
        df_NTL = df_NTL[subcols]
        df_NTL.rename(columns={'VNL': str(year)}, inplace=True)
        df = pd.merge(df, df_NTL, left_on=key_field_NTL, right_on=key_field_NTL)
        print(year, len(df))
    outpath = r'D:\jianguoyun\paper\论文\03NPP-VIIRS-like\plot\pic9\data\NTL.csv'
    df.to_csv(outpath,header=True,index=False)
# ntl_nation_merge_1992_2020()

'''根据夜光强度统计表和GDP表,计算各国1992-2020年的时间序列R2'''
def cal_LinearRegression_nations_GDP_NTL_1992_2020():
    # 加载数据
    path_GDP = r'D:\jianguoyun\paper\论文\03NPP-VIIRS-like\plot\pic9\data\GDP(constant2015US).csv'
    key_field_gdp = 'Country Code'
    df_gdp = pd.read_csv(path_GDP,engine='python',encoding='gbk')
    df_gdp = df_gdp.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    path_NTL = r'D:\jianguoyun\paper\论文\03NPP-VIIRS-like\plot\pic9\data\NTL.csv'
    df_ntl = pd.read_csv(path_NTL)
    df_ntl = df_ntl.applymap(lambda x: x.strip() if isinstance(x, str) else x)
    key_field_ntl= 'ISO3'
    #计算每个国家的时间序列R2
    idlst = []
    nationlst = []
    klst = []
    blst = []
    r2lst = []
    klst_ln = []
    blst_ln = []
    r2lst_ln = []
    nationIDlst = df_ntl[key_field_ntl].values
    for nationID in nationIDlst:
        ntl_nation_df = df_ntl[df_ntl[key_field_ntl] == nationID]
        gdp_nation_df = df_gdp[df_gdp[key_field_gdp] == nationID]
        if (len(ntl_nation_df) < 1) or (len(gdp_nation_df)<1):
            continue
        xlst = []
        ylst = []
        yearlst = []
        for year in range(1992, 2021):
            # print(nationID,year)
            x = ntl_nation_df[str(year)].values[0]
            y = gdp_nation_df[str(year)].values[0]
            if pd.isna(x) or pd.isna(y) or str(y).strip() == '':
                continue
            yearlst.append(year)
            xlst.append(x)
            ylst.append(y)
        if len(yearlst) < 6:
            continue
        xcol = 'NTL'
        ycol = 'POP'
        nationDf = pd.DataFrame(data={xcol: xlst, ycol: ylst})
        # 线性拟合
        model = ols(ycol + " ~ " + xcol, data=nationDf).fit()
        k = model.params[xcol]
        b = model.params['Intercept']
        R2 = model.rsquared
        klst.append(k)
        blst.append(b)
        r2lst.append(R2)

        #对数线性拟合
        nationDf = nationDf.applymap(lambda x: np.log(1e-6 + x))
        model = ols(ycol + " ~ " + xcol, data=nationDf).fit()
        k = model.params[xcol]
        b = model.params['Intercept']
        R2 = model.rsquared
        klst_ln.append(k)
        blst_ln.append(b)
        r2lst_ln.append(R2)
        
        #添加国家
        id = ntl_nation_df['ID'].values[0]
        idlst.append(id)
        nationlst.append(nationID)
    result_df = pd.DataFrame(data={'ID': idlst,key_field_ntl: nationlst,
                                   'k': klst, 'b': blst,'r2': r2lst,
                                   'k_ln': klst_ln, 'b_ln': blst_ln,'r2_ln': r2lst_ln})
    outpath = r'D:\jianguoyun\paper\论文\03NPP-VIIRS-like\plot\pic9\data\r2_NTL_GDP.csv'
    result_df.to_csv(outpath,header=True,index=False)
    


    ##########图 2.15 全球夜间灯光总强度与人口数量时间序列回归分析

'''绘制全球GDP-NTL时序散点拟合图'''
def plot_2_15_world_GDP_NTL_LinearRegression_1992_2023():
    # srDNL ~ GDP 线性回归
    srDNL_path = ROOTPATH + r'\data\world-GDP-NTL.csv'
    srDNL_df = pd.read_csv(srDNL_path, encoding='gb18030', engine='python')
    xcol = 'NTL'
    ycol = 'GDP'

    # # 对gdp和enl取双log
    # srDNL_df = df_log(srDNL_df, xcol, ycol)

    srDNL_ylst = srDNL_df[ycol].values
    srDNL_xlst = srDNL_df[xcol].values
    srDNL_tlst = srDNL_df['year'].values
    # 线性回归
    srDNL_model = ols(ycol + " ~ " + xcol, data=srDNL_df).fit()
    srDNL_k = srDNL_model.params[xcol]
    srDNL_b = srDNL_model.params['Intercept']
    srDNL_R2 = srDNL_model.rsquared
    # 作图
    fig = plt.figure(figsize=(3.5,2.8))
    ax1 = plt.subplot(1,1,1)
    plt.scatter(srDNL_xlst, srDNL_ylst, c=srDNL_tlst)
    plt.ylabel('GDP')
    plt.xlabel('NTL intensity')
    ax1.ticklabel_format(style='sci', scilimits=(0, 0), axis='both')
    x1 = [0, max(srDNL_xlst) + 1]
    y1 = [srDNL_k * xi + srDNL_b for xi in x1]
    plt.plot(x1, y1, color='black')
    cb = plt.colorbar()
    ticks = [t for t in range(1992, 2022, 6)]
    ticks.append(2023)
    cb.set_ticks(ticks)
    plt.text(1.4e8,3.5e13,"$R^2 =%.2f$"%srDNL_R2)
    plt.subplots_adjust(left=0.175, right=0.93, top=0.93, bottom=0.19, hspace=0.2, wspace=0.2)
    ax1.xaxis.set_major_locator(plt.MaxNLocator(6))
    ax1.yaxis.set_major_locator(plt.MaxNLocator(6))

    # OUT
    outpath = ROOTPATH + r'\world_linearRegression1.png'
    plt.savefig(outpath, dpi=600, bbox_inches='tight')
    plt.close()


plot_2_15_world_GDP_NTL_LinearRegression_1992_2023()
