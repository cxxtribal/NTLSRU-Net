import os
import cv2
import numpy as np
import pandas as pd
import gc
from sklearn import metrics
import math

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.sandbox.regression.predstd import wls_prediction_std
from scipy import stats
from matplotlib.colors import Normalize
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import matplotlib as mlp
from  matplotlib import cm
from osgeo import ogr

class imagePlot():
    def readImg(self,path,refData=None):
        inData = cv2.imread(path, flags=cv2.IMREAD_UNCHANGED)
        inData = np.where(np.isnan(inData), 0, inData)
        inData[(inData <= 0)] = 0
        if refData is not None:
            inData[refData == 0] = 0
        return inData

    def minMax_imageShow(self,path,vmin=-1,vmax=-1,cmap='copper',title = ''):
        data = self.readImg(path)
        if vmin == -1:
            vmin = np.min(data)
        if vmax == -1:
            vmax = np.max(data)
        plt.figure()
        plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar()
        if title != '':
            plt.title(title)
    def ax_plot(self,axes,data,cmap,vmin,vmax,title):
        im = axes.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
        # plt.colorbar(im)
        plt.title(title)
        plt.axis('off')
        return im
    def minMax_imageShow_subplots(self,pathInfoLst,rownum,columnnum,vmin=-1,vmax=-1,cmap='copper'):
        fig = plt.figure(figsize=(16,8));
        for i in range(len(pathInfoLst)):
            ax = plt.subplot(rownum,columnnum,i+1)
            path,title = pathInfoLst[i]
            data = self.readImg(path)
            if i == 0:
                if vmin == -1:
                    vmin = np.min(data)
                if vmax == -1:
                    vmax = np.max(data)
                im = self.ax_plot(ax,data,cmap,vmin,vmax,title)
            else:
                self.ax_plot(ax, data, cmap, vmin, vmax, title)
        fig.subplots_adjust(right=0.9)
        l = 0.92
        b = 0.12
        w = 0.015
        h = 1 - 2 * b
        # 对应 l,b,w,h；设置colorbar位置；
        rect = [l, b, w, h]
        cbar_ax = fig.add_axes(rect)
        cb = plt.colorbar(im, cax=cbar_ax)
        # plt.tight_layout()

    def classfied(self,path='',rgb=None,bounds=None):
        if path == '':
            path = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\out\data\VIIRS\NY_2012_VIIRS.tif'
        if rgb is None:
            rgb = ([0, 97, 0], [63, 130, 0], [122, 171, 0], [183, 212, 0], [255, 255, 0], [255, 204, 0], [255, 153, 0],
                   [255, 102, 0], [255, 34, 0])
        if bounds is None:
            bounds = [0,3,10,20,30,40,50,65,100,1000]
        data = self.readImg(path)
        rgb = np.array(rgb) / 255.0
        cmap = colors.ListedColormap(rgb, name='my_color')
        norm = colors.BoundaryNorm(boundaries=bounds, ncolors=len(rgb),clip=True)
        fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.5))
        im1 = ax.imshow(data, norm=norm, cmap=cmap)
        ax.set_axis_off()
        plt.subplots_adjust(bottom=0.1, right=0.8, top=0.9)
        position = fig.add_axes([0.85, 0.15, 0.015, 0.7]) #调整子图的间距和页边缘
        cb = fig.colorbar(im1, cax=position)#用之前的创建的图像生成颜色带的方式；cax是绘制颜色带的轴线
        # cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)#不用之前的图像，即时创建颜色带的方式
        cb.set_ticks(bounds)
    def demo_subplot(self,ax,imgpath,norm,cmap):
        try:
            data = self.readImg(imgpath)
        except:
            print(imgpath)
        im1 = ax.imshow(data, norm=norm, cmap=cmap)
        ax.margins(0)
        # ax.set_axis_off() #关闭坐标，这个关闭的话，ylabel也不能显示
        # 去除黑框
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
    def classfied_subplots_r5c5(self,savepath,dataInfo=None,pathlst=None,rgb=None,bounds=None):
        if dataInfo is None:
            title = 'arid_2012_UNet_01_gradient_x'
            year = '2012'
            predir = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\out\out_patches\2022\arid'
            vnldir = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img04\2012_VNL'
            namelst = ['P1469', 'P1946', 'P1874', 'P2215', 'P1687']
            subplotnamelst = ['VNL','UNet_01_gradient_255','UNet_01_gradient_650','UNet_01_gradient_2175','UNet_01_gradient_2950']
        else:
            title,year,namelst,subplotnamelst = dataInfo
        if pathlst is None:
            pathlst = []
            for name in namelst:
                imgpathlst = []
                imgpathlst.append(os.path.join(vnldir,name+'.tif'))
                for i in range(1,len(subplotnamelst)):
                    imgpathlst.append(os.path.join(predir, name+'_'+year+'_'+ subplotnamelst[i]+'.tif'))
                pathlst.append((name,imgpathlst))
        if rgb is None:
            rgb = ([0, 97, 0], [63, 130, 0], [122, 171, 0], [183, 212, 0], [255, 255, 0], [255, 204, 0], [255, 153, 0],
                   [255, 102, 0], [255, 34, 0])
        if bounds is None:
            bounds = [0,3,10,20,30,40,50,65,100,1000]
        rgb = np.array(rgb) / 255.0
        cmap = colors.ListedColormap(rgb, name='my_color')
        norm = colors.BoundaryNorm(boundaries=bounds, ncolors=len(rgb),clip=True)
        n,m = len(pathlst),len(pathlst[0][1])
        fig, axs = plt.subplots(n, m, figsize=(10.5, 8))
        # fig, axs = plt.subplots(n, m)
        plt.setp(axs.flat, xticks=[], yticks=[]) #不显示坐标
        for i in range(n):
            imgid, imgpathlst = pathlst[i]
            for j in range(m):
                imgpath = imgpathlst[j]
                self.demo_subplot(axs[i,j],imgpath,norm,cmap)
        for ax, ve in zip(axs[0], subplotnamelst):
            ax.set_title(ve, size=8)
        for ax, mode in zip(axs[:, 0], namelst):
            ax.set_ylabel(mode, size=8)
        fig.suptitle(title, fontsize='xx-large',)
        fig.subplots_adjust(left=0.05,right=0.82,wspace=0.1,hspace=0.05) #调整子图的间距和页边缘
        position = fig.add_axes([0.85, 0.15, 0.015, 0.7])
        cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), cax=position)#不用之前的图像，即时创建颜色带的方式
        cb.set_ticks(bounds)
        # savepath = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\out\plot\plt\空间分布\arid_2012_UNet_01_gradient_x.tif'
        plt.savefig(savepath)
        plt.close("all")

class use_imagePlot():
    def use_minMax_imageShow(self):
        imgPlt = imagePlot()
        path_vnl = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\out\data\VIIRS\Cairo_2012_VIIRS.tif'
        imgPlt.imageShow(path_vnl, vmin=1, vmax=200, cmap='copper', title='VIIRS NTL')
        path_cnl = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\out\data\CNL\Cairo_2012_CNL.tif'
        imgPlt.imageShow(path_cnl, vmin=1, vmax=200, cmap='copper', title='CNL')
        path_srntl = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\out\out\Unet\UNet_01_rarid_class\Cairo_2012_01_rarid_class_100.tif'
        imgPlt.imageShow(path_srntl, vmin=1, vmax=200, cmap='copper', title='Cairo_2012_01_rarid_class_100')
        path_srntl = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\out\out\Unet\UNet_01_rarid_class\Cairo_2012_01_rarid_class_2550.tif'
        imgPlt.imageShow(path_srntl, vmin=1, vmax=200, cmap='copper', title='Cairo_2012_01_rarid_class_2550')
        path_srntl = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\out\out\Unet\UNet_01_rarid_class\Cairo_2012_01_rarid_class_2700.tif'
        imgPlt.imageShow(path_srntl, vmin=1, vmax=200, cmap='copper', title='Cairo_2012_01_rarid_class_2700')
        path_srntl = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\out\out\Unet\UNet_01_rarid_class\Cairo_2012_01_rarid_class_3000.tif'
        imgPlt.imageShow(path_srntl, vmin=1, vmax=200, cmap='copper', title='Cairo_2012_01_rarid_class_3000')
    def use_minMax_imageShow_subplots_BJH(self):
        imgPlt = imagePlot()
        region = 'GBA'
        year = '2012'
        pathInfoLst = []
        pathInfoLst.append((r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\out\data\VIIRS\\'+region+'_'+year+'_VIIRS.tif',region+" VNL "+year))
        pathInfoLst.append((r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\out\data\CNL\\'+region+'_2012_CNL.tif',region + " CNL 2012"))
        pathInfoLst.append((r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\out\out\Unet\Unet_labelNoProcess_12_13\\' + region + year + '_Unet_labelNoProcess_12_13_1571.tif',
                           region + " labelNoProcess_12_13_1571 " + year))
        pathInfoLst.append((r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\out\out\Unet\UNet_01_rarid_onehot\\' + region + year + '_Unet_01_rarid_onehot_2100.tif',region + " rarid_onehot_2100 " + year))
        pathInfoLst.append((r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\out\out\Unet\UNet_01_rarid_onehot\\' + region + year + '_Unet_01_rarid_onehot_2300.tif',region + " rarid_onehot_2300 " + year))
        pathInfoLst.append((r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\out\out\Unet\UNet_01_rarid_onehot\\' + region + year + '_Unet_01_rarid_onehot_2850.tif',region + " rarid_onehot_2850 " + year))
        pathInfoLst.append((r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\out\out\Unet\UNet_01_rarid_onehot\\' + region + year + '_Unet_01_rarid_onehot_3400.tif',region + " rarid_onehot_3400 " + year))
        pathInfoLst.append((r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\out\out\Unet\UNet_01_rarid_class\\' + region + year + '_Unet_01_rarid_class_2550.tif',region + " rarid_class_2550 " + year))
        pathInfoLst.append((r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\out\out\Unet\UNet_01_rarid_class\\' + region + year + '_Unet_01_rarid_class_2700.tif',region + " rarid_class_2700 " + year))
        pathInfoLst.append((r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\out\out\Unet\UNet_01_rarid_class\\' + region + year + '_Unet_01_rarid_class_3000.tif',region + " rarid_class_3000 " + year))
        pathInfoLst.append((r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\out\out\Unet\Unet_02\\' + region + year + '_UNet_02_378.tif',region + " UNet_02_378 " + year))
        pathInfoLst.append((r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\out\out\Unet\Unet_02\\' + region + year + '_UNet_02_448.tif',
                           region + " UNet_02_448 " + year))
        pathInfoLst.append((r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\out\out\Unet\Unet_02\\' + region + year + '_UNet_02_462.tif',
                           region + " UNet_02_462 " + year))
        imgPlt.imageShow_subplots(pathInfoLst,4,4,vmin=1,vmax=100,cmap='cool')

    def classfied_plot_diffArid_diffMdls(self):
        #outdir
        outdir = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\out\plot\plt\空间分布'
        aridList = []
        aridList.append(('arid',['P1469', 'P1946', 'P1874', 'P2215', 'P1687']))
        aridList.append(('drySubhumid', ['P1031', 'P1080', 'P1595', 'P2035', 'P2425']))
        aridList.append(('hyperarid', ['P1389', 'P1844', 'P1877', 'P2087', 'P2138']))
        aridList.append(('notArid', ['P1584', 'P1589', 'P2004', 'P2175', 'P2309']))
        aridList.append(('semiarid', ['P1180', 'P1254', 'P1559', 'P2234', 'P2353']))

        ##UNet_01_gradient
        # model = 'UNet_01_gradient'
        # eplst = [255,650,2175,2950]
        ##UNet_01_gradient_urbanMask
        # model = 'UNet_01_gradient_urbanMask'
        # eplst = [225, 550, 1525, 2800]
        ##UNet_03_img04
        # model = 'UNet_03_img04'
        # eplst = [475, 525, 1850, 2975]
        # subplotnamelst = ['VNL']
        # for ep in eplst:
        #     subplotnamelst.append(model + '_' + str(ep))
        # ##arid
        # model = 'arid'
        # subplotnamelst = ['VNL','Unet_01_rarid_class_2550','Unet_01_rarid_class_3000','Unet_01_rarid_without_drySubhumid_3950','UNet_labelNoProcess_12_13_1571']
        # #img
        # model = 'trainset'
        # subplotnamelst = ['VNL','UNet_03_img_1450','UNet_03_img03_475','UNet_03_img04_1850','UNet_labelNoProcess_12_13_1571']
        # #gradient
        # model = 'gradient'
        # subplotnamelst = ['VNL','UNet_01_gradient_2175','UNet_01_gradient_urbanMask_1525','UNet_03_img04_2975','UNet_labelNoProcess_12_13_1571']
        #CNL
        model = 'compare_02'
        subplotnamelst = ['VNL','UNet_03_img04_2975','UNet_01_gradient_urbanMask_1525','UNet_labelNoProcess_12_13_1571','CNL']

        for aridInfo in aridList:
            aridType,namelst = aridInfo
            #dataInfo
            year = '2013'
            title = aridType+'_'+year+'_'+model+'_x'
            dataInfo = (title, year, namelst, subplotnamelst)
            #pathlst
            predir = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\out\out_patches\2022'
            vnldir = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img04\2012_VNL'
            cnldir = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img04\2012_Chen'
            pathlst = []
            for name in namelst:
                imgpathlst = []
                imgpathlst.append(os.path.join(vnldir, name + '.tif'))
                for i in range(1, len(subplotnamelst)):
                    if subplotnamelst[i] == 'CNL':
                        imgpathlst.append(os.path.join(cnldir, name + '.tif'))
                    else:
                        imgpathlst.append(os.path.join(predir,aridType ,name + '_' + year + '_' + subplotnamelst[i] + '.tif'))
                pathlst.append((name, imgpathlst))

            savepath = os.path.join(outdir,title+'.png')
            #plot
            imgPlt = imagePlot()
            imgPlt.classfied_subplots_r5c5(savepath,dataInfo,pathlst)
# useimgPlt = use_imagePlot()
# useimgPlt.classfied_plot_diffArid_diffMdls()


class srDNL_fitAccess():
    def example_plot_box_violinplot(self):
        '''VNL和模拟VNL的风琴箱线图，分析数据分布'''
        path = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\world\analysis\zonalStat\csv\2012_point_prj_adjusted.csv'
        df = pd.read_csv(path)
        datacols = ['VNL', 'CNL', 'UNet_labelNoProcess_12_13_1571', 'UNet_01_gradient_urbanMask_1525',
                    'UNet_01_rarid_class_2550', 'UNet_03_img04_2975']
        plt.rcParams["figure.subplot.bottom"] = 0.23  # keep labels visible
        plt.rcParams["figure.figsize"] = (10.0, 8.0)  # make plot larger in notebook
        fig = plt.figure()
        ax = fig.add_subplot(111)
        data = [df[col].values for col in datacols]
        sm.graphics.violinplot(
            data,
            ax=ax,
            labels=datacols,
            plot_opts={
                "cutoff_val": 5,
                "cutoff_type": "abs",
                "label_fontsize": "small",
                "label_rotation": 30,
            },
        )
        ax.set_ylabel("NTL Intensity")
        ax.set_title("2012 points ")
    def example_plot_scatter_density(self):
        '''
        VNL和模拟VNL的散点密度图
        :return:
        '''
        #https://blog.csdn.net/qq_32442683/article/details/108349336
        path = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\world\analysis\zonalStat\csv\2012_point_prj_adjusted.csv'
        df = pd.read_csv(path)
        xcol = 'UNet_labelNoProcess_12_13_1571'
        ycol = 'VNL'
        stat_dict = self.stats_Param(df, xcol, ycol)
        # ===========Calculate the point density==========
        x = df[xcol]
        y = df[ycol]
        xy = np.vstack([x, y])
        z = stats.gaussian_kde(xy)(xy)
        # ===========Sort the points by density, so that the densest points are plotted last===========
        idx = z.argsort()
        x, y, z = x[idx], y[idx], z[idx]
        plt.rcParams["figure.subplot.bottom"] = 0.23  # keep labels visible
        plt.rcParams["figure.figsize"] = (10.0, 8.0)  # make plot larger in notebook
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot([-5, 400], [-5, 400], 'black', lw=0.8)  # 画的1:1线，线的颜色为black，线宽为0.8
        plt.scatter(x, y, c=z, s=7, edgecolor='', cmap='jet')
        # edgecolor: 设置轮廓颜色的,none是无色
        # 将c设置成一个y值列表并使用参数 cmap.cm.XX来使用某个颜色渐变(映射)
        # https://matplotlib.org/examples/color/colormaps_reference.html 从这个网页可以获得colorbard的名称
        # c=是设置数据点颜色的,在2.0的matplotlib中,edgecolor默认为'none'
        # c=还可以使用RGB颜色,例如 c = (0,0,0.8) 分别表示 红绿蓝,值越接近0,颜色越深.越接近1,颜色越浅
        # plt.axis([0, 1, 0, 1])  # 设置线的范围
        # plt.xlabel('srDNL', family='Times New Roman')
        # plt.ylabel('VNL', family='Times New Roman')
        # plt.xticks(fontproperties='Times New Roman')
        # plt.yticks(fontproperties='Times New Roman')
        plt.xlabel('srDNL')
        plt.ylabel('VNL')
        #设置文本
        RMSE = stat_dict['RMSE']
        MAE = stat_dict['MAE']
        R2 = stat_dict['R2']
        # plt.text(200, 350, '$RMSE=%.3f$' % RMSE, family='Times New Roman')
        # plt.text(200, 340, '$MAE=%.4f$' % MAE, family='Times New Roman')
        # plt.text(200, 330, '$R^2=%.3f$' % R2, family='Times New Roman')
        plt.text(200, 350, '$RMSE=%.3f$' % RMSE)
        plt.text(200, 330, '$MAE=%.4f$' % MAE)
        plt.text(200, 310, '$R^2=%.3f$' % R2)
        plt.xlim(-5, 400)  # 设置x坐标轴的显示范围
        plt.ylim(-5, 400)  # 设置y坐标轴的显示范围
        colorbar_ret = plt.colorbar()

    ########## 计算VNL和模拟VNL的拟合度，R2,RMSE,MAE#####################
    def readCsv(self,path):
        df = pd.read_csv(path)
        return df
    def stats_Param(self, df, xcol, ycol):
        y_true = df[ycol]
        y_pred = df[xcol]
        rmse = math.sqrt(metrics.mean_squared_error(y_true, y_pred))
        mae = metrics.mean_absolute_error(y_true, y_pred)
        evs = metrics.explained_variance_score(y_true, y_pred) #可解释性方差
        r2 = metrics.r2_score(y_true, y_pred)  #r2
        stat_dict = {'RMSE':rmse,'MAE':mae,'explained_variance_score':evs,'R2':r2}
        return stat_dict
    def run_stats_2012(self,path,df):
        if path !="":
            df = self.readCsv(path)
        ycol = 'VNL'
        xcol = 'CNL'
        cnl_stat_dict = self.stats_Param(df, xcol, ycol)
        xcol = 'UNet_labelNoProcess_12_13_1571'
        UNet_lnp_stat_dict = self.stats_Param(df, xcol, ycol)
        xcol = 'UNet_01_gradient_urbanMask_1525'
        UNet_gu_stat_dict = self.stats_Param(df, xcol, ycol)
        xcol = 'UNet_01_rarid_class_2550'
        UNet_rarid_stat_dict = self.stats_Param(df, xcol, ycol)
        xcol = 'UNet_03_img04_2975'
        UNet_img04_stat_dict = self.stats_Param(df, xcol, ycol)
        d = {'CNL': cnl_stat_dict, 'UNet_labelNoProcess_12_13_1571': UNet_lnp_stat_dict,
             'UNet_01_gradient_urbanMask_1525': UNet_gu_stat_dict, 'UNet_01_rarid_class_2550': UNet_rarid_stat_dict,'UNet_03_img04_2975':UNet_img04_stat_dict}
        stat_df = pd.DataFrame(d)
        return stat_df
    def run_stats_2013(self,path):
        df = self.readCsv(path)
        ycol = 'VNL'
        xcol = 'UNet_labelNoProcess_12_13_1571'
        UNet_lnp_stat_dict = self.stats_Param(df, xcol, ycol)
        xcol = 'UNet_01_gradient_urbanMask_1525'
        UNet_gu_stat_dict = self.stats_Param(df, xcol, ycol)
        xcol = 'UNet_01_rarid_class_2550'
        UNet_rarid_stat_dict = self.stats_Param(df, xcol, ycol)
        xcol = 'UNet_03_img04_2975'
        UNet_img04_stat_dict = self.stats_Param(df, xcol, ycol)
        d = {'UNet_labelNoProcess_12_13_1571': UNet_lnp_stat_dict,
             'UNet_01_gradient_urbanMask_1525': UNet_gu_stat_dict, 'UNet_01_rarid_class_2550': UNet_rarid_stat_dict,'UNet_03_img04_2975':UNet_img04_stat_dict}
        stat_df = pd.DataFrame(d)
        return stat_df

    ########## POP和模拟VNL的回归分析#####################
    def func_population_ntl_linearRegression_plot(self,index,xcols,ycol,subdf):
        #线性拟合
        model = ols(ycol+" ~ "+xcols[index] , data=subdf).fit()
        model_summary = model.summary()
        print(model_summary)
        # #线性回归图像
        # fig = plt.figure(figsize=(15, 8))
        # fig = sm.graphics.plot_regress_exog(model, xcols[index], fig=fig)
        #置信区间制图
        x = subdf[xcols[index]]
        y = subdf[ycol]
        # 获取置信区间
        # wls_prediction_std(housing_model)返回三个值,标准差，置信区间下限，置信区间上限
        # 标准差我们这里用不到，用 _ 表示一个虚拟变量占个位
        _, confidence_interval_lower, confidence_interval_upper = wls_prediction_std(model)
        fig, ax = plt.subplots(figsize=(10, 7))
        # 'o' 代表圆形, 也可以使用 'd'(diamonds菱形),'s'(squares正方形)
        ax.plot(x, y, 'o', label="data")
        # 画拟合线，g-- 代表绿色
        ax.plot(x, model.fittedvalues, "g--", label="OLS")
        # 画出置信区间 r-- 代表红色
        ax.plot(x, confidence_interval_upper, "r--")
        ax.plot(x, confidence_interval_lower, "r--")
        #显示拟合方程和R2
        k = model.params[xcols[index]]
        b = model.params['Intercept']
        R2 = model.rsquared
        plt.text(5.5, 6.8, r'$y = %.3fx+%.3f$' % (k,b))
        plt.text(5.5, 6.5, '$R^2=%.3f$' % R2)
        # 显示图例
        ax.legend(loc="best")
        plt.xlabel(xcols[index])
        plt.ylabel('Population')
    def population_ntl_logLinearRegression_2012(self):
        #加载数据
        path_pop = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\world\data\shp\zone\Pop_selectedCountry.csv'
        key_field_pop = 'ELEMID'
        df_pop = pd.read_csv(path_pop)
        year = '2012'
        path_ntl = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\world\analysis\zonalStat\csv\\'+year+'_nation_prj_adjusted.csv'
        key_field_ntl = 'ID'
        df_ntl = pd.read_csv(path_ntl)
        #merge
        df = pd.merge(df_pop,df_ntl,left_on=key_field_pop,right_on=key_field_ntl)
        cols = [year,'VNL','CNL','UNet_labelNoProcess_12_13_1571','UNet_01_gradient_urbanMask_1525','UNet_01_rarid_class_2550','UNet_03_img04_2975']
        subdf = df[cols]
        #对数处理
        for col in cols:
            subdf[col] = np.log10(subdf[col])
        xcols = cols[1:]
        ycol = 'Pop'+year
        subdf.rename(columns={year: 'Pop' + year}, inplace=True)
        index = 0
        self.func_population_ntl_linearRegression_plot(index,xcols,ycol,subdf)
    def annual_population_srCNL_logLinearRegression(self):
        #加载数据
        path_pop = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\world\data\shp\zone\Pop_selectedCountry.csv'
        key_field_pop = 'ELEMID'
        df_pop = pd.read_csv(path_pop)
        klst = []
        blst = []
        r2lst = []
        for i in range(1992,2014):
            print(i)
            # year = '1992'
            year = str(i)
            if year in ('2012','2013'):
                path_ntl = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\world\analysis\zonalStat\csv\\' +year+'_nation_prj_adjusted.csv'
            else:
                path_ntl = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\world\analysis\zonalStat\csv\\'+year+'_nation_prj.csv'
            key_field_ntl = 'ID'
            df_ntl = pd.read_csv(path_ntl)
            # merge
            df = pd.merge(df_pop, df_ntl, left_on=key_field_pop, right_on=key_field_ntl) # 内连接
            cols = [year, 'UNet_labelNoProcess_12_13_1571']
            subdf = df[cols]
            subdf = subdf[subdf['UNet_labelNoProcess_12_13_1571'] > 0]
            #对数处理
            for col in cols:
                subdf[col] = np.log10(subdf[col])
            ycol = 'Pop' + year
            xcol = 'srDNL' + year
            subdf.rename(columns={year: ycol}, inplace=True)
            subdf.rename(columns={'UNet_labelNoProcess_12_13_1571': xcol}, inplace=True)
            #线性拟合
            model = ols(ycol+" ~ "+xcol , data=subdf).fit()
            k = model.params[xcol]
            b = model.params['Intercept']
            R2 = model.rsquared
            klst.append(k)
            blst.append(b)
            r2lst.append(R2)

        xlst = [i for i in range(1992,2014)]
        fig = plt.figure(figsize=(15,9))
        ax = fig.add_subplot(311)
        ax.plot(xlst,r2lst)
        ax.scatter(xlst,r2lst)
        ax.set_ylabel(r'$R^2$')
        ax.set_xlabel('Year')
        ax.set_xticks(xlst)
        ax = fig.add_subplot(312)
        ax.plot(xlst,klst)
        ax.scatter(xlst,klst)
        ax.set_ylabel(r'$k$')
        ax.set_xlabel('Year')
        ax.set_xticks(xlst)
        ax = fig.add_subplot(313)
        ax.plot(xlst,blst)
        ax.scatter(xlst,blst)
        ax.set_ylabel(r'$b$')
        ax.set_xlabel('Year')
        ax.set_xticks(xlst)
        #保存回归参数
        dict = {'year':xlst,'r2':r2lst,'k':klst,'b':blst}
        param_df = pd.DataFrame(dict)
        outpath = r'D:\04study\00Paper\Dissertation\03report\03论文内容\图表\实验一NTL\02POP相关性分析\年度时间序列\0基于验证国家Pop_srDNL回归参数年度时间序列.csv'
        param_df.to_csv(outpath, header=True, index=False)


class matplotlib_shp():
    def plot_point(self,point, symbol='ko', **kwargs):
        x, y = point.GetX(), point.GetY()
        plt.plot(x, y, symbol, **kwargs)

    def plot_line(self,line, symbol='g-', **kwargs):
        x, y = zip(*line.GetPoints())
        plt.plot(x, y, symbol, **kwargs)

    def plot_polygon(self,poly, symbol='r-', **kwargs):
        for i in range(poly.GetGeometryCount()):
            subgeom = poly.GetGeometryRef(i)
            x, y = zip(*subgeom.GetPoints())
            plt.plot(x, y, symbol, **kwargs)

    def plot_layer(self,filename, symbol, layer_index=0, **kwargs):
        ds = ogr.Open(filename)
        for row in ds.GetLayer(layer_index):
            geom = row.geometry()
            geom_type = geom.GetGeometryType()

            if geom_type == ogr.wkbPoint:
                self.plot_point(geom, symbol, **kwargs)
            elif geom_type == ogr.wkbMultiPoint:
                for i in range(geom.GetGeometryCount()):
                    subgeom = geom.GetGeometryRef(i)
                    self.plot_point(subgeom, symbol, **kwargs)

            elif geom_type == ogr.wkbLineString:
                self.plot_line(geom, symbol, **kwargs)
            elif geom_type == ogr.wkbMultiLineString:
                for i in range(geom.GetGeometryCount()):
                    subgeom = geom.GetGeometryRef(i)
                    self.plot_line(subgeom, symbol, **kwargs)

            elif geom_type == ogr.wkbPolygon:
                self.plot_polygon(geom, symbol, **kwargs)
            elif geom_type == ogr.wkbMultiPolygon:
                for i in range(geom.GetGeometryCount()):
                    subgeom = geom.GetGeometryRef(i)
                    self.plot_polygon(subgeom, symbol, **kwargs)

    def test(self):
        os.chdir(r'F:\\烨叶\2020-8-20\\全国省界、市界的行政边界数据_WGS1984')
        # 下面三个谁在上边就先显示谁，我就按照点，线，面来了
        self.plot_layer('省会.shp', 'ko', markersize=5)
        self.plot_layer('省界.shp', 'r-')
        self.plot_layer('中国地图_投影.shp', 'g-', markersize=20)
        plt.axis('equal')
        plt.gca().get_xaxis().set_ticks([])
        plt.gca().get_yaxis().set_ticks([])
        plt.show()





















