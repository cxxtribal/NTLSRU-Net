import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from datetime import datetime
import skimage.measure
import sklearn.metrics
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import os
import time
import sys
import pandas as pd
import pickle
import argparse
from osgeo import gdal,gdalconst

# sys.path.append("/home/zju/cxx/NTL_timeserise/code/ExtractedUrbanFromNTL/commontools")
# sys.path.append("/home/zju/cxx/NTL_timeserise/code/ExtractedUrbanFromNTL/NPP_LIKE")
# sys.path.append("/home/zju/cxx/NTL_timeserise/code/ExtractedUrbanFromNTL")
# sys.path.append("/home/zju/cxx/NTL_timeserise/code/ExtractedUrbanFromNTL/src")

from NPP_LIKE.options import *
from NPP_LIKE.load_Dataset import Dataset_oriDMSP,load_data_path,addBatchDimension,read_imgdata
from NPP_LIKE.net.DRLN import DNLSRNet
from NPP_LIKE.utils import sum_dict
from largeRasterImageOp import getRasterValueByPoints,sumOfDN_RasterValue_maskByPolygon,getMultiAttrsToCsv,sumOfDN_RasterValue_maskByPolygon_cropToCutline_True
from largeRasterImageOp import clipRaster_byEnvelope_shp_sigleGrid
from NPP_LIKE.prepareDataset import getRNTL_YearDict


import rasterOp
from commontools.largeRasterImageOp import rowcol_to_xy,xy_to_rowcol,clipRaster_byextent

Quality_Indices_Lst = ['MPSNR', 'MSSIM', 'RMSE', 'MAE', 'R2','TNLE','MRE','DNnum','r2_max','r2_max_range']

def loss_plot():
    reloaded_config_path = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\net_02_1210\UNet\UNet_03_img04\config\config.pickle'
    with open(reloaded_config_path, "rb") as config_f:
        config = pickle.load(config_f)
    eplst = list(range(1, len(config.avg_loss_G) + 1, 1))
    eplst_01=eplst
    config_01= config
    netname = 'UNet_03_img'

    plt.figure()
    plt.plot(eplst, config.avg_loss_G, label='train')
    plt.plot(eplst, config.test_avg_loss_G, label='test')
    plt.legend()
    plt.title(netname + " ep = " + str(eplst[0]) + ':' + str(eplst[-1]))

    px = [51,100,201,300,400,501,1576]
    py = [config.test_avg_loss_G[i-1] for i in px]
    ptxt = [str(i) for i in px]
    plt.scatter(px,py,c='red')
    for i in range(0,len(px)):
        plt.text(px[i],py[i],ptxt[i])

    ind1 = 100
    ind2=2000
    plt.figure()
    plt.plot(eplst[ind1:ind2], config.avg_loss_G[ind1:ind2], label='train')
    # plt.plot(eplst[ind1:ind2], config.test_avg_loss_G[ind1:ind2], label='test')
    plt.legend()
    plt.title(netname+" ep = "+str(eplst[ind1])+':'+str(eplst[ind2-1]))

    ind1 = 200
    plt.figure()
    plt.plot(eplst[ind1:], config.avg_loss_G[ind1:], label='train')
    # plt.plot(eplst[ind1:], config.test_avg_loss_G[ind1:], label='test')
    plt.legend()
    plt.title(netname + " ep = " + str(eplst[ind1]) + ':' + str(eplst[-1]))
    plt.ylim(20000, 30000)

    ind1 = 1
    ind2 = 200
    for i in range(ind1,ind2):
        print(eplst[i],config.avg_loss_G[i],config.test_avg_loss_G[i])

    # ################################################## VALID
    # net_valid_lst = config_unet.valid_quality_indices_dict['valid']['EPOCH']
    # net_RMSE_lst = config_unet.valid_quality_indices_dict['valid']['RMSE']
    # net_MRE_lst = config_unet.valid_quality_indices_dict['valid']['MRE']
    # net_TNL_lst = config_unet.valid_quality_indices_dict['valid']['TNL']
    #
    # dnet_valid_lst = config_noCA.valid_quality_indices_dict['valid_07']['EPOCH']
    # dnet_RMSE_lst = config_noCA.valid_quality_indices_dict['valid_07']['RMSE']
    # dnet_MRE_lst = config_noCA.valid_quality_indices_dict['valid_07']['MRE']
    # dnet_TNL_lst = config_noCA.valid_quality_indices_dict['valid_07']['TNL']
    # vsNetname = 'DNLSRNet_noCA'
    # ind1 = 0
    # ind2=100
    # plt.figure()
    # plt.subplot(1,3,1)
    # plt.plot(net_valid_lst[ind1:], net_RMSE_lst[ind1:], label='UNet')
    # plt.plot(dnet_valid_lst[ind2:], dnet_RMSE_lst[ind2:], label=vsNetname)
    # plt.legend()
    # plt.title('RMSE')
    # ind1 = 0
    # ind2=80
    # plt.subplot(1,3,2)
    # plt.plot(net_valid_lst[ind1:], net_MRE_lst[ind1:], label='UNet')
    # plt.plot(dnet_valid_lst[ind2:], dnet_MRE_lst[ind2:], label=vsNetname)
    # plt.legend()
    # plt.title('MRE')
    # ind1 = 0
    # ind2=70
    # plt.subplot(1,3,3)
    # plt.plot(net_valid_lst[ind1:], net_TNL_lst[ind1:], label='UNet')
    # plt.plot(dnet_valid_lst[ind2:], dnet_TNL_lst[ind2:], label=vsNetname)
    # plt.legend()
    # plt.title('TNL')

def train_valid_loss_plt(config):
    ################################################## VALID
    net_valid_eplst = np.array(config.valid_quality_indices_dict['valid']['EPOCH'])
    net_RMSE_lst = np.array(config.valid_quality_indices_dict['valid']['RMSE'])
    net_MRE_lst = np.array(config.valid_quality_indices_dict['valid']['MRE'])
    net_TNL_lst = np.array(config.valid_quality_indices_dict['valid']['TNL'])

    testloss_noOtherData = np.array(config.test_avg_loss_G)
    rmselst_noOtherData = net_RMSE_lst
    mrelst_noOtherData = net_MRE_lst
    tnllst_noOtherData = net_TNL_lst

    test_indlst = np.argsort( np.array(config.test_avg_loss_G))
    rmse_indlst = np.argsort(net_RMSE_lst)
    mre_indlst = np.argsort(net_MRE_lst)
    tnl_indlst = np.argsort(net_TNL_lst)

    indlst = test_indlst
    for i in range(0, 5):
        ind = indlst[i]
        print(ind + 1, config.avg_loss_G[ind], config.test_avg_loss_G[ind])
        # print(ind+1,config.avg_loss_G[ind], config.test_avg_loss_G[ind],net_RMSE_lst[ind+1], net_MRE_lst[ind+1],net_TNL_lst[ind+1])

    indlst = rmse_indlst
    for i in range(0, 5):
        ind = indlst[i]
        ep = net_valid_eplst[ind] - 1
        print(net_valid_eplst[ind], config.avg_loss_G[ep], config.test_avg_loss_G[ep], net_RMSE_lst[ind],
              net_MRE_lst[ind], net_TNL_lst[ind])

    netname = 'UNet_03_img'
    vsNetname = 'Unet_01'
    ind1 = 0
    ind2=-1
    plt.figure()
    plt.subplot(1,3,1)
    plt.plot(net_valid_eplst[ind1:ind2], net_RMSE_lst[ind1:ind2], label=netname)
    # plt.plot(dnet_valid_lst[ind2:], dnet_RMSE_lst[ind2:], label=vsNetname)
    plt.legend()
    plt.title('RMSE')
    ind1 = 0
    ind2=-1
    plt.subplot(1,3,2)
    plt.plot(net_valid_eplst[ind1:ind2], net_MRE_lst[ind1:ind2], label=netname)
    # plt.plot(dnet_valid_lst[ind2:], dnet_MRE_lst[ind2:], label=vsNetname)
    plt.legend()
    plt.title('MRE')
    ind1 = 0
    ind2=-1
    plt.subplot(1,3,3)
    plt.plot(net_valid_eplst[ind1:ind2], net_TNL_lst[ind1:ind2], label=netname)
    # plt.plot(dnet_valid_lst[ind2:], dnet_TNL_lst[ind2:], label=vsNetname)
    plt.legend()
    plt.title('TNL')

def train_valid_loss_compare_plt():
    plt.figure()
    # net_valid_eplst = valid_eplst_01
    # net_RMSE_lst = rmselst_01
    # net_MRE_lst = mrelst_01
    # net_TNL_lst = tnllst_01
    # dnet_valid_lst = valid_eplst_noProce
    # dnet_RMSE_lst = rmselst_noProce
    # dnet_MRE_lst = mrelst_noProce
    # dnet_TNL_lst = tnllst_noProce
    # netname = 'UNet_01'
    # vsNetname = 'Unet_labelNoProcess'
    # ind1 = 0
    # ind2 = 50
    # plt.figure()
    # plt.subplot(1, 3, 1)
    # plt.plot(net_valid_eplst[ind1:], net_RMSE_lst[ind1:], label=netname)
    # plt.plot(dnet_valid_lst[ind2:], dnet_RMSE_lst[ind2:], label=vsNetname)
    # plt.legend()
    # plt.title('RMSE')
    # ind1 = 0
    # ind2 = 50
    # plt.subplot(1, 3, 2)
    # plt.plot(net_valid_eplst[ind1:], net_MRE_lst[ind1:], label=netname)
    # plt.plot(dnet_valid_lst[ind2:], dnet_MRE_lst[ind2:], label=vsNetname)
    # plt.legend()
    # plt.title('MRE')
    # ind1 = 0
    # ind2 = 50
    # plt.subplot(1, 3, 3)
    # plt.plot(net_valid_eplst[ind1:], net_TNL_lst[ind1:], label=netname)
    # plt.plot(dnet_valid_lst[ind2:], dnet_TNL_lst[ind2:], label=vsNetname)
    # plt.legend()
    # plt.title('TNL')


    # net_eplst = np.array(list(range(1, len(config_01.avg_loss_G) + 1, 1)))
    # net_testloss = config_01.test_avg_loss_G
    # net_valid_eplst = np.array(config_01.valid_quality_indices_dict['valid']['EPOCH'])
    # net_RMSE_lst =  np.array(config_01.valid_quality_indices_dict['valid']['RMSE'])
    # net_MRE_lst = np.array(config_01.valid_quality_indices_dict['valid']['MRE'])
    # net_TNL_lst = np.array(config_01.valid_quality_indices_dict['valid']['TNL'])
    # dnet_eplst = np.array(list(range(1, len(config_lableNoPro.avg_loss_G) + 1, 1)))
    # dnet_testloss = config_lableNoPro.test_avg_loss_G
    # dnet_valid_lst = np.array(config_lableNoPro.valid_quality_indices_dict['valid']['EPOCH'])
    # dnet_RMSE_lst = np.array(config_lableNoPro.valid_quality_indices_dict['valid']['RMSE'])
    # dnet_MRE_lst =  np.array(config_lableNoPro.valid_quality_indices_dict['valid']['MRE'])
    # dnet_TNL_lst = np.array(config_lableNoPro.valid_quality_indices_dict['valid']['TNL'])
    # dnet_eplst_01 = np.array(list(range(1, len(config_noOtherData.avg_loss_G) + 1, 1)))
    # dnet_testloss_01 = config_noOtherData.test_avg_loss_G
    # dnet_valid_lst_01 = np.array(config_noOtherData.valid_quality_indices_dict['valid']['EPOCH'])
    # dnet_RMSE_lst_01 = np.array(config_noOtherData.valid_quality_indices_dict['valid']['RMSE'])
    # dnet_MRE_lst_01 = np.array(config_noOtherData.valid_quality_indices_dict['valid']['MRE'])
    # dnet_TNL_lst_01 = np.array(config_noOtherData.valid_quality_indices_dict['valid']['TNL'])
    # dnet_eplst_02 = np.array(list(range(1, len(config_NDVI_Cfcvg.avg_loss_G) + 1, 1)))
    # dnet_testloss_02 = config_NDVI_Cfcvg.test_avg_loss_G
    # dnet_valid_lst_02 = np.array(config_NDVI_Cfcvg.valid_quality_indices_dict['valid']['EPOCH'])
    # dnet_RMSE_lst_02 = np.array(config_NDVI_Cfcvg.valid_quality_indices_dict['valid']['RMSE'])
    # dnet_MRE_lst_02 = np.array(config_NDVI_Cfcvg.valid_quality_indices_dict['valid']['MRE'])
    # dnet_TNL_lst_02 =  np.array(config_NDVI_Cfcvg.valid_quality_indices_dict['valid']['TNL'])
    # netname = 'UNet_01'
    # vsNetname = 'Unet_labelNpProcess12_13'
    # vsNetname_01 = 'UNet_01_noOtherData'
    # vsNetname_02 = 'Unet_01_NDVI_Cfcvg12_13'
    # ind1 = 0
    # ind2 = 50
    # ind3=0
    # ind4=0
    # plt.figure()
    # plt.subplot(2, 2, 1)
    # plt.plot(net_valid_eplst[ind1:], net_RMSE_lst[ind1:], label=netname)
    # plt.plot(dnet_valid_lst[ind2:], dnet_RMSE_lst[ind2:], label=vsNetname)
    # plt.plot(dnet_valid_lst_01[ind3:], dnet_RMSE_lst_01[ind3:], label=vsNetname_01)
    # plt.plot(dnet_valid_lst_02[ind4:], dnet_RMSE_lst_02[ind4:], label=vsNetname_02)
    # plt.legend()
    # plt.title('RMSE')
    # ind1 = 0
    # ind2 = 0
    # plt.subplot(2, 2, 2)
    # plt.plot(net_valid_eplst[ind1:], net_MRE_lst[ind1:], label=netname)
    # plt.plot(dnet_valid_lst[ind2:], dnet_MRE_lst[ind2:], label=vsNetname)
    # plt.plot(dnet_valid_lst_01[ind3:], dnet_MRE_lst_01[ind3:], label=vsNetname_01)
    # plt.plot(dnet_valid_lst_02[ind4:], dnet_MRE_lst_02[ind4:], label=vsNetname_02)
    # plt.legend()
    # plt.title('MRE')
    # ind1 = 0
    # ind2 = 0
    # plt.subplot(2, 2, 3)
    # plt.plot(net_valid_eplst[ind1:], net_TNL_lst[ind1:], label=netname)
    # plt.plot(dnet_valid_lst[ind2:], dnet_TNL_lst[ind2:], label=vsNetname)
    # plt.plot(dnet_valid_lst_01[ind3:], dnet_TNL_lst_01[ind3:], label=vsNetname_01)
    # plt.plot(dnet_valid_lst_02[ind4:], dnet_TNL_lst_02[ind4:], label=vsNetname_02)
    # plt.legend()
    # plt.title('TNL')
    # ind1 = 150
    # ind2 = 150
    # endind = 1500
    # plt.subplot(2, 2, 4)
    # plt.plot(net_eplst[ind1:endind], net_testloss[ind1:endind], label=netname)
    # plt.plot(dnet_eplst[ind2:endind], dnet_testloss[ind2:endind], label=vsNetname)
    # plt.plot(dnet_eplst_01[ind1:endind], dnet_testloss_01[ind1:endind], label=vsNetname_01)
    # plt.plot(dnet_eplst_02[ind2:endind], dnet_testloss_02[ind2:endind], label=vsNetname_02)
    # plt.legend()
    # plt.title('Test loss')

def creatSRImg_sigimage(modelname,modelappendix,epoch,dmsppath,viirspath,waterpath,otherdata_dict,saveTifpath,isCUP,config=None):
    '''生成单张超分图像。后面做区域评价时被调用'''
    #get config
    if config is None:
        config = getconfig(modelname,modelappendix,epoch)
        config.dmsp_stat_dict = getDMSP_stat_dict()
        config.viirs_stat_dict = getVIIRS_stat_dict()
    if isCUP:
        device = torch.device('cpu')
        config.device = device

    #get net
    net = getNet_fromModel(config, epoch)
    #get data
    out_dict = load_data_path(dmsppath,viirspath,waterpath,otherdata_dict,'valid',config)
    out_dict = addBatchDimension(out_dict)
    #predict
    net.eval()
    with torch.no_grad():
        gen_hr = generateSISR(config, net, out_dict)
        outdata = outdata_transform_qualityassess(config, gen_hr)
        outdata = outdata.cpu()
        outdata = outdata[0].data.detach().numpy()[0]
    #save to img
    label_inDs = gdal.Open(viirspath)
    rasterOp.outputResult(label_inDs,outdata,saveTifpath)
# modelname,modelappendix = "DNLSRNet","2012_01_seed3000noGradient"
# dmsppath = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\out\01\data\part1\BJH_DNL2012.tif'
# viirspath = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\out\01\data\part1\BJH_VNL2012.tif'
# ndvi_stat_dict = {'normlizedtype': "minMax", 'min': 0, 'max': 1}
# ndvi_stat_dict['path'] = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\out\01\data\part1\BJH_ndvi2012.tif'
# otherdata_dict = {'NDVI': ndvi_stat_dict}
# outdir = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\out\01\out20210910\DNLSRNet_2012_01\DNLSRNet_2012_01_seed3000noGradient'
# if not os.path.exists(outdir):
#     os.makedirs(outdir)
# # epochlst = [1,3,14,37,70,120,217,378,566,650,1120,1512,2000,2730,3290]
# # epochlst = [2004,2016,2104,2319,3000,4000]
# epochlst = [4647]
# for epoch in epochlst:
#     saveTifpath = os.path.join(outdir,str(epoch)+'.tif')
#     creatSRImg_sigimage(modelname, modelappendix, epoch, dmsppath, viirspath, otherdata_dict, saveTifpath)

# dmsppath  = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\out\02\out20210910\NewYork_DNL2012.tif'
# viirspath = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\out\02\out20210910\NewYork_VNL2012.tif'
# ndvi_stat_dict = getNDVI_stat_dict()
# ndvi_stat_dict['path'] = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\out\02\out20210910\NewYork_ndvi2012.tif'
# otherdata_dict = {'NDVI': ndvi_stat_dict}
#
# modelname,modelappendix = "Unet","01"
# outdir = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\out\02\out20210910\UNet\UNet_01'
# if not os.path.exists(outdir):
#     os.makedirs(outdir)
# # epochlst = [1,3,14,37,70,120,217,378,566,650,1120,1512,2000,2730,3290]
# # epochlst = [2004,2016,2104,2319,3000,4000]
# epochlst = [1500]
# for epoch in epochlst:
#     s1 = datetime.now()
#     print('start:',s1)
#     saveTifpath = os.path.join(outdir,str(epoch)+'.tif')
#     creatSRImg_sigimage(modelname, modelappendix, epoch, dmsppath, viirspath, otherdata_dict, saveTifpath,isCUP=True)
#     s2 = datetime.now()
#     print('end:',s2,'  lasting:',s2-s1)


'''评价指标计算'''
class ImageQualityAccess():
    def calculate_psnr(self,im_true, im_test, data_range):
        if len(im_test) > 0:
            err = np.mean(np.square(im_true - im_test))
            if err < 1.0e-10:
                return 100
            return 10 * np.log10((data_range ** 2) / err)
        else:
            return float('inf')
    def image_quality_assessment(self,x_true, x_pred,data_range,mask =None):
        x_true, x_pred = x_true.astype(np.float64), x_pred.astype(np.float64)
        sim = skimage.measure.compare_ssim(X=x_true, Y=x_pred, data_range=data_range,gaussian_weights=True)
        if mask is not None:
            x_true = x_true[mask]
            x_pred = x_pred[mask]
        x_true = x_true.reshape(-1)
        x_pred = x_pred.reshape(-1)
        psnr = self.calculate_psnr(x_true,x_pred,data_range)
        # rmse = 0
        rmse = np.sqrt(sklearn.metrics.mean_squared_error(x_true,x_pred))
        mae = sklearn.metrics.mean_absolute_error(x_true,x_pred)
        x_true_1 = np.where(x_true==0,1,x_true)
        mre = np.mean(abs(x_true-x_pred)/(x_true_1))
        sum_x_true = sum(x_true)
        if sum_x_true == 0:
            sum_x_true = 1
        tnlerror = abs(sum(x_true) - sum(x_pred))/(sum_x_true)
        score = sklearn.metrics.r2_score(x_true,x_pred)
        DNnum = len(x_true.reshape(-1))
        result = {
            'MPSNR': psnr,
            'MSSIM': sim,
            'RMSE': rmse,
            'MAE': mae,
            'R2': score,
            'TNLE':tnlerror,
            'MRE':mre,
            'DNnum':DNnum
        }
        return result
    def cal_bestR2(self,x_true,x_pred):
        dnMinlst = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
        dnMaxlst = list(range(20,150,1))
        r2_max = 0
        bestMinDN = 0
        bestMaxDN = 0
        for minDN in dnMinlst:
            for maxDN in dnMaxlst:
                maskloc =  (x_true>=minDN)&(x_true<=maxDN)
                truedata = x_true[maskloc]
                preddata = x_pred[maskloc]
                score = r2_score(truedata,preddata)
                if score >= r2_max:
                    r2_max = score
                    bestMinDN = minDN
                    bestMaxDN = maxDN
        return r2_max,bestMaxDN,bestMinDN
    def cal_wholeImg_QI(self,x_true, x_pred,data_range,mask =None):
        qi = self.image_quality_assessment(x_true, x_pred,data_range,mask)
        r2_max, bestMaxDN, bestMinDN = self.cal_bestR2(x_true, x_pred)
        qi['r2_max'] = r2_max
        qi['r2_max_range'] = str(bestMinDN)+'-'+str(bestMaxDN)
        return qi

    def save_imgQI_toCSV(self,savepath,indices_lst):
        # cols = ['MAE','MSSIM','RMSE','MPSNR','R2','set0','year','dataset','data']
        cols = ['MPSNR', 'MSSIM', 'RMSE', 'MAE', 'R2','TNLE','MRE','DNnum']
        cols.append('year')
        cols.append('dataset')
        cols.append('data')
        cols.append('model')
        cols.append('epoch')
        df = pd.DataFrame(data=indices_lst)
        df = df[cols]
        if os.path.exists(savepath):
            df.to_csv(savepath, index=False, header=False, mode='a')
        else:
            df.to_csv(savepath, index=False, header=True, mode='a')
    def save_regionImgQI_toCSV_whole(self,savepath,indices_lst):
        cols = ['MPSNR', 'MSSIM', 'RMSE', 'MAE', 'R2','TNLE','MRE','DNnum','r2_max','r2_max_range']
        cols.append('region')
        cols.append('model')
        cols.append('year')
        df = pd.DataFrame(data=indices_lst)
        df = df[cols]
        if os.path.exists(savepath):
            df.to_csv(savepath, index=False, header=False, mode='a')
        else:
            df.to_csv(savepath, index=False, header=True, mode='a')
    def save_regionImgQI_toCSV_bins(self,savepath,indices_lst):
        cols = ['MPSNR', 'MSSIM', 'RMSE', 'MAE', 'R2', 'TNLE', 'MRE', 'DNnum']
        cols.append('region')
        cols.append('model')
        cols.append('bins')
        cols.append('year')
        df = pd.DataFrame(data=indices_lst)
        df = df[cols]
        if os.path.exists(savepath):
            df.to_csv(savepath, index=False, header=False, mode='a')
        else:
            df.to_csv(savepath, index=False, header=True, mode='a')


'''测试集评价'''
class SRImgValidateSetAccess():
    def __init__(self, imgrootdir="", valid_txt="", year="", otherdata_dict=None, saveCSV="", datarange=1000):
        self.imgrootdir = imgrootdir
        self.valid_txt = valid_txt
        self.year = year
        self.otherdata_dict = otherdata_dict
        self.saveCSV = saveCSV
        self.datarange = datarange
    def getPath(self):
        self.dnldir = os.path.join(self.imgrootdir, str(self.year) + '_oriDNL')
        self.vnldir = os.path.join(self.imgrootdir, str(self.year) + '_VNL')
        self.cnldir = os.path.join(self.imgrootdir, '2012_Chen')
        self.waterdir = os.path.join(self.imgrootdir, "Water_Aridity")
        self.dataname = os.path.split(self.imgrootdir)[-1]
        self.dataset = os.path.split(self.valid_txt)[-1].split('.')[0]
    def generate_SRImage_usedefined_patches(self):
        isCUP = False
        outdir = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\out\out_patches\2022'
        modelinfoLst = []
        modelinfoLst.append(("Unet", "01_rarid_without_drySubhumid", 3950, 'NDVI'))
        modelinfoLst.append(("UNet", "03_img", 1450, 'NDVI'))
        modelinfoLst.append(("UNet", "03_img03", 475, 'NDVI'))
        modelinfoLst.append(("UNet", "03_img04", 475, 'NDVI'))
        modelinfoLst.append(("UNet", "03_img04", 525, 'NDVI'))
        modelinfoLst.append(("UNet", "03_img04", 1850, 'NDVI'))
        modelinfoLst.append(("UNet", "03_img04", 2975, 'NDVI'))
        modelinfoLst.append(("UNet", "01_gradient", 255, 'NDVI'))
        modelinfoLst.append(("UNet", "01_gradient", 650, 'NDVI'))
        modelinfoLst.append(("UNet", "01_gradient", 2175, 'NDVI'))
        modelinfoLst.append(("UNet", "01_gradient", 2950, 'NDVI'))
        modelinfoLst.append(("Unet", "01_rarid_class",2550,'NDVI;Water_Aridity'))
        modelinfoLst.append(("Unet", "01_rarid_class",3000,'NDVI;Water_Aridity'))
        modelinfoLst.append(("UNet", "labelNoProcess_12_13", 1571, 'NDVI'))
        modelinfoLst.append(("UNet", "01_gradient_urbanMask", 225, 'NDVI'))
        modelinfoLst.append(("UNet", "01_gradient_urbanMask", 550, 'NDVI'))
        modelinfoLst.append(("UNet", "01_gradient_urbanMask", 1525, 'NDVI'))
        modelinfoLst.append(("UNet", "01_gradient_urbanMask", 2800, 'NDVI'))

        if not os.path.exists(outdir):
            os.makedirs(outdir)
        pathInfoLst = []
        imgdir = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img04'
        namelst = ['P2138','P2087','P1389','P1877','P1844']
        dirname = 'hyperarid'
        for name in namelst:
            pathInfoLst.append((imgdir,dirname,name))
        namelst = ['P1469', 'P1946', 'P1874', 'P2215', 'P1687']
        dirname = 'arid'
        for name in namelst:
            pathInfoLst.append((imgdir, dirname, name))
        namelst = ['P1080', 'P1031', 'P1595', 'P2035', 'P2425']
        dirname = 'drySubhumid'
        for name in namelst:
            pathInfoLst.append((imgdir, dirname, name))
        namelst = ['P1180', 'P2234', 'P2353', 'P1254', 'P1559']
        dirname = 'semiarid'
        for name in namelst:
            pathInfoLst.append((imgdir, dirname, name))
        namelst = ['P2004', 'P1584', 'P2175', 'P2309', 'P1589']
        dirname = 'notArid'
        for name in namelst:
            pathInfoLst.append((imgdir, dirname, name))
        yearlst = ['2012','2013']
        for modelinfo in modelinfoLst:
            modelname, modelappendix, epoch, otherdataNames = modelinfo
            otherdataNamelst = otherdataNames.split(";")
            config = getconfig(modelname, modelappendix, epoch)
            for year in yearlst:
                self.year = year
                for pathinfo in pathInfoLst:
                    print(modelinfo,year,pathinfo)
                    self.imgrootdir,dirname,tifname = pathinfo
                    self.getPath()
                    otherdata_dict = {}
                    for name in otherdataNamelst:
                        if name == "RNTL":
                            stat_dict = config.otherdata_dict['RNTL']
                            otherpath = os.path.join(self.imgrootdir, "2010_RNTL",tifname+'.tif')
                            stat_dict['path'] = otherpath
                            otherdata_dict['RNTL'] =  stat_dict
                        elif name == "Cfcvg":
                            stat_dict = config.otherdata_dict['Cfcvg']
                            otherpath = os.path.join(self.imgrootdir, str(self.year) + '_CfCvg',tifname+'.tif')
                            stat_dict['path'] = otherpath
                            otherdata_dict['Cfcvg'] =  stat_dict
                        elif name == 'NDVI' :
                            stat_dict = config.otherdata_dict['NDVI']
                            otherpath = os.path.join(self.imgrootdir, str(self.year) + '_NDVI',tifname+'.tif')
                            stat_dict['path'] = otherpath
                            otherdata_dict['NDVI'] = stat_dict
                        elif name == 'Water_Aridity':
                            stat_dict = config.otherdata_dict['Water_Aridity']
                            otherpath = os.path.join(self.imgrootdir, 'Water_Aridity',tifname+'.tif')
                            stat_dict['path'] = otherpath
                            otherdata_dict['Water_Aridity'] = stat_dict
                    self.otherdata_dict = otherdata_dict
                    tif_outdir = os.path.join(outdir,dirname)
                    if not os.path.exists(tif_outdir):
                        os.makedirs(tif_outdir)
                    saveTifpath = os.path.join(tif_outdir,tifname +'_'+year + "_" + modelname + '_' + modelappendix + "_" + str(epoch) + '.tif')
                    dmsppath = os.path.join(self.dnldir,tifname+'.tif')
                    viirspath = os.path.join(self.vnldir,tifname+'.tif')
                    waterpath = os.path.join(self.waterdir,tifname+'.tif')
                    if not os.path.exists(saveTifpath):
                        creatSRImg_sigimage(modelname, modelappendix, epoch, dmsppath, viirspath,waterpath, otherdata_dict,
                                            saveTifpath, isCUP=isCUP,config=config)
# srImgAccess = SRImgValidateSetAccess()
# srImgAccess.generate_SRImage_usedefined_patches()





'''区域评价'''
class SRImgRegionAccess():
    def __init__(self,regionIndex="",regionname="",srimgFoldername=""):
        self.rootdir = r'/home/zju/cxx/NTL_timeserise/data/SRCNN/data/out'
        self.regiondir = os.path.join(self.rootdir,regionIndex)
        self.regionIndex = regionIndex
        self.regionname = regionname
        self.srimgFoldername = srimgFoldername
        self.srimgDir = os.path.join(self.rootdir, self.regionIndex, self.srimgFoldername)
        if not os.path.exists(self.srimgDir):
            os.makedirs(self.srimgDir)
    def clip_toGet_input_Data(self):
        outdataDir = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\out\data'
        datainfolst = []
        # datainfolst.append(('2013','VIIRS',r'D:\01data\00整理\02夜间灯光\npp\annualV2\VNL_2013.tif'))
        # datainfolst.append(('2012', 'VIIRS', r'D:\01data\00整理\02夜间灯光\npp\annualV2\VNL_2012.tif'))
        # datainfolst.append(('2013', 'DNL', r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\outdata\02oriGNL_stepSNL\04interp\2013_interp.tif'))
        # datainfolst.append(('2012', 'DNL', r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\outdata\02oriGNL_stepSNL\04interp\2012_interp.tif'))
        # datainfolst.append(('2013', 'NDVI', r'D:\01data\00整理\04NDVI\landsatNDVI\NDVI_2013.tif'))
        # datainfolst.append(('2012', 'NDVI', r'D:\01data\00整理\04NDVI\landsatNDVI\NDVI_2012.tif'))
        # datainfolst.append(('2012', 'CNL', r'D:\01data\00整理\02夜间灯光\npp\Chen2012\resample_align\chenNTL_2012.tif'))
        # datainfolst.append(('2010', 'RNTL', r'D:\01data\00整理\02夜间灯光\grc_interp\2010_interp.tif'))
        # datainfolst.append(('2012', 'CfCvg', r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\outdata\cf_cvg\cf_cvg2012.tif'))
        # datainfolst.append(('2012', 'NDVI', r'D:\01data\00整理\04NDVI\landsatNDVI\250\NDVI_250_transTo_500_2012.tif'))
        # datainfolst.append(('2013', 'AVHRR', r'D:\01data\00整理\04NDVI\AVHRR\2013_AVHRR_500m.tif'))

        # cnlDir = r'D:\01data\00整理\02夜间灯光\grc_interp'
        # for year in (1996,1999,2000,2003,2005):
        #     print(year)
        #     path = os.path.join(cnlDir,str(year)+'_interp.tif')
        #     datainfolst.append((str(year), 'RNTL', path))
        # for year in range(2000,2012):
        #     cnlpath = os.path.join(cnlDir,str(year),"LongNTL_"+str(year)+".tif")
        #     datainfolst.append((str(year), 'CNL',cnlpath))
        cnlpath = r'D:\01data\00整理\干旱半干旱区域\Aridity Index\watermask_AridityIndex.tif'
        datainfolst.append(('', 'Water_Aridity', cnlpath))

        #裁剪区域对应图像
        regionlst = ['BJH', 'NY', 'Cairo', 'GBA', 'YRD']
        for regionname in regionlst:
            polypath = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\out\data\shp\\'+regionname+'_poly.shp'
            for datainfo in datainfolst:
                year,dirname,rasterpath = datainfo
                outdir = os.path.join(outdataDir,dirname)
                if not os.path.exists(outdir):
                    os.makedirs(outdir)
                if year == "":
                    outpath = os.path.join(outdataDir,dirname,regionname+"_"+dirname+'.tif')
                else:
                    outpath = os.path.join(outdataDir,dirname,regionname+"_"+year+'_'+dirname+'.tif')
                if not os.path.exists(outpath):
                    print(regionname, datainfo)
                    clipRaster_byEnvelope_shp_sigleGrid(polypath,rasterpath,outpath,0)
    def generate_single_SRImg_ByNDVI(self):
        modelname, modelappendix = "Unet", "01"
        outdir = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\out\out\\'+ modelname+'\\'+modelname+'_'+modelappendix
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        inputdataDir = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\out\data'
        # yearlst = ['2012','2013']
        yearlst = ['2012']
        regionlst = ['BJH','NY','Cairo','GBA','YRD']
        for regionname in regionlst:
            waterpath = os.path.join(inputdataDir, 'Water_Aridity', regionname + '_Water_Aridity.tif')
            for year in yearlst:
                dmsppath = os.path.join(inputdataDir,'DNL',regionname+"_"+year+'_DNL.tif')
                viirspath = os.path.join(inputdataDir, 'VIIRS', regionname + "_" + year + '_VIIRS.tif')

                epochlst = [1500]
                for epoch in epochlst:
                    s1 = datetime.now()
                    print('start:',s1)
                    # saveTifpath = os.path.join(outdir,regionname + year + "_" + modelname + '_' + modelappendix + "_" + str(epoch) + '.tif')
                    saveTifpath = os.path.join(outdir,
                                               regionname + year + "_NDVI250_" + modelname + '_' + modelappendix + "_" + str(
                                                   epoch) + '.tif')
                    if not os.path.exists(saveTifpath):
                        config = getconfig(modelappendix,modelappendix,epoch)
                        ndvipath = os.path.join(inputdataDir, 'NDVI', regionname + "_" + year + "_NDVI250.tif")
                        ndvi_stat_dict = config.otherdata_dict['NDVI']
                        ndvi_stat_dict['path'] = ndvipath
                        otherdata_dict = {'NDVI': ndvi_stat_dict}
                        creatSRImg_sigimage(modelname, modelappendix, epoch, dmsppath, viirspath,waterpath, otherdata_dict, saveTifpath,isCUP=False,config=config)
                        s2 = datetime.now()
                        print(regionname,year,epoch,'end:',s2,'  lasting:',s2-s1)
    def userDefined_generate_single_SRImg_ByNDVI(self):
        modelname, modelappendix = "Unet", "01_rarid_onehot"
        outdir = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\out\out\Unet\UNet_01_rarid_onehot'
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        inputdataDir = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\out\data'
        regionname = 'Cairo'
        dmsppath = os.path.join(inputdataDir, 'DNL', regionname + "_2012_DNL.tif")
        viirspath = os.path.join(inputdataDir, 'VIIRS', regionname + "_2012_VIIRS.tif")
        waterpath = os.path.join(inputdataDir, 'Water_Aridity', regionname + "_Water_Aridity.tif")
        epochlst = [200,2100,2300]
        for epoch in epochlst:
            s1 = datetime.now()
            print('start:', s1)
            saveTifpath = os.path.join(outdir, regionname +"_2012_"+modelappendix+'_'+str(epoch)+ '.tif')
            if not os.path.exists(saveTifpath):
                config = getconfig(modelname, modelappendix, epoch)
                ndvi_stat_dict = config.otherdata_dict['NDVI']
                ndvi_stat_dict['path'] = os.path.join(inputdataDir, 'NDVI',regionname + "_2012_NDVI.tif")
                arid_stat_dict = config.otherdata_dict['Water_Aridity']
                arid_stat_dict['path'] = os.path.join(inputdataDir, 'Water_Aridity', regionname + "_Water_Aridity.tif")
                otherdata_dict = {'NDVI': ndvi_stat_dict,'Water_Aridity':arid_stat_dict}
                creatSRImg_sigimage(modelname, modelappendix, epoch, dmsppath, viirspath,waterpath, otherdata_dict, saveTifpath,
                                    isCUP=False,config=config)
                s2 = datetime.now()
                print(regionname, epoch, 'end:', s2, '  lasting:', s2 - s1)
    def generate_single_SRImg_DiffMls(self):
        outRoot = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\out\out'
        inputdataDir = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\out\data'
        modelinfoLst = []
        modelinfoLst.append(("Unet", "01_rarid_class",2550,'NDVI;Water_Aridity'))
        modelinfoLst.append(("Unet", "01_rarid_class",3000,'NDVI;Water_Aridity'))

        # yearlst = ['2012','2013']
        yearlst = ['2012']
        # regionlst = ['BJH', 'Cairo', 'GBA', 'YRD', 'NY']
        regionlst = ['Cairo']
        for modelinfo in modelinfoLst:
            modelname, modelappendix,epoch,otherdataNames = modelinfo
            otherdataNamelst = otherdataNames.split(";")
            outdir = os.path.join(outRoot,modelname,modelname + '_' + modelappendix)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            config = getconfig(modelname, modelappendix, epoch)
            for regionname in regionlst:
                for year in yearlst:
                    t1 = datetime.now()
                    dmsppath = os.path.join(inputdataDir, 'DNL', regionname + "_" + year + '_DNL.tif')
                    viirspath = os.path.join(inputdataDir, 'VIIRS', regionname + "_" + year + '_VIIRS.tif')
                    waterpath = os.path.join(inputdataDir, 'Water_Aridity', regionname + '_Water_Aridity.tif')
                    otherdata_dict = {}
                    for name in otherdataNamelst:
                        if name == "RNTL":
                            stat_dict = config.otherdata_dict['RNTL']
                            otherpath = os.path.join(inputdataDir, name,regionname + "_2010_" + name + ".tif")
                            stat_dict['path'] = otherpath
                            otherdata_dict['RNTL'] =  stat_dict
                        elif name == "Cfcvg":
                            stat_dict = config.otherdata_dict['Cfcvg']
                            otherpath = os.path.join(inputdataDir, name,regionname + "_" + year + "_" + name + ".tif")
                            stat_dict['path'] = otherpath
                            otherdata_dict['Cfcvg'] =  stat_dict
                        elif  name == "NDVI":
                            stat_dict = config.otherdata_dict['NDVI']
                            otherpath = os.path.join(inputdataDir, name,regionname + "_" + year + "_" + name + ".tif")
                            stat_dict['path'] = otherpath
                            otherdata_dict['NDVI'] = stat_dict
                        elif name == 'Water_Aridity':
                            stat_dict = config.otherdata_dict['Water_Aridity']
                            otherpath = os.path.join(inputdataDir, name, regionname + "_" + name + ".tif")
                            stat_dict['path'] = otherpath
                            otherdata_dict['Water_Aridity'] = stat_dict
                        elif name == 'AVHRR':
                            stat_dict = config.otherdata_dict['AVHRR']
                            otherpath = os.path.join(inputdataDir, name,regionname + "_" + year + "_" + name + ".tif")
                            stat_dict['path'] = otherpath
                            otherdata_dict['AVHRR'] = stat_dict
                    saveTifpath = os.path.join(outdir,regionname + year + "_" + modelname + '_' + modelappendix + "_" + str(epoch) + '.tif')
                    if not os.path.exists(saveTifpath):
                        creatSRImg_sigimage(modelname, modelappendix, epoch, dmsppath, viirspath,waterpath, otherdata_dict,
                                            saveTifpath, isCUP=True,config=config)
                    t2 = datetime.now()
                    print(modelinfo,regionname,year,'时间：',t2-t1,'s')
    def generate_single_SRImg_DiffYears(self):
        outRoot = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\out\out'
        inputdataDir = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\out\data'
        modelinfoLst = []
        # modelinfoLst.append(("Unet", "01_train12_13", 2000,'NDVI'))
        # modelinfoLst.append(("Unet", "01_RNTL", 1950, 'RNTL'))
        # modelinfoLst.append(("Unet", "labelNoProcess_12_13", 1571, 'NDVI'))
        modelinfoLst.append(("Unet", "01_noOtherData", 3550, ''))

        rntlYearDict = getRNTL_YearDict()

        yearlst = [str(year) for year in range(1992,2012)]
        regionlst = ['BJH', 'NY', 'Cairo', 'GBA', 'YRD']
        for modelinfo in modelinfoLst:
            modelname, modelappendix,epoch,otherdataNames = modelinfo
            otherdataNamelst = otherdataNames.split(";")
            outdir = os.path.join(outRoot,modelname,modelname + '_' + modelappendix)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            config = getconfig(modelname, modelappendix, epoch)
            for regionname in regionlst:
                waterpath = os.path.join(inputdataDir, 'Water_Aridity', regionname + '_Water_Aridity.tif')
                for year in yearlst:
                    print(modelinfo,regionname,year)
                    dmsppath = os.path.join(inputdataDir, 'DNL', regionname + "_" + year + '_DNL.tif')
                    viirspath = os.path.join(inputdataDir, 'VIIRS', regionname + "_2012_VIIRS.tif")
                    otherdata_dict = {}
                    for name in otherdataNamelst:
                        if name == "RNTL":
                            stat_dict = config.otherdata_dict['RNTL']
                            rntlyear = str(rntlYearDict[int(year)])
                            otherpath = os.path.join(inputdataDir, name,regionname + "_" + rntlyear + "_" +  name + ".tif")
                            stat_dict['path'] = otherpath
                            otherdata_dict['RNTL'] =  stat_dict
                        elif name == "Cfcvg":
                            stat_dict = config.otherdata_dict['Cfcvg']
                            otherpath = os.path.join(inputdataDir, name,regionname + "_" + year + "_" + name + ".tif")
                            stat_dict['path'] = otherpath
                            otherdata_dict['Cfcvg'] =  stat_dict
                        elif name == "NDVI" :
                            stat_dict = config.otherdata_dict['NDVI']
                            otherpath = os.path.join(inputdataDir, name,regionname + "_" + year + "_" + name + ".tif")
                            stat_dict['path'] = otherpath
                            otherdata_dict['NDVI'] = stat_dict
                    saveTifpath = os.path.join(outdir,regionname + year + "_" + modelname + '_' + modelappendix + "_" + str(epoch) + '.tif')
                    if not os.path.exists(saveTifpath):
                        creatSRImg_sigimage(modelname, modelappendix, epoch, dmsppath, viirspath,waterpath, otherdata_dict,
                                            saveTifpath, isCUP=False,config=config)


    def createSRImg(self,modelname,modelappendix,epoch,year,otherdata_dict,batchsize=30,clipPadding=20):
        config = getconfig(modelname,modelappendix,epoch)
        device = torch.device('cpu')
        config.device = device
        net = getNet_fromModel(config, epoch)
        sampleTxtPath = os.path.join(self.rootdir,self.regionIndex,'valid.txt')
        dmsp_indir = os.path.join(self.rootdir, self.regionIndex, 'inputs',str(year)+'_oriDNL')
        viirs_indir = os.path.join(self.rootdir, self.regionIndex, 'inputs',str(year)+'_VNL')
        outdir = os.path.join(self.srimgDir, 'patches_'+modelname+"_"+modelappendix+"_"+str(epoch)+"_"+str(year))
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        dataset = Dataset_oriDMSP(config,sampleTxtPath,'valid',viirs_indir,dmsp_indir,otherdata_dict,is_multi_years_combined=False)
        dataloader = torch.utils.data.DataLoader(dataset,batch_size=batchsize,shuffle=False,drop_last=False)
        files = dataset.files
        # 投影信息
        tifpath = os.path.join(viirs_indir, files[0] + ".tif")
        geoProj = rasterOp.getProj_img(tifpath)
        #create sr img
        net.eval()
        with torch.no_grad():
            for idx, out_dict in enumerate(dataloader):
                gen_hr = generateSISR(config,net,out_dict)
                # 用归一化前的数据计算损失，输出  label 为原始归一化前数据
                outdata = outdata_transform_qualityassess(config, gen_hr)
                #转cpu
                outdata = outdata.cpu()
                # if jj==0:
                #     break
                #输出图像
                fileIndex = out_dict['index']
                batchnum = len(gen_hr)
                for i in range(batchnum):
                    gen_hr_i = outdata[i].data.detach().numpy()[0]
                    fileIndex_i = fileIndex[i].data.detach().numpy()[0]
                    image_id = files[fileIndex_i]
                    viirspath = os.path.join(viirs_indir,image_id + ".tif")
                    viirsDs = gdal.Open(viirspath)
                    gentrans_i = viirsDs.GetGeoTransform()
                    print(image_id)
                    #输出图像
                    # saveTifpath = os.path.join(outdir,image_id+'.tif')
                    # rasterOp.write_img(saveTifpath, geoProj, gentrans_i, gen_hr_i)
                    #输出裁剪图像
                    h,w = gen_hr_i.shape
                    gen_hr_i_clip = gen_hr_i[clipPadding:h-clipPadding,clipPadding:w-clipPadding]
                    xmin_new,ymax_new = rowcol_to_xy(gentrans_i,clipPadding,clipPadding)
                    geotrans_i_new = gentrans_i.copy()
                    geotrans_i_new[0] = xmin_new
                    geotrans_i_new[3] = ymax_new
                    saveTifpath_clip = os.path.join(outdir,image_id+"_padding"+str(clipPadding)+".tif")
                    rasterOp.write_img(saveTifpath_clip, geoProj, geotrans_i_new, gen_hr_i_clip)
                    del viirsDs,gen_hr_i,fileIndex_i,gentrans_i,gen_hr_i_clip,geotrans_i_new
    def createSRImg_Partition(self,modelname,modelappendix,DNSplitValues,epochlst,year,otherdata_dict,batchsize=30,clipPadding=20):
        #get net
        minvalues = []
        maxvalues = []
        for i in range(len(DNSplitValues)-1):
            minvalues.append(DNSplitValues[i])
            maxvalues.append(DNSplitValues[i+1])
        configlst = []
        netlst = []
        for i in range(epochlst):
            strModelappendix = modelappendix+'_'+str(minvalues[i])+'-'+str(maxvalues[i])
            config = getconfig(modelname, strModelappendix, epochlst[i])
            device = torch.device('cpu')
            config.device = device
            net = getNet_fromModel(config, epochlst[i])
            net.eval()
            configlst.append(config)
            netlst.append(net)
        #get SR IMG
        config = configlst[-1]
        for i in range(1,len(DNSplitValues)-1):
            modelappendix += "-"+str(DNSplitValues[i])
        epoch = epochlst[-1]
        sampleTxtPath = os.path.join(self.rootdir,self.regionIndex,'valid.txt')
        dmsp_indir = os.path.join(self.rootdir, self.regionIndex, 'inputs',str(year)+'_oriDNL')
        viirs_indir = os.path.join(self.rootdir, self.regionIndex, 'inputs',str(year)+'_VNL')
        outdir = os.path.join(self.srimgDir, 'patches_'+modelname+"_"+modelappendix+"_"+str(epoch)+"_"+str(year))
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        dataset = Dataset_oriDMSP(config,sampleTxtPath,'valid',viirs_indir,dmsp_indir,otherdata_dict,is_multi_years_combined=False)
        dataloader = torch.utils.data.DataLoader(dataset,batch_size=batchsize,shuffle=False,drop_last=False)
        files = dataset.files
        # 投影信息
        tifpath = os.path.join(viirs_indir, files[0] + ".tif")
        geoProj = rasterOp.getProj_img(tifpath)
        with torch.no_grad():
            for idx, out_dict in enumerate(dataloader):
                gen_hr = generateSISR_partition(out_dict,configlst,netlst,minvalues,maxvalues)
                # 用归一化前的数据计算损失，输出  label 为原始归一化前数据
                outdata = outdata_transform_qualityassess(config, gen_hr)
                #转cpu
                outdata = outdata.cpu()
                # if jj==0:
                #     break
                #输出图像
                batchnum = len(gen_hr)
                fileIndex = out_dict['index']
                for i in range(batchnum):
                    gen_hr_i = outdata[i].data.detach().numpy()[0]
                    fileIndex_i = fileIndex[i].data.detach().numpy()[0]
                    image_id = files[fileIndex_i]
                    viirspath = os.path.join(viirs_indir, image_id + ".tif")
                    viirsDs = gdal.Open(viirspath)
                    gentrans_i = viirsDs.GetGeoTransform()
                    print(image_id)
                    #输出图像
                    # saveTifpath = os.path.join(outdir,image_id+'.tif')
                    # rasterOp.write_img(saveTifpath, geoProj, gentrans_i, gen_hr_i)
                    #输出裁剪图像
                    h,w = gen_hr_i.shape
                    gen_hr_i_clip = gen_hr_i[clipPadding:h-clipPadding,clipPadding:w-clipPadding]
                    xmin_new,ymax_new = rowcol_to_xy(gentrans_i,clipPadding,clipPadding)
                    geotrans_i_new = gentrans_i.copy()
                    geotrans_i_new[0] = xmin_new
                    geotrans_i_new[3] = ymax_new
                    saveTifpath_clip = os.path.join(outdir,image_id+"_padding"+str(clipPadding)+".tif")
                    rasterOp.write_img(saveTifpath_clip, geoProj, geotrans_i_new, gen_hr_i_clip)
                    del gen_hr_i,fileIndex_i,gentrans_i,gen_hr_i_clip,geotrans_i_new
    def accessSRImgQuality_CNL(self,targetImgpath,predImgpath,outpath,year):
        viirs_inData = read_imgdata(targetImgpath)
        chen_inData = read_imgdata(predImgpath)
        viirs_mask = np.where(viirs_inData > 1, viirs_inData, 0)
        chen_mask = np.where(viirs_inData > 1, chen_inData, 0)
        imgQA = ImageQualityAccess()
        quality_metrics = imgQA.cal_wholeImg_QI(viirs_mask, chen_mask, data_range=1000.)
        quality_metrics['region'] = self.regionname
        quality_metrics['model'] = 'CNL'
        quality_metrics['year'] = year
        imgQA.save_regionImgQI_toCSV_whole(outpath, [quality_metrics])
    def accessSRImgQuality_modelGenHr(self,yearlst,modelLst,outpath):
        imgQA = ImageQualityAccess()
        indices_lst = []
        for year in yearlst:
            viirsTifpath = os.path.join(self.regiondir, 'data', self.regionname + "_VNL" + str(year) + ".tif")
            viirs_inData = read_imgdata(viirsTifpath)
            viirs_mask = np.where(viirs_inData > 1, viirs_inData, 0)
            for j in range(len(modelLst)):
                model = modelLst[j]
                gen_hrTifpath = os.path.join(self.regiondir, 'output', self.regionname + "_" + model + ".tif")
                genHr_inData = read_imgdata(gen_hrTifpath)
                genHr_mask = np.where((viirs_inData > 1)&(genHr_inData>0), genHr_inData, 0)
                genHr_inDs = gdal.Open(gen_hrTifpath)
                rasterOp.outputResult(genHr_inDs,genHr_mask,gen_hrTifpath)
                quality_metrics = imgQA.cal_wholeImg_QI(viirs_mask, genHr_mask, data_range=1000.)
                quality_metrics['region'] = self.regionname
                quality_metrics['model'] = model
                quality_metrics['year'] = year
                indices_lst.append(quality_metrics)
        imgQA.save_regionImgQI_toCSV_whole(outpath, indices_lst)
    def accessSRImgQuality_Bins(self,targetImgpath,predImgpath,outpath,model,year,viirsBins=[]):
        viirs_inData = read_imgdata(targetImgpath)
        viirs_mask = np.where(viirs_inData > 1, viirs_inData, 0)
        inData = read_imgdata(predImgpath)
        data_mask = np.where(viirs_inData > 0, inData, 0)
        imgQA = ImageQualityAccess()
        indices_lst = []
        for k in range(0, len(viirsBins) + 1):
            if k == 0:
                mask_loc = (viirs_mask < viirsBins[k])
                strtype = "bins 0 - " + str(viirsBins[k])
            elif k == len(viirsBins):
                mask_loc = (viirs_mask >= viirsBins[k - 1])
                strtype = "bins gt " + str(viirsBins[k - 1])
            else:
                mask_loc = ((viirs_mask >= viirsBins[k - 1]) & (viirs_mask < viirsBins[k]))
                strtype = "bins " + str(viirsBins[k - 1]) + "-" + str(viirsBins[k])

            label_i_mask_mask = np.where(mask_loc, viirs_mask, 0)
            geh_hr_i_mask = np.where(mask_loc, data_mask, 0)
            genhr_qualityAccess = imgQA.image_quality_assessment(label_i_mask_mask, geh_hr_i_mask,
                                                                 data_range=1000., mask=mask_loc)
            genhr_qualityAccess['bins'] = strtype
            genhr_qualityAccess['region'] = self.regionname
            genhr_qualityAccess['model'] =model
            genhr_qualityAccess['year'] = year
            indices_lst.append(genhr_qualityAccess)
        imgQA.save_regionImgQI_toCSV_bins(outpath, indices_lst)
    def accessSRImgQuality(self,yearlst,modelLst,outpath,isWholeQuality=True,viirsBins=[],saveIndics=True):
        imgQA = ImageQualityAccess()
        indices_lst = []
        for year in yearlst:
            viirsTifpath = os.path.join(self.regiondir,'data',self.regionname+"_VNL"+str(year)+".tif")
            viirs_inData = read_imgdata(viirsTifpath)
            viirs_mask = np.where(viirs_inData>1,viirs_inData,0)
            if year == 2012:
                chenTifpath = os.path.join(self.regiondir, 'data', self.regionname + "_chen" + str(year) + ".tif")
                chen_inData = read_imgdata(chenTifpath)
                chen_mask = np.where(viirs_inData>1,chen_inData,0)
            for j in range(len(modelLst)):
                modelinfo = modelLst[j]
                model = modelinfo[0]+"_"+modelinfo[1]+"_"+str(modelinfo[2])+"_"+str(year)
                gen_hrTifpath = os.path.join(self.regiondir,'output',self.regionname+"_"+model+".tif")
                genHr_inData = read_imgdata(gen_hrTifpath)
                genHr_mask = np.where(viirs_inData > 1, genHr_inData, 0)
                #质量评价
                if isWholeQuality:
                    quality_metrics = imgQA.image_quality_assessment(viirs_mask, genHr_mask, data_range=1000.)
                    quality_metrics['type'] = 'srNTL'
                    quality_metrics['region'] = self.regionname
                    quality_metrics['model'] = modelinfo[0]+"_"+modelinfo[1]+"_"+str(modelinfo[2])
                    quality_metrics['year'] = year
                    indices_lst.append(quality_metrics)
                    # 分段汇总
                if len(viirsBins) > 0:
                    for k in range(0, len(viirsBins) + 1):
                        if k == 0:
                            mask_loc = (viirs_mask < viirsBins[k])
                            strtype = "bins 0 - " + str(viirsBins[k])
                        elif k == len(viirsBins):
                            mask_loc = (viirs_mask >= viirsBins[k - 1])
                            strtype = "bins gt " + str(viirsBins[k - 1])
                        else:
                            mask_loc = ((viirs_mask >= viirsBins[k - 1]) & (viirs_mask < viirsBins[k]))
                            strtype = "bins " + str(viirsBins[k - 1]) + "-" + str(viirsBins[k])

                        label_i_mask_mask = np.where(mask_loc, viirs_mask, 0)
                        geh_hr_i_mask = np.where(mask_loc, genHr_mask, 0)
                        genhr_qualityAccess = imgQA.image_quality_assessment(label_i_mask_mask, geh_hr_i_mask,
                                                                        data_range=1000., mask=mask_loc)
                        genhr_qualityAccess['type'] = 'srNTL'
                        genhr_qualityAccess['bins'] = strtype
                        genhr_qualityAccess['region'] = self.regionname
                        genhr_qualityAccess['model'] =  modelinfo[0]+"_"+modelinfo[1]+"_"+str(modelinfo[2])
                        genhr_qualityAccess['year'] = year
                        indices_lst.append(genhr_qualityAccess)
                        # if (j == 0)&(year==2012):
                        #     chen_i_mask = np.where(mask_loc, chen_mask, 0)
                        #     chen_qualityAccess = imgQA.image_quality_assessment(label_i_mask_mask, chen_i_mask,
                        #                                                         data_range=1000., mask=mask_loc)
                        #     chen_qualityAccess['type'] = 'CNL'
                        #     chen_qualityAccess['bins'] = strtype
                        #     chen_qualityAccess['region'] = self.regionname
                        #     chen_qualityAccess['model'] = 'CNL'
                        #     chen_qualityAccess['year'] = year
                        #     indices_lst.append(chen_qualityAccess)
        # 保存
        if saveIndics:
            imgQA.save_regionImgQI_toCSV_bins(outpath, indices_lst)
        return indices_lst

    def regionImgAnalysis(self):
        vnlpath = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\out\01\data\BJH_VNL2012.tif'
        viirs_inData = read_imgdata(vnlpath)
        viirs_mask = np.where(viirs_inData > 1, viirs_inData, 0)
        # 每个区间段的散点图
        genpath = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\out\01\data\BJH_DNL2012.tif'
        # genpath = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\out\01\out\BJH_DNLSRNet_ds_01_2824_2012.tif'
        genHr_inData = read_imgdata(genpath)
        genHr_mask = np.where(viirs_inData > 1, genHr_inData, 0)
        binlst = list(range(0, 110, 10))
        anbinlst = [150, 200, 250, 300]
        binlst.extend(anbinlst)
        figure = plt.figure(figsize=(20, 9))
        for i in range(len(binlst) - 1):
            plt.subplot(4, 4, i + 1)
            minVlue = binlst[i]
            maxValue = binlst[i + 1]
            viirsvalues = viirs_mask[(viirs_mask > minVlue) & (viirs_mask <= maxValue)]
            genValues = genHr_mask[(viirs_mask > minVlue) & (viirs_mask <= maxValue)]
            plt.scatter(viirsvalues, genValues)
            if minVlue < 100:
                xticks = range(minVlue, maxValue + 2, 1)
            else:
                xticks = range(minVlue, maxValue + 5, 5)
            plt.xticks(xticks)
            plt.title(str(minVlue) + "-" + str(maxValue))
            plt.ylim(0, 100)
        plt.subplot(4, 4, 15)
        viirsvalues = viirs_mask[(viirs_mask > binlst[-1])]
        genValues = genHr_mask[(viirs_mask > binlst[-1])]
        plt.scatter(viirsvalues, genValues)
        plt.title("gt" + str(binlst[-1]))
        plt.ylim(0, 100)

        # 每个区间段的直方图
        binlst = [0, 3, 7, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100]
        gendir = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\out\01\out'
        outdir = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\out\01\plt\03hist'
        gennamelst = []
        # gennamelst.append('DNL2012')
        # gennamelst.append('AE_gradient_loss_-3')
        # gennamelst.append('DNLSRNet_01_85_2012')
        # gennamelst.append('DNLSRNet_01_170_2012')
        # gennamelst.append('DNLSRNet_01_225_2012')
        # gennamelst.append('DNLSRNet_01_400_2012')
        # gennamelst.append('DNLSRNet_01_536_2012')
        # gennamelst.append('DNLSRNet_01_753_2012')
        # gennamelst.append('DNLSRNet_01_977_2012')
        # gennamelst.append('EDSR_01_lr4_250')
        # gennamelst.append('EDSR_01_lr4_2702')
        # gennamelst.append('chen2012')
        # gennamelst.append('EDSR_01_lr4_noln_3500')
        for genname in gennamelst:
            genpath = os.path.join(gendir, 'BJH_' + genname + '.tif')
            genHr_inDs, genHr_inBand, genHr_inData, genHr_noValue = rasterOp.getRasterData(genpath)
            genHr_mask = np.where(viirs_inData > 0, genHr_inData, 0)
            figure = plt.figure(figsize=(16, 9))
            for i in range(0, len(binlst)):
                plt.subplot(4, 4, i + 1)
                if i < len(binlst) - 1:
                    minVlue = binlst[i]
                    maxValue = binlst[i + 1]
                    gendata = genHr_mask[(viirs_mask > minVlue) & (viirs_mask < maxValue)]
                    plt.title(str(minVlue) + "-" + str(maxValue))
                else:
                    gendata = genHr_mask[(viirs_mask >= binlst[-1])]
                    plt.title("gt" + str(maxValue))
                plt.hist(gendata, bins=200)
            plt.subplots_adjust(left=0.04, top=0.96, right=0.96, bottom=0.04, wspace=0.3, hspace=0.3)
            outpath = os.path.join(outdir, genname + '.png')
            plt.savefig(outpath)
            plt.close()

        # DNL
        genpath = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\out\01\data\BJH_DNL2012.tif'
        genHr_inDs, genHr_inBand, genHr_inData, genHr_noValue = rasterOp.getRasterData(genpath)
        genHr_mask = np.where(viirs_inData > 0, genHr_inData, 0)
        binlst = list(range(0, 70, 10))
        figure = plt.figure(figsize=(20, 9))
        for i in range(len(binlst) - 1):
            plt.subplot(2, 4, i + 1)
            minVlue = binlst[i]
            maxValue = binlst[i + 1]
            maskdata = (genHr_mask >= minVlue) & (genHr_mask < maxValue)
            viirsvalues = viirs_mask[maskdata]
            genValues = genHr_mask[maskdata]
            plt.scatter(genValues, viirsvalues)
            xticks = range(minVlue, maxValue, 1)
            plt.xticks(xticks)
            plt.title(str(minVlue) + "-" + str(maxValue))
            # plt.ylim(0,100)

        # 每个区间的中值 散点图
        binlst = list(range(0, 100, 2))
        anbinlst = list(range(110, 300, 50))
        binlst.extend(anbinlst)
        gendir = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\out\01\out'
        outdir = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\out\01\plt\02mean'
        gennamelst = []
        gennamelst.append('chen2012')
        for genname in gennamelst:
            genpath = os.path.join(gendir, 'BJH_' + genname + '.tif')
            genHr_inDs, genHr_inBand, genHr_inData, genHr_noValue = rasterOp.getRasterData(genpath)
            genHr_mask = np.where(viirs_inData > 0, genHr_inData, 0)
            xlst = []
            ylst = []
            stdlst = []
            lower_q_lst = []
            higher_q_lst = []
            int_r_lst = []
            for i in range(0, len(binlst)):
                if i < len(binlst) - 1:
                    minVlue = binlst[i]
                    maxValue = binlst[i + 1]
                    viirsdata = viirs_mask[(viirs_mask > minVlue) & (viirs_mask <= maxValue)]
                    gendata = genHr_mask[(viirs_mask > minVlue) & (viirs_mask <= maxValue)]
                else:
                    viirsdata = viirs_mask[(viirs_mask > binlst[-1])]
                    gendata = genHr_mask[(viirs_mask > binlst[-1])]
                if len(viirsdata) == 0:
                    xlst.append(0)
                    ylst.append(0)
                    stdlst.append(0)
                    lower_q_lst.append(0)
                    higher_q_lst.append(0)
                    int_r_lst.append(0)
                else:
                    xlst.append(np.mean(viirsdata))
                    ylst.append(np.mean(gendata))
                    stdlst.append(np.std(gendata))
                    lower_q = np.quantile(gendata, 0.25, interpolation='lower')  # 下四分位
                    hiegher_q = np.quantile(gendata, 0.75, interpolation='higher')  # 上四分位
                    lower_q_lst.append(lower_q)  # 四分位距
                    higher_q_lst.append(hiegher_q)
                    int_r_lst.append(hiegher_q - lower_q)
            figure = plt.figure(figsize=(16, 6))
            plt.subplot(1, 3, 1)
            plt.scatter(xlst, ylst, label='mean')
            plt.scatter(xlst, lower_q_lst, label='0.25')
            plt.scatter(xlst, higher_q_lst, label='0.75')
            plt.plot(xlst, xlst)
            plt.legend()
            plt.title('DN')
            plt.xlim(0, 110)
            plt.ylim(0, 80)
            plt.subplot(1, 3, 2)
            plt.scatter(xlst, int_r_lst)
            plt.plot(xlst, xlst)
            plt.title('interval')
            plt.xlim(0, 110)
            plt.ylim(0, 80)
            plt.subplot(1, 3, 3)
            plt.scatter(xlst, stdlst)
            plt.plot(xlst, xlst)
            plt.title('std')
            plt.xlim(0, 110)
            plt.ylim(0, 80)
            outpath = os.path.join(outdir, genname + '.png')
            plt.savefig(outpath)
            plt.close()

        # 找到使r2最大的数值分布
        num_lim_lst = np.arange(10, 501, 10)
        gendir = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\out\01\out'
        outdir = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\out\01\plt\04r2 2013'
        gennamelst = []
        gennamelst.append('chen2012')
        for genname in gennamelst:
            r2_max = 0
            r2_max_index = -1
            genpath = os.path.join(gendir, 'BJH_' + genname + '.tif')
            genHr_inDs, genHr_inBand, genHr_inData, genHr_noValue = rasterOp.getRasterData(genpath)
            genHr_mask = np.where(viirs_inData > 0, genHr_inData, 0)
            for i in range(0, len(num_lim_lst)):
                viirsdata = viirs_mask[(viirs_mask <= num_lim_lst[i])]
                gendata = genHr_mask[(viirs_mask <= num_lim_lst[i])]
                score = r2_score(viirsdata, gendata)
                print(genname, num_lim_lst[i], score)
                if score >= r2_max:
                    r2_max = score
                    r2_max_index = i
            if r2_max_index > -1:
                best_num_lim = num_lim_lst[r2_max_index]
            else:
                best_num_lim = 100
            gendata, viirsdata = genHr_mask[(viirs_mask <= best_num_lim)], viirs_mask[(viirs_mask <= best_num_lim)]
            figure = plt.figure(figsize=(6, 6))
            plt.scatter(gendata, viirsdata)
            plt.title("genhr ")
            plt.xlabel('predict DNL')
            plt.ylabel('target DNL')
            plt.ylim(0, best_num_lim)
            plt.xlim(0, best_num_lim)
            plt.xticks(np.arange(0, best_num_lim + 1, 10))
            plt.yticks(np.arange(0, best_num_lim + 1, 10))
            score = r2_score(viirs_mask, genHr_mask)
            plt.text(best_num_lim // 2, best_num_lim + 8, "R2 : %0.4f" % score, fontdict={'size': '10'})
            plt.text(best_num_lim // 2, best_num_lim + 5, "R2 (DN<=%d): %0.4f" % (best_num_lim, r2_max),
                     fontdict={'size': '10'})
            outpath = os.path.join(outdir, genname + '.png')
            plt.savefig(outpath)
            plt.close()

srRegionAccess = SRImgRegionAccess()
# #裁剪区域图像，用于模型预测
# srRegionAccess.clip_toGet_input_Data()
# #生成区域超分图像
# srRegionAccess.generate_single_SRImg_DiffMls()
#自定义条件生成超分图像
# srRegionAccess.userDefined_generate_single_SRImg_ByNDVI()
#预测其他年份数据
# srRegionAccess.generate_single_SRImg_DiffYears()





'''全球尺度评价'''
class SRImgWorldAccess():
    def __init__(self):
        self.rootdir = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\world'
        # self.IdsTxt = r'/home/zju/cxx/NTL_timeserise/data/SRCNN/data/world/inputs/img/grid/Id.txt'
        # self.ntl2012 = r'/home/zju/cxx/NTL_timeserise/data/SRCNN/data/world/inputs/img/2012_oriDNL'
        # self.ntl2013 = r'/home/zju/cxx/NTL_timeserise/data/SRCNN/data/world/inputs/img/2013_oriDNL'
        # self.vnl2012 = r'/home/zju/cxx/NTL_timeserise/data/SRCNN/data/world/inputs/img/2012_VNL'
        # self.vnl2013 = r'/home/zju/cxx/NTL_timeserise/data/SRCNN/data/world/inputs/img/2013_VNL'
        # self.imgdir = r'/home/zju/cxx/NTL_timeserise/data/SRCNN/data/world/inputs/img'
        self.imgdir = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\world\data\img'
        self.resultdir = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\world\result'
        self.srGenDataRootDir = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\world\result'

    def getPath(self,year):
        self.year = year
        self.dnl_dir = os.path.join(self.imgdir,'oriDNL',str(year))
        self.ndvi_dir = os.path.join(self.imgdir, 'NDVI', str(year))
        self.waterdir = os.path.join(self.imgdir, 'Water_Aridity')
        if str(year) in ('2012','2013'):
            self.vnl_dir = os.path.join(self.imgdir, 'VNL', str(year))
        else:
            self.vnl_dir = os.path.join(self.imgdir, 'VNL', '2012')
    def createSRImg(self,modelinfo,year,idtxtpath,batchsize=30,clipPadding=30,isCPU=True,config=None,dmsp_indir="",viirs_indir="",water_indir="",otherdata_dict=None):
        '''根据神经网络训练模型生成超分图像patches'''
        modelname, modelappendix, epoch, otherdataNames = modelinfo
        otherdataNamelst = otherdataNames.split(";")
        if config is None:
            config = getconfig(modelname,modelappendix,epoch)
        if isCPU:
            device = torch.device('cpu')
            config.device = device
        net = getNet_fromModel(config, epoch)
        outdir = os.path.join(self.resultdir,str(year),'patches_'+modelname+"_"+modelappendix+"_"+str(epoch)+"_ClipPd"+str(clipPadding))
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        #路径
        self.getPath(year)
        if dmsp_indir != "":
            self.dnl_dir = dmsp_indir
        if viirs_indir !="":
            self.vnl_dir = viirs_indir
        if water_indir !="":
            self.waterdir = water_indir
        if otherdata_dict is None:
            otherdata_dict = {}
            for name in otherdataNamelst:
                if name == "NDVI":
                    stat_dict = config.otherdata_dict['NDVI']
                    otherpath = self.ndvi_dir
                    stat_dict['path'] = otherpath
                    otherdata_dict['NDVI'] = stat_dict
                elif name == 'Water_Aridity':
                    stat_dict = config.otherdata_dict['Water_Aridity']
                    otherpath = self.waterdir
                    stat_dict['path'] = otherpath
                    otherdata_dict['Water_Aridity'] = stat_dict
            self.otherdata_dict = otherdata_dict
        else:
            self.otherdata_dict = otherdata_dict
        #run
        dataset = Dataset_oriDMSP(config,idtxtpath,'valid',self.vnl_dir,self.dnl_dir,self.waterdir,self.otherdata_dict,is_multi_years_combined=False)
        dataloader = torch.utils.data.DataLoader(dataset,batch_size=batchsize,shuffle=False,drop_last=False)
        files = dataset.files
        # 投影信息
        tifpath = os.path.join(self.vnl_dir, files[0] + ".tif")
        geoProj = rasterOp.getProj_img(tifpath)
        #create sr img
        net.eval()
        with torch.no_grad():
            for idx, out_dict in enumerate(dataloader):
                gen_hr = generateSISR(config,net,out_dict)
                # 用归一化前的数据计算损失，输出  label 为原始归一化前数据
                outdata = outdata_transform_qualityassess(config, gen_hr)
                #转cpu
                outdata = outdata.cpu()
                viirsdata = out_dict['viirsdata_ori'].cpu()
                dmspdata = out_dict['dmspdata_ori_inter'].cpu()
                # if jj==0:
                #     break
                #输出图像
                batchnum = len(gen_hr)
                fileIndex = out_dict['index']
                for i in range(batchnum):
                    gen_hr_i = outdata[i].data.detach().numpy()[0]
                    viirs_i = viirsdata[i].data.detach().numpy()[0]
                    dmsp_i = dmspdata[i].data.detach().numpy()[0]
                    fileIndex_i = fileIndex[i].data.detach().numpy()[0]
                    image_id = files[fileIndex_i]
                    viirspath = os.path.join(self.vnl_dir,image_id + ".tif")
                    viirsDs = gdal.Open(viirspath)
                    gentrans_i = viirsDs.GetGeoTransform()
                    print(image_id)
                    #输出图像
                    # saveTifpath = os.path.join(outdir,image_id+'_nomask.tif')
                    # if not os.path.exists(saveTifpath):
                    #     rasterOp.write_img(saveTifpath, geoProj, gentrans_i, gen_hr_i)
                    saveTifpath_clip = os.path.join(outdir,image_id+"_cp"+str(clipPadding)+".tif")
                    if not os.path.exists(saveTifpath_clip):
                        #mask
                        maskloc = (viirs_i<=1) | (dmsp_i==0)
                        gen_hr_i[maskloc]=0
                        #输出裁剪图像
                        h,w = gen_hr_i.shape
                        gen_hr_i_clip = gen_hr_i[clipPadding:h-clipPadding,clipPadding:w-clipPadding]
                        xmin_new,ymax_new = rowcol_to_xy(gentrans_i,clipPadding,clipPadding)
                        geotrans_i_new = list(gentrans_i)
                        geotrans_i_new[0] = xmin_new
                        geotrans_i_new[3] = ymax_new
                        rasterOp.write_img(saveTifpath_clip, geoProj, geotrans_i_new, gen_hr_i_clip)
                        del gen_hr_i_clip,geotrans_i_new,
                    del viirsDs,gen_hr_i,fileIndex_i,gentrans_i,dmsp_i
                    gc.collect()
                del gen_hr,outdata,viirsdata,dmspdata,out_dict
    def createSRImg_run(self):
        # device_ids = [0,1]
        # device_ids = [0]
        # torch.cuda.set_device(0)

        isCpu = False
        batchsize = 1
        clippad = 30

        modelinfoLst = []
        modelinfoLst.append(('UNet','labelNoProcess_12_13',1571,'NDVI'))
        # modelinfoLst.append(('UNet','03_img04',2975,'NDVI'))
        # modelinfoLst.append(('UNet', '01_gradient_urbanMask', 1525, 'NDVI'))
        # modelinfoLst.append(('UNet', '01_rarid_class', 2550, 'NDVI;Water_Aridity'))
        # for year in range(1992,2000,1):
        for year in range(2018, 2019, 1):
            idTxtpathlst = []
            idTxtpathlst.append(r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\world\data\shp\grid\grids_04.txt')
            idTxtpathlst.append(r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\world\data\shp\grid\grids_03.txt')
            idTxtpathlst.append(r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\world\data\shp\grid\grids_02.txt')
            idTxtpathlst.append(r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\world\data\shp\grid\grids_01.txt')
            for modelinfo in modelinfoLst:
                for idTxtpath in idTxtpathlst:
                    print(year,modelinfo,idTxtpath)
                    t1 = datetime.now()
                    self.createSRImg(modelinfo,year,idTxtpath,batchsize=batchsize,clipPadding=clippad,isCPU=isCpu)
                    t2 = datetime.now()
                    print(year,'starttime:',t1,' endtime:',t2,' lasting time:',t2-t1)

    def createImgIDtxt_noGenHr(self,inpath,outpath,modelname,modelappendix,epoch,year,clipPadding=50):
        '''将未生成超分图像的图像id单独整理成一个txt。有时出现意外导致只有部分图像块生成了超分图像。'''
        file_list = tuple(open(inpath, "r"))
        file_list = [id_.rstrip() for id_ in file_list]
        with open(outpath,'a') as f:
            for fi in file_list:
                outdir = os.path.join(self.resultdir, str(year),
                                      'patches_' + modelname + "_" + modelappendix + "_" + str(epoch))
                saveTifpath = os.path.join(outdir, fi + '_nomask.tif')
                saveTifpath_clip = os.path.join(outdir, fi + "_p" + str(clipPadding) + "_masked.tif")
                if (not os.path.exists(saveTifpath))|(not os.path.exists(saveTifpath_clip)):
                    print(fi)
                    f.write(fi+'\n')
    def extractMultiValueToPoints_Continent(self,pointShpPath,regionName,year,outCsvPath,PID):
        '''根据随机点文件提取对应地栅格像元值'''
        cols = ['PID','Region','Year','VNL','srNL','CNL']
        rasNamelst = []
        raspathlst = []
        raspathlst.append(os.path.join(self.imgdir,'regionImg','VNL_'+str(year)+regionName+'.tif'));rasNamelst.append('VNL')
        raspathlst.append(os.path.join(self.srGenDataRootDir, str(year),'srWorldImg', 'srNTL_' + str(year) + regionName + '.tif'));rasNamelst.append('srNL')
        if year == 2012:
            raspathlst.append(os.path.join(self.imgdir, 'regionImg', 'CNL_' + str(year) + regionName + '.tif'));rasNamelst.append('CNL')
        result_df = getRasterValueByPoints(pointShpPath,"",raspathlst,rasNamelst,[PID])
        df = pd.DataFrame(columns=cols)
        if (os.path.isfile(outCsvPath) == False):
            df.to_csv(outCsvPath, header=True, index=False)
        df['PID'] = result_df[PID]
        df['Region'] = regionName
        df['Year'] = year
        df['VNL'] = result_df['VNL']
        df['srNL'] = result_df['srNL']
        if year == 2012:
            df['CNL'] = result_df['CNL']
        df.to_csv(outCsvPath, mode='a', header=False, index=False)
    def extractMultiValueToPolygon_Continent(self,polyShpPath,regionName,fidCsvPath,outCsvPath):
        fidcol = 'PID';regioncol = 'Region';nationcol = 'nation';citycol = 'city';yearcol = 'Year'
        cols1 = [fidcol,regioncol,nationcol,citycol,yearcol]
        cols2 = ['VNL','srNL','CNL']
        cols1.extend(cols2)
        df = pd.DataFrame(columns=cols1)
        if (os.path.isfile(outCsvPath) == False):
            df.to_csv(outCsvPath, header=True, index=False)
        dir1 = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\world\inputs\img\regionImg'
        dir2 = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\world\result\2012\srWorldImg'
        dir3 = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\world\result\2013\srWorldImg'
        dataInfos = []
        dataInfos.append(('CNL',dir1,2012))
        dataInfos.append(('VNL', dir1, 2012))
        dataInfos.append(('VNL', dir1, 2013))
        dataInfos.append(('srNTL', dir3, 2013))
        dataInfos.append(('srNTL' , dir2, 2012))
        ras_pathLst = []
        for i in range(len(dataInfos)):
            dataname,dir,year = dataInfos[i]
            path = os.path.join(dir,'prj_'+dataname+'_'+str(year)+regionName+'.tif')
            ras_pathLst.append(path)
        fid_df = pd.read_csv(fidCsvPath)
        t1 = datetime.now()
        print('satrt time:',t1)
        for row in fid_df.itertuples():
            result_dic1 = {}
            result_dic2 = {}
            fid = getattr(row, fidcol)
            nation = getattr(row,nationcol)
            city = getattr(row, citycol)
            result_dic1[fidcol] = fid
            result_dic1[nationcol] = nation
            result_dic1[citycol] = city
            result_dic2[fidcol] = fid
            result_dic2[nationcol] = nation
            result_dic2[citycol] = city
            print(fid)
            sql = fidcol+' = '+str(fid)
            # stat_result = sumOfDN_RasterValue_maskByPolygon(polyShpPath, ras_pathLst, sql)  #速度太慢了，换下面的
            stat_result = sumOfDN_RasterValue_maskByPolygon_cropToCutline_True(polyShpPath, ras_pathLst, sql)
            for i in range(len(ras_pathLst)):
                dataname,dir,year = dataInfos[i]
                value = stat_result[i]
                if year == 2012:
                    if dataname == 'srNTL':
                        result_dic1['srNL'] = value
                    else:
                        result_dic1[dataname]=value
                elif year == 2013:
                    if dataname == 'srNTL':
                        result_dic2['srNL'] = value
                    else:
                        result_dic2[dataname] = value
            result_dic1[yearcol] = 2012
            result_dic2[yearcol] = 2013
            result_dic1[regioncol] = regionName
            result_dic2[regioncol] = regionName
            result_dict_lst = []
            result_dict_lst.append(result_dic1)
            result_dict_lst.append(result_dic2)
            df = pd.DataFrame(data=result_dict_lst)
            df = df[cols1]
            df.to_csv(outCsvPath, mode='a', header=False, index=False)
        t2 = datetime.now()
        print('end time:',t2,'total seconds:',t2-t1)
    def extractMultiValueToProvince_Continent(self,polyShpPath,regionName,fidCsvPath,outCsvPath):
        fidcol = 'PID';regioncol = 'Region';nationcol = 'nation';citycol = 'province';yearcol = 'Year'
        cols1 = [fidcol,regioncol,nationcol,citycol,yearcol]
        cols2 = ['VNL','srNL','CNL']
        cols1.extend(cols2)
        df = pd.DataFrame(columns=cols1)
        if (os.path.isfile(outCsvPath) == False):
            df.to_csv(outCsvPath, header=True, index=False)
        dir1 = os.path.join(self.imgdir,'regionImg')
        dir2 = os.path.join(self.resultdir,'2012','srWorldImg')
        dir3 = os.path.join(self.resultdir,'2013','srWorldImg')
        dataInfos = []
        dataInfos.append(('CNL',dir1,2012))
        dataInfos.append(('VNL', dir1, 2012))
        dataInfos.append(('VNL', dir1, 2013))
        dataInfos.append(('srNTL', dir3, 2013))
        dataInfos.append(('srNTL' , dir2, 2012))
        ras_pathLst = []
        for i in range(len(dataInfos)):
            dataname,dir,year = dataInfos[i]
            path = os.path.join(dir,'prj_'+dataname+'_'+str(year)+regionName+'.tif')
            ras_pathLst.append(path)
        fid_df = pd.read_csv(fidCsvPath)
        t1 = datetime.now()
        print('satrt time:',t1)
        for row in fid_df.itertuples():
            result_dic1 = {}
            result_dic2 = {}
            fid = getattr(row, fidcol)
            nation = getattr(row,nationcol)
            city = getattr(row, citycol)
            result_dic1[fidcol] = fid
            result_dic1[nationcol] = nation
            result_dic1[citycol] = city
            result_dic2[fidcol] = fid
            result_dic2[nationcol] = nation
            result_dic2[citycol] = city
            print(fid)
            sql = fidcol+' = '+str(fid)
            # stat_result = sumOfDN_RasterValue_maskByPolygon(polyShpPath, ras_pathLst, sql)  #速度太慢了，换下面的
            stat_result = sumOfDN_RasterValue_maskByPolygon_cropToCutline_True(polyShpPath, ras_pathLst, sql)
            for i in range(len(ras_pathLst)):
                dataname,dir,year = dataInfos[i]
                value = stat_result[i]
                if year == 2012:
                    if dataname == 'srNTL':
                        result_dic1['srNL'] = value
                    else:
                        result_dic1[dataname]=value
                elif year == 2013:
                    if dataname == 'srNTL':
                        result_dic2['srNL'] = value
                    else:
                        result_dic2[dataname] = value
            result_dic1[yearcol] = 2012
            result_dic2[yearcol] = 2013
            result_dic1[regioncol] = regionName
            result_dic2[regioncol] = regionName
            result_dict_lst = []
            result_dict_lst.append(result_dic1)
            result_dict_lst.append(result_dic2)
            df = pd.DataFrame(data=result_dict_lst)
            df = df[cols1]
            df.to_csv(outCsvPath, mode='a', header=False, index=False)
        t2 = datetime.now()
        print('end time:',t2,'total seconds:',t2-t1)
    def extractMultiValueToNation_Continent(self,polyShpPath,regionName,fidCsvPath,outCsvPath):
        fidcol = 'PID';regioncol = 'Region';nationcol = 'nation';yearcol = 'Year'
        cols1 = [fidcol,regioncol,nationcol,yearcol]
        cols2 = ['VNL','srNL','CNL']
        cols1.extend(cols2)
        df = pd.DataFrame(columns=cols1)
        if (os.path.isfile(outCsvPath) == False):
            df.to_csv(outCsvPath, header=True, index=False)
        dir1 = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\world\inputs\img\regionImg'
        dir2 = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\world\result\2012\srWorldImg'
        dir3 = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\world\result\2013\srWorldImg'
        dataInfos = []
        dataInfos.append(('CNL',dir1,2012))
        dataInfos.append(('VNL', dir1, 2012))
        dataInfos.append(('VNL', dir1, 2013))
        dataInfos.append(('srNTL', dir3, 2013))
        dataInfos.append(('srNTL' , dir2, 2012))
        ras_pathLst = []
        for i in range(len(dataInfos)):
            dataname,dir,year = dataInfos[i]
            path = os.path.join(dir,'prj_'+dataname+'_'+str(year)+regionName+'.tif')
            ras_pathLst.append(path)
        fid_df = pd.read_csv(fidCsvPath)
        t1 = datetime.now()
        print('satrt time:',t1)
        for row in fid_df.itertuples():
            result_dic1 = {}
            result_dic2 = {}
            fid = getattr(row, fidcol)
            nation = getattr(row,nationcol)
            result_dic1[fidcol] = fid
            result_dic1[nationcol] = nation
            result_dic2[fidcol] = fid
            result_dic2[nationcol] = nation
            print(fid)
            sql = fidcol+' = '+str(fid)
            # stat_result = sumOfDN_RasterValue_maskByPolygon(polyShpPath, ras_pathLst, sql)  #速度太慢了，换下面的
            stat_result = sumOfDN_RasterValue_maskByPolygon_cropToCutline_True(polyShpPath, ras_pathLst, sql)
            for i in range(len(ras_pathLst)):
                dataname,dir,year = dataInfos[i]
                value = stat_result[i]
                if year == 2012:
                    if dataname == 'srNTL':
                        result_dic1['srNL'] = value
                    else:
                        result_dic1[dataname]=value
                elif year == 2013:
                    if dataname == 'srNTL':
                        result_dic2['srNL'] = value
                    else:
                        result_dic2[dataname] = value
            result_dic1[yearcol] = 2012
            result_dic2[yearcol] = 2013
            result_dic1[regioncol] = regionName
            result_dic2[regioncol] = regionName
            result_dict_lst = []
            result_dict_lst.append(result_dic1)
            result_dict_lst.append(result_dic2)
            df = pd.DataFrame(data=result_dict_lst)
            df = df[cols1]
            df.to_csv(outCsvPath, mode='a', header=False, index=False)
        t2 = datetime.now()
        print('end time:',t2,'total seconds:',t2-t1)
    def statR2_concat(self,srcCSVPath,appendCSVPath,outpath,commonCols,appendColname):
        # srcCSVPath = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\world\inputs\r2Valid_loc\nation\stat_nationTNL.csv'
        # appendCSVPath = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\world\inputs\r2Valid_loc\nation\DNLSRNet_2012_01_2000_stats_nation.csv'
        # outpath = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\world\inputs\r2Valid_loc\stat\r2_nation.csv'
        # commonCols = ['PID','Region','Year','nation']
        # appendColname = ['DNLSRNet_2012_01_2000']
        # srcDF = pd.read_csv(srcCSVPath)
        # appendDF = pd.read_csv(appendCSVPath)
        # newdf = pd.merge(srcDF, appendDF, how='left', on=commonCols)
        # newdf.to_csv(outpath,header=True,index=False)

        srcCSVPath = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\world\inputs\r2Valid_loc\province\stat_provinceTNL.csv'
        appendCSVPath = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\world\inputs\r2Valid_loc\province\DNL_2012_stats_province_DNL.csv'
        outpath = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\world\inputs\r2Valid_loc\province\append_DNL.csv'
        commonCols = ['PID', 'Region', 'Year', 'nation']
        appendColname = ['DNLSRNet_2012_01_2000']
        srcDF = pd.read_csv(srcCSVPath)
        appendDF = pd.read_csv(appendCSVPath)
        newdf = pd.merge(srcDF, appendDF, how='left', on=commonCols)
        newdf.to_csv(outpath, header=True, index=False)

    def extractPointValue(self,modellst,cols,year = '2012'):
        regionlst = []
        # regionlst.append('Africa')
        regionlst.append('Asia')
        regionlst.append('Europe')
        regionlst.append('NorthAmerica')
        regionlst.append('Oceania')
        regionlst.append('SouthAmerica')
        sql = "1=1"
        fieldLst = ['id']
        outdir = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\world\analysis\zonalStat\csv'
        shp_dir = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\world\data\shp\zone\point'
        vnl_dir = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\world\data\img\zone\VNL'
        cnl_dir = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\world\data\img\zone\CNL'
        srDNL_dir = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\world\result'
        outpath = os.path.join(outdir,year+'_point_prj.csv')

        for region in regionlst:
            print(region)
            shp_path = os.path.join(shp_dir,region+'_point.shp')
            ras_pathLst = [];  rasNameLst = []
            ras_pathLst.append(os.path.join(vnl_dir,year+'_prj_'+region+'_VNL.tif'));rasNameLst.append('VNL');
            if year == '2012':
                ras_pathLst.append(os.path.join(cnl_dir, year + '_prj_' + region + '_CNL.tif'));rasNameLst.append('CNL');
            for modelname in modellst:
                ras_pathLst.append(os.path.join(srDNL_dir,year,modelname,year+'_prj_'+region+'_'+modelname+'.tif' ));rasNameLst.append(modelname);
            result_df = getRasterValueByPoints(shp_path, sql, ras_pathLst, rasNameLst, fieldLst)
            result_df = result_df[cols]
            if os.path.exists(outpath):
                result_df.to_csv(outpath,header=False,index=False,mode='a')
            else:
                result_df.to_csv(outpath, header=True, index=False, mode='a')

# srWorld = SRImgWorldAccess()
# srWorld.createSRImg_run()

# ############extractValueByPoints
# modellst = []
# modellst.append('UNet_01_gradient_urbanMask_1525')
# modellst.append('UNet_01_rarid_class_2550')
# modellst.append('UNet_03_img04_2975')
# modellst.append('UNet_labelNoProcess_12_13_1571')
# cols = ['id',	'VNL',	'CNL',	'UNet_01_gradient_urbanMask_1525',	'UNet_01_rarid_class_2550',	'UNet_03_img04_2975',	'UNet_labelNoProcess_12_13_1571']
# yearlst = []
# yearlst.append('2012')
# # yearlst.append('2013')
# for year in yearlst:
#     srWorld.extractPointValue(modellst,cols,year)

class NDVIMissingData_Analysis():
    def AVHRR_NDVI(self):
        rootdir = r'D:\01data\00整理\04NDVI\NDVImissingArea\AVHRRNDVI'
        geonum = 3
        df = pd.DataFrame()
        for year in range(1992,2000):
            path = os.path.join(rootdir,'AVHRR_NDVI_'+str(year)+'_geo'+str(geonum)+'.tif')
            _,_,inData,_ = rasterOp.getRasterData(path)
            inData = np.where(np.isnan(inData)|np.isinf(inData)|np.isneginf(inData)|(inData<0),0,inData)
            df[str(year)] = inData.flatten()
        df.boxplot()
        # minvalue = np.min(inData)
        # maxvalue = np.max(inData)
        # meanvalue = np.mean(inData)
        # medianvalue = np.median(inData)
        # valueP25 = np.percentile(inData,25)
        # valueP75 = np.percentile(inData, 75)
        # stdvalue = np.std(inData)
        # varvalue = np.var(inData)
    def GHSL_POP(self,geonum):
        rootdir = r'D:\01data\00整理\04NDVI\NDVImissingArea\GHSL_POP\aggregate10000m'
        df = pd.DataFrame()
        for year in [1990,2000]:
            path = os.path.join(rootdir,'GHSL_POP_'+str(year)+'_geo'+str(geonum)+'.tif')
            _,_,inData,_ = rasterOp.getRasterData(path)
            inData = np.where(np.isnan(inData)|np.isinf(inData)|np.isneginf(inData)|(inData<0),0,inData)
            df[str(year)] = inData.flatten()
        plt.figure()
        df.boxplot()

        # plt.figure()
        # plt.scatter(df['1990'], df['2000'])
        # plt.xlabel('1990')
        # plt.ylabel('2000')
        # maxValue = max(df['2000'])
        # xlst = [0,5,maxValue]
        # plt.plot(xlst, xlst, color='r', linestyle=':')
        # score = r2_score(df['1990'], df['2000'])
        # plt.text(200000, 3500000, 'y = x')
        # plt.text(200000, 3000000, 'R2 = ' + "{:.3f}".format(score))
        # # plt.xlim(0,2000000)





def reload(path,outpath):
    filelst = tuple(open(path,'r'))
    filelst = [id_.rstrip() for id_ in filelst]
    for file in filelst:
        arr = file.split('_')
        a = int(arr[0])
        if a < 49:
            with open(outpath,'a+') as f:
                f.write(file+'\n')




