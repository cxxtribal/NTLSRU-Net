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
from commontools.common import FileOpr
from commontools.largeRasterImageOp import getRasterValueByPoints

Quality_Indices_Lst = ['MPSNR', 'MSSIM', 'RMSE', 'MAE', 'R2','TNLE','MRE','DNnum','r2_max','r2_max_range']

def modelloss_plot():
    rootpath = r'/home/zju/cxx/NTL_timeserise/data/SRCNN/net_02_0925/DNLSRNet'
    model = 'DNLSRNet'
    modelappendlst = ['01','02','ds_01']
    configlst = []
    for i in range(len(modelappendlst)):
        reloaded_config_path = os.path.join(rootpath,model+'_'+modelappendlst[i],'config','config.pickle')
        with open(reloaded_config_path, "rb") as config_f:
            config = pickle.load(config_f)
            configlst.append(config)
    #train loss
    figure = plt.figure()
    for i in range(len(modelappendlst)):
        plt.plot(configlst[i].avg_loss_G[300:],label=modelappendlst[i])
    plt.legend()
    plt.title('train loss')
    outpath = r'/home/zju/cxx/NTL_timeserise/trainloss.png'
    plt.savefig(outpath)
    #test loss
    figure = plt.figure()
    for i in range(len(modelappendlst)):
        plt.plot(configlst[i].test_avg_loss_G[300:],label=modelappendlst[i])
    plt.legend()
    plt.title('test loss')
    outpath = r'/home/zju/cxx/NTL_timeserise/testloss.png'
    plt.savefig(outpath)
    #rmse loss
    figure = plt.figure()
    for i in range(len(modelappendlst)):
        plt.plot(configlst[i].valid_quality_indices_dict['valid_05']['RMSE'][500:],label=modelappendlst[i])
    plt.legend()
    plt.title('RMSE')
    outpath = r'/home/zju/cxx/NTL_timeserise/RMSEloss1.png'
    plt.savefig(outpath)
    #MRE
    figure = plt.figure()
    for i in range(len(modelappendlst)):
        plt.plot(configlst[i].valid_quality_indices_dict['valid_05']['MRE'][300:],label=modelappendlst[i])
    plt.legend()
    plt.title('MRE')
    outpath = r'/home/zju/cxx/NTL_timeserise/MREloss.png'
    plt.savefig(outpath)
    #TNL
    figure = plt.figure()
    for i in range(len(modelappendlst)):
        plt.plot(configlst[i].valid_quality_indices_dict['valid_05']['TNL'][300:],label=modelappendlst[i])
    plt.legend()
    plt.title('TNL')
    outpath = r'/home/zju/cxx/NTL_timeserise/TNLloss.png'
    plt.savefig(outpath)


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
        x_true_1 = np.where(x_true==0,1,x_true) # x_true=0时不能为除数，所以把0的像元都替换为1.
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
    def __init__(self,imgrootdir="",valid_txt="",year="",otherdata_dict=None,saveCSV="",datarange=1000):
        self.imgrootdir = imgrootdir
        self.valid_txt = valid_txt
        self.year = year
        self.otherdata_dict = otherdata_dict
        self.saveCSV = saveCSV
        self.datarange = datarange
    def getPath(self):
        self.dnldir = os.path.join( self.imgrootdir,str(self.year)+'_oriDNL')
        self.vnldir = os.path.join( self.imgrootdir,str(self.year)+'_VNL')
        self.cnldir = os.path.join( self.imgrootdir,'2012_Chen')
        self.waterdir = os.path.join(self.imgrootdir, "Water_Aridity")
        self.dataname =os.path.split(self.imgrootdir)[-1]
        self.dataset = os.path.split(self.valid_txt)[-1].split('.')[0]

    def calCNL_QI(self):
        file_list = tuple(open(self.valid_txt, "r"))
        file_list = [id_.rstrip() for id_ in file_list]
        imgQA = ImageQualityAccess()
        for i in range(len(file_list)):
            viirspath = os.path.join(self.vnldir,file_list[i]+'.tif')
            cnlpath = os.path.join(self.cnldir, file_list[i] + '.tif')
            vnldata = read_imgdata(viirspath)
            cnldata = read_imgdata(cnlpath)
            maskloc = (vnldata <= 1)
            cnldata[maskloc] = 0
            vnldata[maskloc] = 0
            quality_metrics = imgQA.image_quality_assessment(vnldata, cnldata, self.datarange)
            if i == 0:
                gen_indices = quality_metrics
            else:
                gen_indices = sum_dict(gen_indices, quality_metrics)
        img_num_sum = len(file_list)
        for index in gen_indices:
            gen_indices[index] = gen_indices[index] / img_num_sum
        gen_indices['year'] = self.year
        gen_indices['dataset'] = self.dataset
        gen_indices['data'] = self.dataname
        gen_indices['model'] = 'CNL'
        gen_indices['epoch'] = 0
        imgQA.save_imgQI_toCSV(self.saveCSV,[gen_indices])
    def calQualityIndic(self,modelname,modelappendix,epoch,batchsize=30,isCPU=True,config=None):
        if config is None:
            config = getconfig(modelname,modelappendix,epoch)
        if isCPU:
            device = torch.device('cpu')
            config.device = device
        dataset = Dataset_oriDMSP(config, self.valid_txt,'valid', self.vnldir, self.dnldir, self.waterdir, self.otherdata_dict, is_multi_years_combined=False)
        dataloader = torch.utils.data.DataLoader(dataset,batch_size=batchsize,shuffle=False,drop_last=False)
        img_num_sum = 0
        imgQA = ImageQualityAccess()

        net = getNet_fromModel(config, epoch)
        net.eval()
        with torch.no_grad():
            for idx, out_dict in enumerate(dataloader):
                gen_hr = generateSISR(config, net, out_dict)
                # 用归一化前的数据计算损失，输出  label 为原始归一化前数据
                outdata = outdata_transform_qualityassess(config, gen_hr)
                # 转cpu
                outdata = outdata.cpu()
                labeldata = out_dict['viirsdata_ori'].cpu()
                # if jj==0:
                #     break
                # 计算评价指标
                batchnum = len(gen_hr)
                for i in range(batchnum):
                    gen_hr_i = outdata[i].data.detach().numpy()[0]
                    label_i = labeldata[i].data.detach().numpy()[0]
                    maskloc = (label_i<=1)
                    gen_hr_i[maskloc] = 0
                    label_i[maskloc] = 0
                    quality_metrics = imgQA.image_quality_assessment(label_i,gen_hr_i,self.datarange )
                    if img_num_sum == 0:
                        gen_indices = quality_metrics
                    else:
                        gen_indices = sum_dict(gen_indices,quality_metrics)
                    img_num_sum +=1
                    del gen_hr_i
        for index in gen_indices:
            gen_indices[index] = gen_indices[index] / img_num_sum
        gen_indices['year'] = self.year
        gen_indices['dataset'] = self.dataset
        gen_indices['data'] = self.dataname
        gen_indices['model'] = modelname+'_'+modelappendix
        gen_indices['epoch'] = epoch
        imgQA.save_imgQI_toCSV(self.saveCSV,[gen_indices])
    def calQualityIndic_partition(self,modelname,modelappendix,DNSplitValues,epochlst,batchsize=30,isCPU=True):
        #get net
        minvalues = []
        maxvalues = []
        for i in range(len(DNSplitValues)-1):
            minvalues.append(DNSplitValues[i])
            maxvalues.append(DNSplitValues[i+1])
        configlst = []
        netlst = []
        for i in range(len(epochlst)):
            strModelappendix = modelappendix+'_'+str(minvalues[i])+'-'+str(maxvalues[i])
            config = getconfig(modelname, strModelappendix, epochlst[i])
            if isCPU:
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
        dataset = Dataset_oriDMSP(config,self.valid_txt,'valid',self.vnldir,self.dnldir,self.otherdata_dict,is_multi_years_combined=False)
        dataloader = torch.utils.data.DataLoader(dataset,batch_size=batchsize,shuffle=False,drop_last=False)
        img_num_sum = 0
        imgQA = ImageQualityAccess()
        with torch.no_grad():
            for idx, out_dict in enumerate(dataloader):
                gen_hr = generateSISR_partition(out_dict, configlst, netlst, minvalues, maxvalues)
                # 用归一化前的数据计算损失，输出  label 为原始归一化前数据
                outdata = outdata_transform_qualityassess(config, gen_hr)
                # 转cpu
                outdata = outdata.cpu()
                labeldata = out_dict['viirsdata_ori'].cpu()
                # if jj==0:
                #     break
                # 计算评价指标
                batchnum = len(gen_hr)
                for i in range(batchnum):
                    gen_hr_i = outdata[i].data.detach().numpy()[0]
                    label_i = labeldata[i].data.detach().numpy()[0]
                    gen_hr_i[np.isnan(gen_hr_i)]=0
                    gen_hr_i[np.isinf(gen_hr_i)] = 0
                    maskloc = (label_i<=1)
                    gen_hr_i[maskloc] = 0
                    label_i[maskloc] = 0

                    quality_metrics = imgQA.image_quality_assessment(label_i,gen_hr_i,self.datarange )
                    if img_num_sum == 0:
                        gen_indices = quality_metrics
                    else:
                        gen_indices = sum_dict(gen_indices,quality_metrics)
                    img_num_sum +=1
                    del gen_hr_i
        for index in gen_indices:
            gen_indices[index] = gen_indices[index] / img_num_sum
        gen_indices['year'] = self.year
        gen_indices['dataset'] = self.dataset
        gen_indices['data'] = self.dataname
        gen_indices['model'] = modelname+'_'+modelappendix
        gen_indices['epoch'] = epoch
        imgQA.save_imgQI_toCSV(self.saveCSV,[gen_indices])

    def calQualityIndic_NPP02_Model(self,modelname,modelappendix,epoch,batchsize=30,isCPU=True):
        config = getconfig(modelname,modelappendix,epoch)
        if isCPU:
            device = torch.device('cpu')
            config.device = device
        config.dmsp_stat_dict =  getDMSP_stat_dict()
        config.viirs_stat_dict = getVIIRS_stat_dict()
        config.otherdata_dict = self.otherdata_dict
        dataset = Dataset_oriDMSP(config,self.valid_txt,'valid',self.vnldir,self.dnldir,self.otherdata_dict,is_multi_years_combined=False)
        dataloader = torch.utils.data.DataLoader(dataset,batch_size=batchsize,shuffle=False,drop_last=False)
        img_num_sum = 0
        imgQA = ImageQualityAccess()

        net = getNet_fromModel(config, epoch)
        net.eval()
        with torch.no_grad():
            for idx, out_dict in enumerate(dataloader):
                gen_hr = generateSISR(config, net, out_dict)
                # 用归一化前的数据计算损失，输出  label 为原始归一化前数据
                outdata = outdata_transform_qualityassess(config, gen_hr)
                # 转cpu
                outdata = outdata.cpu()
                labeldata = out_dict['viirsdata_ori'].cpu()
                # if jj==0:
                #     break
                # 计算评价指标
                batchnum = len(gen_hr)
                for i in range(batchnum):
                    gen_hr_i = outdata[i].data.detach().numpy()[0]
                    label_i = labeldata[i].data.detach().numpy()[0]
                    maskloc = (label_i<=1)
                    gen_hr_i[maskloc] = 0
                    label_i[maskloc] = 0
                    quality_metrics = imgQA.image_quality_assessment(label_i,gen_hr_i,self.datarange )
                    if img_num_sum == 0:
                        gen_indices = quality_metrics
                    else:
                        gen_indices = sum_dict(gen_indices,quality_metrics)
                    img_num_sum +=1
                    del gen_hr_i
        for index in gen_indices:
            gen_indices[index] = gen_indices[index] / img_num_sum
        gen_indices['year'] = self.year
        gen_indices['dataset'] = self.dataset
        gen_indices['data'] = self.dataname
        gen_indices['model'] = modelname+'_'+modelappendix
        gen_indices['epoch'] = epoch
        imgQA.save_imgQI_toCSV(self.saveCSV,[gen_indices])

    def run_UNet(self):
        saveCSV = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\Unet.csv'
        modelname,modelappendix = "Unet","01"
        epochlst = [1500]
        isCPU = False
        datalst = []
        datalst.append((r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img04',
                        r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\04\train_07.txt'))
        datalst.append((r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img04',
                        r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\04\test_07.txt'))
        datalst.append((r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img04',
                        r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\04\valid_07.txt'))
        datalst.append((r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img03',
                        r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\03\train_05.txt'))
        datalst.append((r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img03',
                        r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\03\valid_05.txt'))
        datalst.append((r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img',
                        r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\01\train.txt'))
        datalst.append((r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img',
                        r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\01\valid.txt'))
        datalst.append((r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img',
                        r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\02\lt10_gt0.txt'))
        datalst.append((r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img',
                        r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\02\lt50_gt0.txt'))
        datalst.append((r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img',
                        r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\02\gt50.txt'))

        for year in range(2012,2014):
            self.year = year
            self.saveCSV = saveCSV
            for dirinfo in datalst:
                self.imgrootdir, self.valid_txt = dirinfo
                self.getPath()
                for epoch in epochlst:
                    config = getconfig()
                    ndvi_stat_dict = config.otherdata_dict['NDVI']
                    ndvi_stat_dict['path'] = os.path.join(self.imgrootdir, str(self.year) + '_NDVI')
                    self.otherdata_dict = {'NDVI': ndvi_stat_dict}
                    self.calQualityIndic(modelname, modelappendix, epoch, batchsize=10, isCPU=isCPU,config=config)

        #cnl
        self.year = '2012'
        self.saveCSV = saveCSV
        for dirinfo in datalst:
            self.imgrootdir, self.valid_txt = dirinfo
            self.getPath()
            self.calCNL_QI()
    def run_UNet_DiffMdls(self):
        isCPU = False
        bz = 10
        saveCSV = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\Unet_03.csv'
        self.saveCSV = saveCSV
        modelinfoLst = []
        # # modelinfoLst.append(("Unet", "01_RNTL",350,'RNTL'))
        # # modelinfoLst.append(("Unet", "01_train12_13", 450,'NDVI'))
        # # modelinfoLst.append(("Unet", "01",1150,'NDVI'))
        # # modelinfoLst.append(("Unet", "01_MeanStd",827,'NDVI'))
        # # modelinfoLst.append(("Unet", "01_RNTL_Cfcvg", 350,'RNTL;Cfcvg'))
        # # modelinfoLst.append(("Unet", "01_Cfcvg",408,'Cfcvg'))
        # # modelinfoLst.append(("Unet", "01_RNTL_Cfcvg", 1900,'RNTL;Cfcvg'))
        # # modelinfoLst.append(("Unet", "01_noOtherData",1150,''))
        # # modelinfoLst.append(("Unet", "01_NDVI_Cfcvg_12_13",2800,'NDVI;Cfcvg'))
        # # modelinfoLst.append(("Unet", "01_rarid",3950,'NDVI'))
        # # modelinfoLst.append(("Unet", "01_rarid_without_drySubhumid",3950,'NDVI'))
        # # modelinfoLst.append(("Unet", "02", 448,'NDVI'))
        # # modelinfoLst.append(("Unet", "01_rarid_onehot_keepWater",2150,'NDVI;Water_Aridity'))
        # # modelinfoLst.append(("Unet", "01_rarid_onehot", 3400,'NDVI;Water_Aridity'))
        # # modelinfoLst.append(("Unet", "01_rarid_class_AVHRR",1650,'AVHRR;Water_Aridity'))
        # # modelinfoLst.append(("Unet", "02_basefilter64",925,'NDVI;Water_Aridity'))
        # # modelinfoLst.append(("Unet", "02_basefilter32",3275,'NDVI;Water_Aridity'))
        #
        # # modelinfoLst.append(("UNet", "03_img", 1450, 'NDVI'))
        # # modelinfoLst.append(("UNet", "03_img03", 475, 'NDVI'))
        # # modelinfoLst.append(("UNet", "03_img04", 475, 'NDVI'))
        # # modelinfoLst.append(("UNet", "03_img04", 525, 'NDVI'))
        # modelinfoLst.append(("UNet", "03_img04", 1850, 'NDVI'))
        # modelinfoLst.append(("UNet", "03_img04", 2975, 'NDVI'))
        # # modelinfoLst.append(("UNet", "01_gradient", 255, 'NDVI'))
        # # modelinfoLst.append(("UNet", "01_gradient", 650, 'NDVI'))
        # modelinfoLst.append(("UNet", "01_gradient", 2175, 'NDVI'))
        # modelinfoLst.append(("UNet", "01_gradient", 2950, 'NDVI'))
        # # modelinfoLst.append(("Unet", "01_rarid_class",2550,'NDVI;Water_Aridity'))
        # # modelinfoLst.append(("Unet", "01_rarid_class",3000,'NDVI;Water_Aridity'))
        # modelinfoLst.append(("UNet", "labelNoProcess_12_13", 1571, 'NDVI'))
        # # modelinfoLst.append(("UNet", "01_gradient_urbanMask", 225, 'NDVI'))
        # # modelinfoLst.append(("UNet", "01_gradient_urbanMask", 550, 'NDVI'))
        # modelinfoLst.append(("UNet", "01_gradient_urbanMask", 1525, 'NDVI'))
        # modelinfoLst.append(("UNet", "01_gradient_urbanMask", 2800, 'NDVI'))



        datalst = []
        datalst.append((r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img03',
                        r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\03\img03_hyperarid.txt'))
        datalst.append((r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img',
                        r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\01\train.txt'))
        datalst.append((r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img',
                        r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\01\valid.txt'))
        datalst.append((r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img',
                        r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\01\aridity.txt'))
        datalst.append((r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img',
                        r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\02\lt10_gt0.txt'))
        datalst.append((r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img',
                        r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\02\lt50_gt0.txt'))
        datalst.append((r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img',
                        r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\02\gt50.txt'))
        datalst.append((r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img04',
                        r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\04\test_07.txt'))
        datalst.append((r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img04',
                        r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\04\valid_07.txt'))
        datalst.append((r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img03',
                        r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\03\img03_arid.txt'))
        datalst.append((r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img03',
                        r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\03\img03_Semiarid.txt'))
        datalst.append((r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img03',
                        r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\03\img03_subhumid.txt'))
        datalst.append((r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img03',
                        r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\03\img03_noArid.txt'))
        datalst.append((r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img03',
                        r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\03\train_05.txt'))
        datalst.append((r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img03',
                        r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\03\valid_05.txt'))
        datalst.append((r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img04',
                        r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\04\train_07.txt'))

        # yearlst = ['2012','2013']
        yearlst = [ '2012']
        for year in yearlst:
            self.year = year
            for dirinfo in datalst:
                for modelinfo in modelinfoLst:
                    modelname, modelappendix,epoch,otherdataNames = modelinfo
                    otherdataNamelst = otherdataNames.split(";")
                    config = getconfig(modelname, modelappendix, epoch)
                    print(modelinfo,self.year,dirinfo)
                    self.imgrootdir, self.valid_txt = dirinfo
                    self.getPath()
                    otherdata_dict = {}
                    for name in otherdataNamelst:
                        if name == "RNTL":
                            stat_dict = config.otherdata_dict['RNTL']
                            otherpath = os.path.join(self.imgrootdir, "2010_RNTL")
                            stat_dict['path'] = otherpath
                            otherdata_dict['RNTL'] =  stat_dict
                        elif name == "Cfcvg":
                            stat_dict = config.otherdata_dict['Cfcvg']
                            otherpath = os.path.join(self.imgrootdir, str(self.year) + '_CfCvg')
                            stat_dict['path'] = otherpath
                            otherdata_dict['Cfcvg'] =  stat_dict
                        elif name == 'NDVI' :
                            stat_dict = config.otherdata_dict['NDVI']
                            otherpath = os.path.join(self.imgrootdir, str(self.year) + '_NDVI')
                            stat_dict['path'] = otherpath
                            otherdata_dict['NDVI'] = stat_dict
                        elif name == 'Water_Aridity':
                            stat_dict = config.otherdata_dict['Water_Aridity']
                            otherpath = os.path.join(self.imgrootdir, 'Water_Aridity')
                            stat_dict['path'] = otherpath
                            otherdata_dict['Water_Aridity'] = stat_dict
                        elif name == 'AVHRR':
                            stat_dict = config.otherdata_dict['AVHRR']
                            otherpath = os.path.join(self.imgrootdir, str(self.year) + '_AVHRR')
                            stat_dict['path'] = otherpath
                            otherdata_dict['AVHRR'] = stat_dict
                    self.otherdata_dict = otherdata_dict
                    t1 = datetime.now()
                    self.calQualityIndic(modelname, modelappendix, epoch, batchsize=bz, isCPU=isCPU,config=config)
                    t2 = datetime.now()
                    print('结束时长：',t2-t1)

        self.year = '2012'
        self.saveCSV = saveCSV
        for dirinfo in datalst:
            self.imgrootdir, self.valid_txt = dirinfo
            self.getPath()
            t1 = datetime.now()
            self.calCNL_QI()
            t2 = datetime.now()
            print(dirinfo,'结束时长：', t2 - t1)


    def runDNLSRNet(self):
        saveCSV = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\Unet.csv'
        isCPU = False
        datalst = []
        # datalst.append((r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img04',
        #                 r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\04\train_07.txt'))
        # datalst.append((r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img04',
        #                 r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\04\test_07.txt'))
        # datalst.append((r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img04',
        #                 r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\04\valid_07.txt'))
        # datalst.append((r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img03',
        #                 r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\03\train_05.txt'))
        # datalst.append((r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img03',
        #                 r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\03\valid_05.txt'))
        # datalst.append((r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img',
        #                 r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\01\train.txt'))
        # datalst.append((r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img',
        #                 r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\01\valid.txt'))
        datalst.append((r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img',
                        r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\02\lt10_gt0.txt'))
        datalst.append((r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img',
                        r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\02\lt50_gt0.txt'))
        datalst.append((r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img',
                        r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\02\gt50.txt'))

        modelInfolst = []
        modelInfolst.append(('DNLSRNet','2012_01_seed3000noGradient',3070))
        modelInfolst.append(('DNLSRNet', '2012_01', 2000))
        modelInfolst.append(('DNLSRNet', '2012_01_seed3000', 3999))
        modelInfolst.append(('DNLSRNet', '2012_01_seed3000_NoCA', 1576))
        for modelInfo in modelInfolst:
            modelname, modelappendix, epoch = modelInfo
            for year in range(2012,2014):
                self.year = year
                self.saveCSV = saveCSV
                for dirinfo in datalst:
                    self.imgrootdir, self.valid_txt = dirinfo
                    self.getPath()
                    self.otherdata_dict = {}
                    ndvi_stat_dict = getNDVI_stat_dict()
                    ndvi_stat_dict['path'] = os.path.join( self.imgrootdir,str(self.year)+'_NDVI')
                    self.otherdata_dict = {'NDVI':ndvi_stat_dict}
                    self.calQualityIndic_NPP02_Model(modelname, modelappendix, epoch, batchsize=10, isCPU=isCPU)


srImgAccess = SRImgValidateSetAccess()
# srImgAccess.runDNLSRNet()
# srImgAccess.run_UNet()
# srImgAccess.run_UNet_DiffMdls()


'''区域评价'''
class SRImgRegionAccess():
    def __init__(self,regionIndex,regionname,srimgFoldername):
        self.rootdir = r'/home/zju/cxx/NTL_timeserise/data/SRCNN/data/out'
        self.regiondir = os.path.join(self.rootdir,regionIndex)
        self.regionIndex = regionIndex
        self.regionname = regionname
        self.srimgFoldername = srimgFoldername
        self.srimgDir = os.path.join(self.rootdir, self.regionIndex, self.srimgFoldername)
        if not os.path.exists(self.srimgDir):
            os.makedirs(self.srimgDir)
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
                # rasterOp.outputResult(genHr_inDs,genHr_mask,gen_hrTifpath)
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

    def stat_model_totalNTLintensity(self):
        dir1 = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\out\out\Unet'
        savepath = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\out\plot\plt\model_tnl.csv'
        regionlst = ['BJH', 'NY', 'Cairo', 'GBA', 'YRD']
        yearlst = [i for i in range(1992,2014)]
        pathInfoLst = []
        pathInfoLst.append((dir1,'Unet_labelNoProcess_12_13','_Unet_labelNoProcess_12_13_1571'))
        pathInfoLst.append((dir1, 'Unet_01_noOtherData', '_Unet_01_noOtherData_3550'))
        pathInfoLst.append((dir1, 'Unet_01_train12_13', '_Unet_01_train12_13_2000'))
        pathInfoLst.append((dir1, 'Unet_01_RNTL', '_Unet_01_RNTL_1950'))
        # pathInfoLst.append((dir2, 'CNL', '_CNL'))
        tnl_result = []
        for pathInfo in pathInfoLst:
            dirpath,dirname,appendstr = pathInfo
            for year in yearlst:
                for region in regionlst:
                    path = os.path.join(dirpath,dirname,region+str(year)+appendstr+'.tif')
                    data = read_imgdata(path)
                    tnl = data.sum()
                    tnl_result.append({'year':year,'region':region,'model':dirname,'tnl':tnl})
        df = pd.DataFrame(data=tnl_result)
        df.to_csv(savepath, header=True, index=False)
    def stat_cnl_totalNTLintensity(self):
        dir1 = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\out\data'
        savepath = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\out\plot\plt\cnl_tnl.csv'
        regionlst = ['BJH', 'NY', 'Cairo', 'GBA', 'YRD']
        yearlst = [i for i in range(2000,2013)]
        pathInfoLst = []
        pathInfoLst.append((dir1,'CNL','_CNL'))
        tnl_result = []
        for pathInfo in pathInfoLst:
            dirpath,dirname,appendstr = pathInfo
            for year in yearlst:
                for region in regionlst:
                    path = os.path.join(dirpath,dirname,region+'_'+str(year)+appendstr+'.tif')
                    data = read_imgdata(path)
                    tnl = data.sum()
                    tnl_result.append({'year':year,'region':region,'model':dirname,'tnl':tnl})
        df = pd.DataFrame(data=tnl_result)
        df.to_csv(savepath, header=True, index=False)

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
            genHr_inData = read_imgdata(genpath)
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
        genHr_inData = read_imgdata(genpath)
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
            genHr_inData = read_imgdata((genpath))
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
            genHr_inData = read_imgdata(genpath)
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
# regionlst = []
# regionlst.append(('01','BJH'))
# regionlst.append(('02','NewYork'))
# regionlst.append(('03','cairo'))
# regionlst.append(('04','YRD'))
# regionlst.append(('05','GBA'))
# srimgFoldername = 'output'
# year = 2013
# # # 生成超分图像
# # modellst = []
# # modellst.append(('AE','gradient_loss_-3',1001))
# # modellst.append(('DNLSRNet','01_ndvimasked',972))
# # modellst.append(('DNLSRNet','gntl_train1_01',1447))
# # modellst.append(('DNLSRNet','train1_01',1270))
# # modellst.append(('DNLSRNet','01_noln',700))
# # modellst.append(('DNLSRNet','02',2269))
# # modellst.append(('DNLSRNet','ds_01',2926))
# # modellst.append(('DNLSRNet','01',1013))
# # for regionInfo in regionlst:
# #     regionIndex, regionname = regionInfo
# #     srRegionQA = SRImgRegionAccess(regionIndex,regionname,srimgFoldername)
# #     for modelname,modelappendix,epoch in modellst:
# #         srRegionQA.createSRImg(modelname,modelappendix,epoch,year)
#
# # #计算指标
# modelLst = []
# modelLst.append('DNLSRNet_01_ndvimasked_972_2013')
# modelLst.append('DNLSRNet_01_noln_700_2012')
# modelLst.append('DNLSRNet_gntl_train1_01_1447_2013')
# modelLst.append('DNLSRNet_02_2269_2013')
# modelLst.append('DNLSRNet_train1_01_1270_2013')
# modelLst.append('DNLSRNet_ds_01_2926_2013')
# modelLst.append('DNLSRNet_01_1013_2013')
# modelLst.append('AE_gradient_loss_-3_1001_2013')
# outpath = r'/home/zju/cxx/NTL_timeserise/data/SRCNN/data/out/regionQI_wholeImg.csv'
# for regionInfo in regionlst:
#     regionIndex, regionname = regionInfo
#     print(regionInfo)
#     srRegionQA = SRImgRegionAccess(regionIndex,regionname,srimgFoldername)
#     targetImgpath = os.path.join(srRegionQA.regiondir,'data',srRegionQA.regionname+"_VNL"+str(year)+".tif")
#     # predImgpath = os.path.join(srRegionQA.regiondir, 'data', srRegionQA.regionname + "_chen" + str(year) + ".tif")
#     # srRegionQA.accessSRImgQuality_CNL(targetImgpath,predImgpath,outpath,year)
#     srRegionQA.accessSRImgQuality_modelGenHr([year],modelLst,outpath)




'''全球尺度评价'''
class SRImgWorldAccess():
    def __init__(self):
        self.rootdir = r'/home/zju/cxx/NTL_timeserise/data/SRCNN/data/world'
        self.IdsTxt = r'/home/zju/cxx/NTL_timeserise/data/SRCNN/data/world/inputs/img/grid/Id.txt'
        self.ntl2012 = r'/home/zju/cxx/NTL_timeserise/data/SRCNN/data/world/inputs/img/2012_oriDNL'
        self.ntl2013 = r'/home/zju/cxx/NTL_timeserise/data/SRCNN/data/world/inputs/img/2013_oriDNL'
        self.vnl2012 = r'/home/zju/cxx/NTL_timeserise/data/SRCNN/data/world/inputs/img/2012_VNL'
        self.vnl2013 = r'/home/zju/cxx/NTL_timeserise/data/SRCNN/data/world/inputs/img/2013_VNL'
        # self.imgdir = r'/home/zju/cxx/NTL_timeserise/data/SRCNN/data/world/inputs/img'
        self.imgdir = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\world\inputs\img'
        self.resultdir = r'/home/zju/cxx/NTL_timeserise/data/SRCNN/data/world/result'
        self.srGenDataRootDir = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\world\result'
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

    def mergeCsvFiles_diffData(self,year='2012',dataname='VNL',regiontype='nation'):
        '''
        dbf转csv,合并不同区域的数据到csv文件
        :param year:
        :param dataname:
        :param regiontype:
        :return:
        '''
        dbfDir = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\world\analysis\zonalStat\dbf'
        csvDir = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\world\analysis\zonalStat\csv'
        csvpath = os.path.join(csvDir, regiontype+'_'+dataname + '_' + year + '_prj.csv')
        regionlst = []
        regionlst.append('Africa')
        regionlst.append('Asia')
        regionlst.append('Europe')
        regionlst.append('NorthAmerica')
        regionlst.append('Oceania')
        regionlst.append('SouthAmerica')
        reCollst = ['ID']
        if regiontype == 'nation':
            collst = ['ELEMID']
        elif regiontype == 'province':
            collst = ['GID_1']
        elif regiontype == 'city':
            collst = ['GID_2']

        if dataname == 'VNL':
            collst.append('ZONE_CODE')
            collst.append('COUNT')
            collst.append('AREA')
            collst.append('SUM')
            reCollst.append('ZONE_CODE')
            reCollst.append('COUNT')
            reCollst.append('AREA')
            reCollst.append(dataname)
        elif dataname == 'CNL':
            collst.append('SUM')
            reCollst.append(dataname)
        else:
            collst.append('ZONE_CODE')
            collst.append('COUNT')
            collst.append('AREA')
            collst.append('SUM')
            reCollst.append('ZONE_CODE')
            reCollst.append('COUNT')
            reCollst.append('AREA')
            reCollst.append(dataname)

        fileOp = FileOpr()
        dataframe_lst = []
        for region in regionlst:
            dbfpath = os.path.join(dbfDir,year+'_'+dataname+'_'+regiontype+'_prj_'+region+'.dbf')
            if os.path.exists(dbfpath):
                df = fileOp.readDBF_asDataFrame(dbfpath)
                subdf = df[collst]
                subdf.columns=reCollst
                dataframe_lst.append(subdf)
        result = pd.concat(dataframe_lst,axis=0)
        if os.path.exists(csvpath):
            result.to_csv(csvpath, header=False, index=False, mode='a')
        else:
            result.to_csv(csvpath,header=True,index=False)
    def mergeData(self,modellst ,year='2012',regiontype='nation'):
        '''
        csv列上扩展合并，合并VNL\CNL\Model的统计数据
        :param modellst:
        :param year:
        :param regiontype:
        :return:
        '''
        inDir = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\world\analysis\zonalStat\csv'
        outPath = os.path.join(inDir,year+'_'+regiontype+'_prj.csv')
        idCol = 'ID'
        # vnlpath = os.path.join(inDir, regiontype+'_'+'VNL_' + year + '_prj.csv')
        # vnlDf = pd.read_csv(vnlpath)

        # #CNL
        # cnlpath = os.path.join(inDir, regiontype+'_'+'CNL_' + year + '_prj.csv')
        # if os.path.exists(cnlpath):
        #     cnlDf = pd.read_csv(cnlpath)
        #     #列上扩展
        #     vnlDf = pd.merge(vnlDf, cnlDf, how='inner', on=idCol)

        #modellst
        # df = vnlDf
        df = None
        for modelname in modellst:
            path = os.path.join(inDir,regiontype+'_'+modelname+'_'+year + '_prj.csv')
            modelDF = pd.read_csv(path)
            if df is None:
                df = modelDF
            else:
                df = pd.merge(df, modelDF, how='inner', on=idCol)
        if os.path.exists(outPath):
            df.to_csv(outPath, header=False, index=False, mode='a')
        else:
            df.to_csv(outPath,header=True,index=False)
    def mergeData_srDNLExisted(self,modellst ,year='2012',regiontype='nation'):
        '''
        csv列上扩展合并，合并VNL\CNL\Model的统计数据
        :param modellst:
        :param year:
        :param regiontype:
        :return:
        '''
        srDNL_dir=r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\world\analysis\zonalStat\1'
        inDir = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\world\analysis\zonalStat\csv'
        idCol = 'ID'

        for y in range(2001,2012):
            path = os.path.join(srDNL_dir,str(y)+'_nation_prj.csv' )
            df = pd.read_csv(path)
            cnlpath = os.path.join(inDir, regiontype + '_' + 'CNL_' + str(y) + '_prj.csv')
            cnlDf = pd.read_csv(cnlpath)
            # 列上扩展
            df = pd.merge(df, cnlDf, how='inner', on=idCol)
            outPath = os.path.join(inDir, str(y) + '_' + regiontype + '_prj.csv')
            df.to_csv(outPath,header=True,index=False)
    def mergeData_VNL(self,year='2014',regiontype='nation'):
        '''
        csv列上扩展合并，合并VNL\CNL\Model的统计数据
        :param modellst:
        :param year:
        :param regiontype:
        :return:
        '''
        inDir = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\world\analysis\zonalStat\csv'
        outPath = os.path.join(inDir,year+'_'+regiontype+'_prj.csv')
        idCol = 'ID'
        vnlpath = os.path.join(inDir, regiontype+'_'+'VNL_' + year + '_prj.csv')
        vnlDf = pd.read_csv(vnlpath)
        if os.path.exists(outPath):
            vnlDf.to_csv(outPath, header=False, index=False, mode='a')
        else:
            vnlDf.to_csv(outPath,header=True,index=False)

    def calR2_AllData(self,modellst,year='2012',regiontype='nation'):
        inDir = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\world\analysis\zonalStat\csv'
        outPath = os.path.join(inDir,year+'_'+regiontype+'_prj.csv')
        # outPath = os.path.join(inDir,year+'_'+regiontype+'.csv')
        df = pd.read_csv(outPath)
        subdf = df
        ycol = 'VNL'

        #CNL
        if year == '2012':
            xcol = 'CNL'
            score1 = r2_score(subdf[xcol], subdf[ycol])
            print(regiontype+' '+year+' VNL & CNL:', score1)

        #model
        for modelname in modellst:
            xcol = modelname
            score1 = r2_score(subdf[xcol], subdf[ycol])
            print(regiontype+' '+year+' VNL & '+modelname+':', score1)
    def calR2_DataFilter(self,modellst,year='2012',regiontype='nation'):
        inDir = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\world\analysis\zonalStat\csv'
        outPath = os.path.join(inDir,year+'_'+regiontype+'_prj.csv')
        # outPath = os.path.join(inDir,year+'_'+regiontype+'.csv')
        df = pd.read_csv(outPath)
        subdf = df
        ycol = 'VNL'
        xcol = 'UNet_labelNoProcess_12_13_1571'
        xcol = 'UNet_03_img04_2975'
        oriDf = df
        # df = oriDf[((oriDf[xcol]<60)&(oriDf[ycol]<60))|((oriDf[xcol]>=60)&(oriDf[ycol]>=60))]

        #选择合适的阈值
        # uplst = [20,30,40,50,60,70,80,90,100,110.120,130,140,150,200,300,500,1000,10000]
        uplst = [150]
        lowlst = [0]
        for i in range(len(uplst)):
            for j in range(len(lowlst)):
                lowlimits = lowlst[j]
                uplimits = uplst[i]
                xcol = 'CNL'
                subdf = df[((df[ycol] <= uplimits) & (df[ycol] >= lowlimits))]
                subdf = subdf[((subdf[xcol] <= uplimits) & (subdf[xcol] >= lowlimits))]
                score = r2_score(subdf[xcol], subdf[ycol])
                print(year+' '+xcol+'~'+ycol+" r2: ", score,' ('+ str(lowlimits)+'~'+str(uplimits)+')')
                plt.figure()
                plt.scatter(subdf[xcol], subdf[ycol], color='g')
                maxvalue = subdf[ycol].max()
                xlst = [0, 1,maxvalue ]
                plt.plot(xlst, xlst, color='r', linestyle=':')
                plt.xlabel(xcol)
                plt.ylabel(ycol)
                plt.title(year+" r2: "+ str(score) +' ('+ str(lowlimits)+'~'+str(uplimits)+')')

                xcol = 'UNet_labelNoProcess_12_13_1571'
                subdf = df[((df[ycol] <= uplimits) & (df[ycol] >= lowlimits))]
                subdf = subdf[((subdf[xcol] <= uplimits) & (subdf[xcol] >= lowlimits))]
                score = r2_score(subdf[xcol], subdf[ycol])
                plt.figure()
                plt.scatter(subdf[xcol], subdf[ycol], color='g')
                maxvalue = subdf[ycol].max()
                xlst = [0, 1,maxvalue ]
                plt.plot(xlst, xlst, color='r', linestyle=':')
                plt.xlabel(xcol)
                plt.ylabel(ycol)
                plt.title(year+" r2: "+ str(score) +' ('+ str(lowlimits)+'~'+str(uplimits)+')')


                # 计算不同数据的R2
                uplst = [70]
                lowlst = [0]
                for i in range(len(uplst)):
                    for j in range(len(lowlst)):
                        lowlimits = lowlst[j]
                        uplimits = uplst[i]
                        # CNL
                        if year == '2012':
                            xcol = 'CNL'
                            subdf = df[((df[ycol] <= uplimits) & (df[ycol] >= lowlimits))]
                            subdf = subdf[((subdf[xcol] <= uplimits) & (subdf[xcol] >= lowlimits))]
                            score = r2_score(subdf[xcol], subdf[ycol])
                            print(regiontype + ' ' + year + ' VNL & CNL:', score)
                        # model
                        for modelname in modellst:
                            xcol = modelname
                            subdf = df[((df[ycol] <= uplimits) & (df[ycol] >= lowlimits))]
                            subdf = subdf[((subdf[xcol] <= uplimits) & (subdf[xcol] >= lowlimits))]
                            score = r2_score(subdf[xcol], subdf[ycol])
                            print(regiontype + ' ' + year + ' VNL & ' + modelname + ':', score)

                # plt.figure()
                # plt.scatter(subdf[xcol], subdf[ycol], color='g')
                # maxvalue = subdf[ycol].max()
                # xlst = [0, 1,maxvalue ]
                # plt.plot(xlst, xlst, color='r', linestyle=':')
                # plt.xlabel(xcol)
                # plt.ylabel(ycol)
                # plt.title(year+" r2: "+ str(score) +' ('+ str(lowlimits)+'~'+str(uplimits)+')')
                # print(year+' '+xcol+'~'+ycol+" r2: ", score,' ('+ str(lowlimits)+'~'+str(uplimits)+')')


    def calR2_nation(self):
        fidcol = 'PID'; regioncol = 'Region';yearcol = 'Year';
        vnlCol = 'VNL';srNLCol= 'srNL';cnlCol = 'CNL';
        csvpath = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\world\inputs\r2Valid_loc\stat\r2_nation.csv'
        df = pd.read_csv(csvpath)
        # 2012
        df_2012 = df[df[yearcol]==2012]
        df_2012 = df_2012[(df_2012[vnlCol]>10)&(df_2012[srNLCol]>10)]
        df_2013 = df[df[yearcol]==2013]
        df_2013 = df_2013[(df_2013[vnlCol]>10)&(df_2013[srNLCol]>10)]
        #score
        score1 = r2_score(df_2012[vnlCol], df_2012[srNLCol])
        print('2012 VNL & srNRL:',score1)
        scorelst_2012_sr = []
        scorelst_2013_sr = []
        scorelst_2012_cnl = []
        limits = [100, 500,1e3, 5e3,1e4,5e4, 1e5, 5e5,1e6,5e6, 1e7,5e7]
        for ll in limits:
            subdf = df_2012[df_2012[vnlCol] <= ll]
            score1 = r2_score(subdf[vnlCol], subdf[srNLCol])
            score3 = r2_score(subdf[vnlCol], subdf[cnlCol])
            subdf2 = df_2013[df_2013[vnlCol] <= ll]
            score2 = r2_score(subdf2[vnlCol], subdf2[srNLCol])
            print('max limits:' + str(ll), '2012 SR:', '{:.3f}'.format(score1), '2012 CNL:', '{:.3f}'.format(score3),
                  '2013 SR:', '{:.3f}'.format(score2))
            scorelst_2012_sr.append(score1)
            scorelst_2013_sr.append(score2)
            scorelst_2012_cnl.append(score3)

        # 2012 sr PLT
        score1 = r2_score(df_2012[vnlCol], df_2012[srNLCol])
        print('2012 VNL & srNRL:', score1)
        plt.figure()
        plt.scatter(df_2012[vnlCol], df_2012[srNLCol], color='g')
        xlst = [0, 1, 1e6, 1e7, 1.5e7, 2e7]
        plt.plot(xlst, xlst, color='r', linestyle=':')
        plt.xlabel('VNL')
        plt.ylabel('srNL')
        plt.xlim(0, 1e6)
        plt.ylim(0, 1e6)
        plt.text(6e5, 9e5, 'R2 = ' + "{:.3f}".format(score1))
        plt.text(6e5, 8e5, 'best R2 = ' + "{:.3f}".format(max(scorelst_2012_sr)))
        # 2012 cnl PLT
        score1 = r2_score(df_2012[vnlCol], df_2012[cnlCol])
        print('2012 VNL & CNL:', score1)
        plt.figure()
        plt.scatter(df_2012[vnlCol], df_2012[cnlCol], color='g')
        xlst = [0, 1, 1e6, 1e7, 1.5e7, 2e7]
        plt.plot(xlst, xlst, color='r', linestyle=':')
        plt.xlabel('VNL')
        plt.ylabel('srNL')
        plt.xlim(0, 1e6)
        plt.ylim(0, 1e6)
        plt.text(6e5, 9e5, 'R2 = ' + "{:.3f}".format(score1))
        plt.text(6e5, 8e5, 'best R2 = ' + "{:.3f}".format(max(scorelst_2012_cnl)))
        # 2013 sr plt
        score1 = r2_score(df_2013[vnlCol], df_2013[srNLCol])
        print('2013 VNL & srNRL:', score1)
        plt.figure()
        plt.scatter(df_2013[vnlCol], df_2013[srNLCol], color='g')
        xlst = [0, 1, 1e6, 1e7, 1.5e7, 2e7]
        plt.plot(xlst, xlst, color='r', linestyle=':')
        plt.xlabel('VNL')
        plt.ylabel('srNL')
        plt.xlim(0, 1e6)
        plt.ylim(0, 1e6)
        plt.text(6e5, 9e5, 'R2 = ' + "{:.3f}".format(score1))
        plt.text(6e5, 8e5, 'best R2 = ' + "{:.3f}".format(np.nanmax(scorelst_2013_sr)))
        df.drop()
    def calR2_city(self):
        fidcol = 'PID'; regioncol = 'Region';yearcol = 'Year';
        vnlCol = 'VNL';srNLCol= 'srNL';cnlCol = 'CNL';
        csvpath = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\world\inputs\r2Valid_loc\city\stat_cityTNL.csv'
        df = pd.read_csv(csvpath)

        # 2012
        df_2012 = df[df[yearcol]==2012]
        df_2012 = df_2012[(df_2012[vnlCol]>10)&(df_2012[srNLCol]>10)]
        df_2013 = df[df[yearcol]==2013]
        df_2013 = df_2013[(df_2013[vnlCol]>10)&(df_2013[srNLCol]>10)]
        #R2
        score1 = r2_score(df_2012[vnlCol], df_2012[srNLCol])
        print('2012 VNL & srNRL:',score1)
        scorelst_2012_sr = []
        scorelst_2013_sr = []
        scorelst_2012_cnl = []
        limits = [100,1000,5000,10000,15000,20000,25000,30000,40000,50000,60000,70000,80000,90000,100000,150000,200000]
        for ll in limits:
            subdf = df_2012[df_2012[vnlCol] <= ll]
            score1 = r2_score(subdf[vnlCol], subdf[srNLCol])
            score3 = r2_score(subdf[vnlCol], subdf[cnlCol])
            subdf2 = df_2013[df_2013[vnlCol] <= ll]
            score2 = r2_score(subdf2[vnlCol], subdf2[srNLCol])
            print('max limits:' + str(ll), '2012 SR:', '{:.3f}'.format(score1), '2012 CNL:', '{:.3f}'.format(score3),
                  '2013 SR:', '{:.3f}'.format(score2))
            scorelst_2012_sr.append(score1)
            scorelst_2013_sr.append(score2)
            scorelst_2012_cnl.append(score3)
        # 2012 sr PLT
        score1 = r2_score(df_2012[vnlCol], df_2012[srNLCol])
        print('2012 VNL & srNRL:', score1)
        plt.figure()
        plt.scatter(df_2012[vnlCol], df_2012[srNLCol], color='g')
        xlst = [0, 1, 1e5]
        plt.plot(xlst, xlst, color='r', linestyle=':')
        plt.xlabel('VNL')
        plt.ylabel('srNL')
        plt.xlim(0, 1e5)
        plt.ylim(0, 1e5)
        plt.text(2e4, 9e4, 'R2 = ' + "{:.3f}".format(score1))
        plt.text(2e4, 8e4, 'best R2 = ' + "{:.3f}".format(max(scorelst_2012_sr)))
        # 2012 cnl PLT
        score1 = r2_score(df_2012[vnlCol], df_2012[cnlCol])
        print('2012 VNL & srNRL:', score1)
        plt.figure()
        plt.scatter(df_2012[vnlCol], df_2012[cnlCol], color='g')
        xlst = [0, 1, 1e5]
        plt.plot(xlst, xlst, color='r', linestyle=':')
        plt.xlabel('VNL')
        plt.ylabel('srNL')
        plt.xlim(0, 1e5)
        plt.ylim(0, 1e5)
        plt.text(2e4, 9e4, 'R2 = ' + "{:.3f}".format(score1))
        plt.text(2e4, 8e4, 'best R2 = ' + "{:.3f}".format(max(scorelst_2012_cnl)))
        # 2013 sr plt
        score1 = r2_score(df_2013[vnlCol], df_2013[srNLCol])
        print('2012 VNL & srNRL:', score1)
        plt.figure()
        plt.scatter(df_2013[vnlCol], df_2013[srNLCol], color='g')
        xlst = [0, 1, 1e5]
        plt.plot(xlst, xlst, color='r', linestyle=':')
        plt.xlabel('VNL')
        plt.ylabel('srNL')
        plt.xlim(0, 1e5)
        plt.ylim(0, 1e5)
        plt.text(2e4, 9e4, 'R2 = ' + "{:.3f}".format(score1))
        plt.text(2e4, 8e4, 'best R2 = ' + "{:.3f}".format(max(scorelst_2013_sr)))
    def calR2_province(self):
        fidcol = 'PID';
        regioncol = 'Region';
        yearcol = 'Year';
        vnlCol = 'VNL';
        srNLCol = 'srNL';
        cnlCol = 'CNL';
        csvpath = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\world\inputs\r2Valid_loc\province\stat_provinceTNL_DelAbnormal.csv';
        df = pd.read_csv(csvpath)

        # 2012
        df_2012 = df[df[yearcol] == 2012]
        df_2012 = df_2012[(df_2012[vnlCol] > 10) & (df_2012[srNLCol] > 10)]
        df_2013 = df[df[yearcol] == 2013]
        df_2013 = df_2013[(df_2013[vnlCol] > 10) & (df_2013[srNLCol] > 10)]
        # R2
        score1 = r2_score(df_2012[vnlCol], df_2012[srNLCol])
        print('2012 VNL & srNRL:', score1)
        score1 = r2_score(df_2012[vnlCol], df_2012[cnlCol])
        print('2012 VNL & CNL:', score1)
        score1 = r2_score(df_2013[vnlCol], df_2013[srNLCol])
        print('2013 VNL & srNRL:', score1)
        scorelst_2012_sr = []
        scorelst_2013_sr = []
        scorelst_2012_cnl = []
        limits = [100, 1000, 5000, 10000, 15000, 20000, 25000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000,
                  150000, 200000]
        for ll in limits:
            subdf = df_2012[df_2012[vnlCol] <= ll]
            score1 = r2_score(subdf[vnlCol], subdf[srNLCol])
            score3 = r2_score(subdf[vnlCol], subdf[cnlCol])
            subdf2 = df_2013[df_2013[vnlCol] <= ll]
            score2 = r2_score(subdf2[vnlCol], subdf2[srNLCol])
            print('max limits:' + str(ll), '2012 SR:', '{:.3f}'.format(score1), '2012 CNL:', '{:.3f}'.format(score3),
                  '2013 SR:', '{:.3f}'.format(score2))
            scorelst_2012_sr.append(score1)
            scorelst_2013_sr.append(score2)
            scorelst_2012_cnl.append(score3)
        # 2012 sr PLT
        score1 = r2_score(df_2012[vnlCol], df_2012[srNLCol])
        print('2012 VNL & srNRL:', score1)
        plt.figure()
        plt.scatter(df_2012[vnlCol], df_2012[srNLCol], color='g')
        xlst = [0, 1, 1e5]
        plt.plot(xlst, xlst, color='r', linestyle=':')
        plt.xlabel('VNL')
        plt.ylabel('srNL')
        plt.xlim(0, 1e5)
        plt.ylim(0, 1e5)
        plt.text(2e4, 9e4, 'R2 = ' + "{:.3f}".format(score1))
        plt.text(2e4, 8e4, 'best R2 = ' + "{:.3f}".format(max(scorelst_2012_sr)))
        # 2012 cnl PLT
        score1 = r2_score(df_2012[vnlCol], df_2012[cnlCol])
        print('2012 VNL & srNRL:', score1)
        plt.figure()
        plt.scatter(df_2012[vnlCol], df_2012[cnlCol], color='g')
        xlst = [0, 1, 1e5]
        plt.plot(xlst, xlst, color='r', linestyle=':')
        plt.xlabel('VNL')
        plt.ylabel('srNL')
        plt.xlim(0, 1e5)
        plt.ylim(0, 1e5)
        plt.text(2e4, 9e4, 'R2 = ' + "{:.3f}".format(score1))
        plt.text(2e4, 8e4, 'best R2 = ' + "{:.3f}".format(max(scorelst_2012_cnl)))
        # 2013 sr plt
        score1 = r2_score(df_2013[vnlCol], df_2013[srNLCol])
        print('2012 VNL & srNRL:', score1)
        plt.figure()
        plt.scatter(df_2013[vnlCol], df_2013[srNLCol], color='g')
        xlst = [0, 1, 1e5]
        plt.plot(xlst, xlst, color='r', linestyle=':')
        plt.xlabel('VNL')
        plt.ylabel('srNL')
        plt.xlim(0, 1e5)
        plt.ylim(0, 1e5)
        plt.text(2e4, 9e4, 'R2 = ' + "{:.3f}".format(score1))
        plt.text(2e4, 8e4, 'best R2 = ' + "{:.3f}".format(max(scorelst_2013_sr)))
    def calR2_point(self):
        fidcol = 'PID';
        regioncol = 'Region';
        yearcol = 'Year';
        vnlCol = 'VNL';
        srNLCol = 'srNL';
        cnlCol = 'CNL';
        csvpath = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\world\inputs\r2Valid_loc\randomPoints\global_smaple_01_oriGRC\rasterValueByPoints.csv';
        df = pd.read_csv(csvpath)

        # 2012
        df_2012 = df[df[yearcol] == 2012]
        df_2012 = df_2012[(df_2012[vnlCol] >= 0) & (df_2012[srNLCol] >= 0) & (df_2012[cnlCol] >=0)]
        df_2013 = df[df[yearcol] == 2013]
        df_2013 = df_2013[(df_2013[vnlCol]>=0)&(df_2013[srNLCol]>=0)]
        # R2
        score1 = r2_score(df_2012[vnlCol], df_2012[srNLCol])
        print('2012 VNL & srNRL:', score1)
        score1 = r2_score(df_2012[vnlCol], df_2012[cnlCol])
        print('2012 VNL & CNL:', score1)
        score1 = r2_score(df_2013[vnlCol], df_2013[srNLCol])
        print('2013 VNL & srNRL:', score1)
        scorelst_2012_sr = []
        scorelst_2013_sr = []
        scorelst_2012_cnl = []
        limits = list(range(10,200,10))
        for ll in limits:
            subdf = df_2012[df_2012[vnlCol] <= ll]
            score1 = r2_score(subdf[vnlCol], subdf[srNLCol])
            score3 = r2_score(subdf[vnlCol], subdf[cnlCol])
            subdf2 = df_2013[df_2013[vnlCol] <= ll]
            score2 = r2_score(subdf2[vnlCol], subdf2[srNLCol])
            print('max limits:' + str(ll), '2012 SR:', '{:.3f}'.format(score1), '2012 CNL:', '{:.3f}'.format(score3),
                  '2013 SR:', '{:.3f}'.format(score2))
            scorelst_2012_sr.append(score1)
            scorelst_2013_sr.append(score2)
            scorelst_2012_cnl.append(score3)
        # 2012 sr PLT
        score1 = r2_score(df_2012[vnlCol], df_2012[srNLCol])
        print('2012 VNL & srNRL:', score1)
        plt.figure()
        plt.scatter(df_2012[vnlCol], df_2012[srNLCol], color='g')
        xlst = [0, 1, 1e5]
        plt.plot(xlst, xlst, color='r', linestyle=':')
        plt.xlabel('VNL')
        plt.ylabel('srNL')
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        plt.text(50, 90, 'R2 = ' + "{:.3f}".format(score1))
        plt.text(50, 80, 'best R2 = ' + "{:.3f}".format(max(scorelst_2012_sr)))
        # 2012 cnl PLT
        score1 = r2_score(df_2012[vnlCol], df_2012[cnlCol])
        print('2012 VNL & srNRL:', score1)
        plt.figure()
        plt.scatter(df_2012[vnlCol], df_2012[cnlCol], color='g')
        xlst = [0, 1, 1e5]
        plt.plot(xlst, xlst, color='r', linestyle=':')
        plt.xlabel('VNL')
        plt.ylabel('srNL')
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        plt.text(50, 90, 'R2 = ' + "{:.3f}".format(score1))
        plt.text(50, 80, 'best R2 = ' + "{:.3f}".format(max(scorelst_2012_cnl)))
        # 2013 sr plt
        score1 = r2_score(df_2013[vnlCol], df_2013[srNLCol])
        print('2012 VNL & srNRL:', score1)
        plt.figure()
        plt.scatter(df_2013[vnlCol], df_2013[srNLCol], color='g')
        xlst = [0, 1, 1e5]
        plt.plot(xlst, xlst, color='r', linestyle=':')
        plt.xlabel('VNL')
        plt.ylabel('srNL')
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        plt.text(50, 90, 'R2 = ' + "{:.3f}".format(score1))
        plt.text(50, 80, 'best R2 = ' + "{:.3f}".format(max(scorelst_2013_sr)))
    def calR2(self):
        csvpath = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\world\inputs\r2Valid_loc\stat\r2_nation.csv'
        yearcol = 'Year';
        vnlCol = 'VNL'
        srNLCol = 'DNLSRNet_2012_01_2000'
        df = pd.read_csv(csvpath)
        df_2012 = df[df[yearcol]==2012]

        #NATION
        df_2012 = df_2012[(df_2012[vnlCol]>10)&(df_2012[srNLCol]>10)]
        #score
        score1 = r2_score(df_2012[vnlCol], df_2012[srNLCol])
        print('2012 VNL & srNRL:',score1)
        scorelst_2012_sr = []
        limits = [500,1e3, 5e3,1e4,5e4, 1e5, 5e5,1e6,5e6, 1e7,5e7]
        for ll in limits:
            subdf = df_2012[df_2012[vnlCol] <= ll]
            score1 = r2_score(subdf[vnlCol], subdf[srNLCol])
            print('max limits:' + str(ll), '2012 SR:', '{:.3f}'.format(score1))
            scorelst_2012_sr.append(score1)
        plt.figure()
        plt.scatter(df_2012[vnlCol], df_2012[srNLCol], color='g')
        xlst = [0, 1, 1e6, 1e7, 1.5e7, 2e7]
        plt.plot(xlst, xlst, color='r', linestyle=':')
        plt.xlabel('VNL')
        plt.ylabel('srNL')
        plt.xlim(0, 1e6)
        plt.ylim(0, 1e6)
        plt.text(6e5, 9e5, 'R2 = ' + "{:.3f}".format(score1))
        plt.text(6e5, 8e5, 'best R2 = ' + "{:.3f}".format(max(scorelst_2012_sr)))

srWorld = SRImgWorldAccess()
# # ########mergeCsvFiles_diffData
# # yearlst = []
# # for year in range(1992,2012):
# #     yearlst.append(str(year))
# # # regiontypelst = ['nation','province','city']
# # regiontypelst = ['province']
# # # datanamelst = ['VNL','CNL','UNet_labelNoProcess_12_13_1571']
# # datanamelst = []
# # datanamelst.append('UNet_labelNoProcess_12_13_1571')
# # # datanamelst.append('UNet_01_gradient_urbanMask_1525')
# # # datanamelst.append('UNet_01_rarid_class_2550')
# # # datanamelst.append('UNet_03_img04_2975')
# # for year in yearlst:
# #     for regiontype in regiontypelst:
# #         for dataname in datanamelst:
# #             print(year,dataname,regiontype)
# #             srWorld.mergeCsvFiles_diffData(year,dataname,regiontype)
#
# #2000-2011CNL
# yearlst = []
# for year in range(2000,2012):
#     yearlst.append(str(year))
# regiontypelst = ['nation']
# datanamelst = []
# datanamelst.append('CNL')
# for year in yearlst:
#     for regiontype in regiontypelst:
#         for dataname in datanamelst:
#             print(year,dataname,regiontype)
#             srWorld.mergeCsvFiles_diffData(year,dataname,regiontype)
# #2014-2020CNL
# yearlst = []
# for year in range(2014,2021):
#     yearlst.append(str(year))
# regiontypelst = ['nation']
# datanamelst = []
# datanamelst.append('VNL')
# for year in yearlst:
#     for regiontype in regiontypelst:
#         for dataname in datanamelst:
#             print(year,dataname,regiontype)
#             srWorld.mergeCsvFiles_diffData(year,dataname,regiontype)

# #################mergeData
# modellst = []
# modellst.append('UNet_labelNoProcess_12_13_1571')
# # modellst.append('UNet_01_gradient_urbanMask_1525')
# # modellst.append('UNet_01_rarid_class_2550')
# # modellst.append('UNet_03_img04_2975')
# yearlst = []
# for year in range(1992,2012):
#     yearlst.append(str(year))
# # regiontypelst = ['nation','province','city']
# regiontypelst = ['province']
# for year in yearlst:
#     for regiontype in regiontypelst:
#         print(year,regiontype)
#         srWorld. mergeData(modellst ,year,regiontype)
# # #########calR2
# # modellst = []
# # modellst.append('UNet_labelNoProcess_12_13_1571')
# # modellst.append('UNet_01_gradient_urbanMask_1525')
# # modellst.append('UNet_01_rarid_class_2550')
# # modellst.append('UNet_03_img04_2975')
# # yearlst = []
# # yearlst.append('2012')
# # # yearlst.append('2013')
# # # regiontypelst = ['nation','province','city']
# # regiontypelst = ['point']
# # for year in yearlst:
# #     for regiontype in regiontypelst:
# #         srWorld.calR2_AllData(modellst,year,regiontype)



def reload(path,outpath):
    filelst = tuple(open(path,'r'))
    filelst = [id_.rstrip() for id_ in filelst]
    for file in filelst:
        arr = file.split('_')
        a = int(arr[0])
        if a < 49:
            with open(outpath,'a+') as f:
                f.write(file+'\n')




