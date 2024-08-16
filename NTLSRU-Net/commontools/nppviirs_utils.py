import numpy as np
import pandas as pd
from scipy import ndimage
from osgeo import gdalconst
import os
import multiprocessing as mp

import torch
import torch.nn as nn

import rasterOp
import tools
from functools import wraps
import time

def fn_timer(function):
  @wraps(function)
  def function_timer(*args, **kwargs):
    t0 = time.time()
    result = function(*args, **kwargs)
    t1 = time.time()
    print ("Total time running %s: %s seconds" %(function.__name__, str(t1-t0)))
    return result
  return function_timer

class ImageFilterProcess:
    def SuminOneStd_filter_callfun(self,buffer):
        """
        一个标准差总和滤波：以滑动窗口内像元值在一个标准差内的像素点像元值总和作为中心像素点的像元值
        :param buffer: 中心像元所在的局部范围的像元值，1行n列，ndarray类型
        :return: 过滤后中心像素点的像元值
        """
        meandata = np.nanmean(buffer)
        stddata = np.nanstd(buffer)
        # 一个标准差范围内的像元索引
        index = (buffer>=meandata-stddata)&(buffer<=meandata+stddata)
        n=2
        while(len(index)<1):
            index = (buffer >= meandata - n*stddata) & (buffer <= meandata + n*stddata)
            n = n+1
        #求和：像元值在一个标准差内的像元求和
        sumdata = buffer[index].sum()
        return sumdata
    def Mean_filter_callfun(self,buffer):
        '''
        均值滤波：以滑动窗口内像素值平均值作为中心像素点的像素值
        :param buffer:
        :return:
        '''
        return np.nanmean(buffer)
    def Ptp_filter_callfun(self,buffer):
        '''
        均值滤波：以滑动窗口内像素值平均值作为中心像素点的像素值
        :param buffer:
        :return:
        '''
        minValue = np.nanmin(buffer)
        maxValue = np.nanmax(buffer)
        return maxValue-minValue
    def direction_selected(self,buffer):
        '''
           分别计算八个方向与中心像元的距离和标准差，选择标准差最小、像素值与中心像元距离近的方向
           :param buffer: 传入窗口数据，1行n列,ndarray类型。窗口大小为单数，中心像元为窗口中间的像元。
           :return: 记录标准差最小方向上的像元，list类型
        '''
        windowsize = int(len(buffer) ** 0.5)
        r = int(np.floor(windowsize / 2))
        data = buffer.reshape(windowsize, windowsize)
        # 以标准差最小为原则选择方向
        centerRowIndex, centerColIndex = r, r
        minRowIndex, minColIndex, maxRowIndex, maxColIndex = 0, 0, windowsize - 1, windowsize - 1
        centerValue = data[centerRowIndex, centerColIndex]
        values2 = []  # 记录标准差小的方向上的像元值
        std2 = float('inf')
        dis2 = float('inf')
        # 分别计算八个方向与中心像元的距离和标准差，选择标准差最小、像素值与中心像元距离近的方向
        for xi in [-1, 0, 1]:
            for yi in [-1, 0, 1]:
                if (xi == 0) & (yi == 0):
                    continue
                surroundvalues = []
                surdis = 0
                # 获取某一方向上的值，计算与中心像元距离以及标准差
                for i in range(1, r + 1):
                    surRowIndex = centerRowIndex + yi * i
                    surColIndex = centerColIndex + xi * i
                    if (minColIndex <= surColIndex <= maxColIndex) & (minRowIndex <= surRowIndex <= maxRowIndex):
                        survalue = data[surRowIndex, surColIndex]
                        if survalue >= 0:
                            surroundvalues.append(survalue)
                            surdis = surdis + (centerValue - survalue) ** 2
                if len(surroundvalues) < r:
                    continue
                surroundvalues.append(centerValue)
                surstd = np.nanstd(surroundvalues)
                # 标准差优先
                if (std2 > surstd) | ((std2 == surstd) & (dis2 > surdis)):
                    dis2 = surdis
                    std2 = surstd
                    values2 = surroundvalues[:]
        return values2
    def Directionfilter_callfun(self, buffer, filterType):
        '''
        方向滤波器：以标准差最小选择确定滤波方向，根据filterType指定滤波方式，如中值滤波、均值滤波、极大值滤波
        :param buffer: 滤波是中心像元所在窗口的所有像元，1行n列
        :param filterType:指定滤波方式，如中值滤波、均值滤波、极大值滤波
        :return: 滤波结果，作为滤波后中心像元的值
        '''
        valueLst = self.direction_selected(buffer)
        if (valueLst is None) | (len(valueLst)==0):
            return 0
        result = 0
        if filterType == 'mean':
           result = np.nanmean(valueLst)
        elif filterType == 'median':
            result = np.nanmedian(valueLst)
        elif filterType == 'max':
            result = np.nanmax(valueLst)
        else:
            result = np.nanmean(valueLst)
        return  result

    @fn_timer
    def SuminOneStd_filter(self,indata,windowsize):
        '''
        滤波：以滑动窗口内像元值在一个标准差内的像素点像元值总和作为中心像素点的像元值进行滤波。
        origin为0，滤波器中心与中心像元重叠，输出大小等于输入。
        边缘扩展方式为‘reflect’ (d c b a | a b c d | d c b a)，通过反射最后一个像素的边缘来扩展输入。
        :param indata:输入图象
        :param windowsize:窗口大小，如windowsize=3，即与中心像元相邻1个像素点，半径为1.windowssize=5，半径为2.
        :return:
        '''
        filterdata = ndimage.generic_filter(indata,self.SuminOneStd_filter_callfun,size=windowsize)
        return filterdata

    @fn_timer
    def filter_scipy_generic_filter(self,indata,windowsize,statisticType):
        filterdata=None
        if statisticType == 'SuminOneStd':
            filterdata = ndimage.generic_filter(indata, self.SuminOneStd_filter_callfun, size=windowsize)
        elif statisticType == 'mean':
            filterdata = ndimage.generic_filter(indata, self.Mean_filter_callfun, size=windowsize)
        elif statisticType == 'median':
            filterdata = ndimage.median_filter(indata,size=windowsize)
        elif statisticType == 'max':
            filterdata = ndimage.maximum_filter(indata,size=windowsize)
        elif statisticType == 'min':
            filterdata = ndimage.minimum_filter(indata, size=windowsize)
        elif statisticType == 'ptp':
            filterdata = ndimage.generic_filter(indata, self.Ptp_filter_callfun, size=windowsize)
        else:
            filterdata = None
        return filterdata

    @fn_timer
    def filterByDirected_scipy_generic_filter(self,indata,windowsize,filterType):
        '''
        自适应方向滤波器
        :param indata: 图像数据，二维数组
        :param windowsize: 滑动窗口大小，如windowsize=3，则窗口半径为1.
        :param filterType: 滤波方式
        :return:
        '''
        filterdata = ndimage.generic_filter(indata,self.Directionfilter_callfun,size=windowsize,extra_keywords={'filterType':filterType})
        return filterdata

    @fn_timer
    def filter_user_defined(self,indata,rows, cols,windowsize,statisticTypeLst):
        surroundData = rasterOp.getSurroundsPixels_roundR(indata, rows, cols, int(np.floor(windowsize / 2)))
        surroundData_reshape = surroundData.reshape(rows * cols, windowsize * windowsize)
        pool = mp.Pool(3)
        multi_res = [pool.apply_async(tools.calstatistic, (surroundData_reshape[i, :], statisticTypeLst)) for i in
                     range(0, rows * cols)]
        statresultLst = [res.get() for res in multi_res]
        statresultArray = np.array(statresultLst)
        # 读入特征
        dataframe = pd.DataFrame()
        for i in range(0, len(statisticTypeLst)):
            colname = statisticTypeLst[i] + '_w' + str(windowsize)
            dataframe[colname] = statresultArray[:, i]
        return dataframe

class LandcoverdataProcess:
    def FROM_GLC_ImperiousSurface_Extracted(self,filepath,outpath):
        inDs,inBand,inData,noValue = rasterOp.getRasterData(filepath)
        inData[(inData < 80) | (inData > 89)] = 0
        inData[inData != 0] = 1
        noValue = 0
        basename = os.path.basename(filepath)
        outfile =  os.path.join(outpath,basename)
        rasterOp.outputResult_setOthers(inDs, inData, outfile, noValue, gdalconst.GDT_Byte)

    def Batch_FROM_GLC_ImperiousSurface_Extracted(self,fileDir,outDir):
        fileLst = ['70E_40N','70E_50N','80E_30N','80E_40N','80E_50N','90E_30N','90E_50N','100E_20N']
        fileLst.append('100E_30N')
        fileLst.append('100E_40N')
        fileLst.append('100E_50N')
        fileLst.append('110E_20N')
        fileLst.append('110E_30N')
        fileLst.append('110E_40N')
        fileLst.append('110E_50N')
        fileLst.append('110E_60N')
        fileLst.append('120E_30N')
        fileLst.append('120E_40N')
        fileLst.append('120E_50N')
        fileLst.append('120E_60N')
        fileLst.append('130E_50N')
        for f in fileLst:
            print(f)
            filepath = os.path.join(fileDir,f+'.tif')
            self.FROM_GLC_ImperiousSurface_Extracted(filepath,outDir)

class temptools:
    def removeNoData_Value(self,indir,outdir):
        '''
        GEE下载OLS NTL，影像的NoData值自动设置成0.我们这里把noData值设为255
        :param indir: 影像所在牡蛎
        :param outdir: 输出影像所在目录
        :return:
        '''
        #得到所有tif
        datanames = os.listdir(indir)
        for name in datanames:
            if name.endswith('.tif'):
                outpath = os.path.join(outdir,name)
                inpath = os.path.join(indir,name)
                rasterOp.changeNoDataValue(inpath, outpath, 255)

'''卷积方式做自适应方向滤波'''
class Net_DirFiletr(nn.Module):
    def __init__(self,radius):
        super(Net_DirFiletr,self).__init__()
        self.radius = radius
        kernel = self.getDirWeight()
        kernel = torch.FloatTensor(kernel).unsqueeze(dim=1)
        self.weight = nn.Parameter(data=kernel, requires_grad=False)
        self.pad = self.radius
    def getDirWeight(self):
        r = self.radius
        rowsLst = []
        for yi in [-1, 0, 1]:
            for xi in [-1, 0, 1]:
                if (yi == 0) & (xi == 0):
                    continue
                # create weights\n",
                winsize = r * 2 + 1
                rows = [[0] * winsize for i in range(winsize)]
                for i in range(r + 1):
                    rows[r + yi * i][r + xi * i] = 1
                rowsLst.append(rows)
        rowsLst = np.array(rowsLst)
        return rowsLst
    def forward(self,x):
        #local std = sqrt([sum(power(x,2)) - power(sum(x),2)/n]/(n-1))
        n_Pixels = self.radius+1
        x1 = torch.nn.functional.conv2d(x, self.weight, bias=None, stride=1, padding=self.pad, dilation=1, groups=1)
        mean_x = x1/n_Pixels
        y1 = (x1*x1)/n_Pixels
        x2 = x*x
        y2 = torch.nn.functional.conv2d(x2, self.weight, bias=None, stride=1, padding=self.pad, dilation=1, groups=1)
        var_x = (y2 - y1) / (n_Pixels-1)
        max_var_Index = torch.argmin(var_x,dim=1,keepdim=True)
        result = torch.gather(mean_x,1,max_var_Index)
        return result

def run_Net_DirFiletr(clipdata,radius):
    torch.cuda.empty_cache()
    net = Net_DirFiletr(radius)
    net.cuda()
    x = torch.FloatTensor(clipdata).unsqueeze(0).unsqueeze(0)
    x = x.cuda()
    result = net(x)
    result = result.cpu()
    result = result.squeeze(0).squeeze(0).numpy()
    net.cpu()
    return result

