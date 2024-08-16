import numpy as np
import matplotlib.pyplot as plt
#get_ipython().magic('matplotlib inline')
#import geopandas
from osgeo import gdal
import sys
from osgeo.gdalconst import *
from osgeo.osr import SpatialReference
import os
import time
import math
import scipy.stats as scipyStats
from sklearn import preprocessing
import csv
import pandas as pd
import numpy as np
import pandas as pd
import sklearn as skl
from sklearn import svm
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import cross_val_score
from sklearn.cluster import k_means
from sklearn.model_selection import GridSearchCV
#get_ipython().magic('matplotlib inline')
import shapely as shp
import pyproj as prj
import cv2
import fiona as fi
import time
import gc

'''
栅格操作
'''
#gdal 打开栅格数据
def readRasterFile(filepath):
    # open the image
    data = gdal.Open(filepath,GA_ReadOnly)
    if data is None:
        print('Could not open img')
        sys.exit(1)
    return data

#gdal 获取某一波段信息
def getBand(data,index):
    inBand = data.GetRasterBand(index)
    return inBand

#生成空白的新的栅格数据
#inData = inBand.ReadAsArray(0, 0, cols, rows).astype(np.float)
#outData = np.zeros((rows, cols,len(outfilenames)), np.float)
def createNewRasterData(rows,cols,bandcount,datatype):
    outData = np.zeros((rows, cols,bandcount), datatype)
    return outData

#gdal 读取波段数据
def getBandData(inDs,inBand):
    # get image size
    rows = inDs.RasterYSize
    cols = inDs.RasterXSize
 
    # read the input data
    inBand = inDs.GetRasterBand(1)
    inData = inBand.ReadAsArray(0, 0, cols, rows).astype(np.float)
    return inData
    
def getRasterData(path):
    inDs = readRasterFile(path)
    inBand = getBand(inDs,1)
    inData = getBandData(inDs,inBand)
    noValue = inBand.GetNoDataValue()
    return inDs,inBand,inData,noValue

#输出栅格数据
def getProj_img(tif_path):
    srcImage = gdal.Open(tif_path)
    geoProj = srcImage.GetProjection()
    return geoProj

def write_img(filename,im_proj,im_geotrans,im_data):
    if 'int8' in im_data.dtype.name:
        datatype = gdal.GDT_Byte
    elif 'int16' in im_data.dtype.name:
        datatype = gdal.GDT_UInt16
    else:
        datatype = gdal.GDT_Float32

    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1,im_data.shape

    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)
    if dataset is None:
        print('Could not create ')
        sys.exit(1)

    dataset.SetGeoTransform(im_geotrans)
    dataset.SetProjection(im_proj)
    if im_bands == 1:
        outBand = dataset.GetRasterBand(1)
        outBand.WriteArray(im_data, 0, 0)
        outBand.FlushCache()
    else:
        for i in range(im_bands):
            outBand = dataset.GetRasterBand(i+1)
            outBand.WriteArray(im_data[i], 0, 0)
            outBand.FlushCache()
    dataset = None
    del dataset

def write_img_setDataType(filename,im_proj,im_geotrans,im_data,datatype,novalue):
    # if 'int8' in im_data.dtype.name:
    #     datatype = gdal.GDT_Byte
    # elif 'int16' in im_data.dtype.name:
    #     datatype = gdal.GDT_UInt16
    # else:
    #     datatype = gdal.GDT_Float32
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1,im_data.shape

    driver = gdal.GetDriverByName("GTiff")
    dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)
    if dataset is None:
        print('Could not create ')
        sys.exit(1)

    dataset.SetGeoTransform(im_geotrans)
    dataset.SetProjection(im_proj)
    if im_bands == 1:
        outBand = dataset.GetRasterBand(1)
        outBand.SetNoDataValue(novalue)
        outBand.WriteArray(im_data, 0, 0)
        outBand.FlushCache()
    else:
        for i in range(im_bands):
            outBand = dataset.GetRasterBand(i+1)
            outBand.SetNoDataValue(novalue)
            outBand.WriteArray(im_data[i], 0, 0)
            outBand.FlushCache()
    dataset = None
    del dataset

def outputResult(inDs,outData,outpath):
    gdal.AllRegister()
    # create the output image
    driver = inDs.GetDriver()

    rows = inDs.RasterYSize
    cols = inDs.RasterXSize

    outDs = driver.Create(outpath, cols, rows, 1, GDT_Float32)
    if outDs is None:
        print('Could not create ')
        sys.exit(1)
    outBand = outDs.GetRasterBand(1)

    # write the output data
    outBand.WriteArray(outData, 0, 0)

    # flush data to disk, set the NoData value and calculate stats
    outBand.FlushCache()
    stats = outBand.GetStatistics(0, 1)

    # georeference the image and set the projection
    outDs.SetGeoTransform(inDs.GetGeoTransform())
    outDs.SetProjection(inDs.GetProjection())

    #inDs = None
    outDs = None

def outputResult_setNoDataValue(inDs, outData, outpath,noValue):
    gdal.AllRegister()
    # create the output image
    driver = inDs.GetDriver()

    rows = inDs.RasterYSize
    cols = inDs.RasterXSize

    outDs = driver.Create(outpath, cols, rows, 1, GDT_Float32)
    if outDs is None:
        print('Could not create ')
        sys.exit(1)
    outBand = outDs.GetRasterBand(1)
    outBand.SetNoDataValue(noValue)

    # write the output data
    outBand.WriteArray(outData, 0, 0)

    # flush data to disk, set the NoData value and calculate stats
    outBand.FlushCache()
    stats = outBand.GetStatistics(0, 1)

    # georeference the image and set the projection
    outDs.SetGeoTransform(inDs.GetGeoTransform())
    outDs.SetProjection(inDs.GetProjection())

    # inDs = None
    # outDs = None
    del outDs

def outputResult_setOthers(inDs, outData, outpath, noValue, datatype=GDT_Float32):
    '''
    :param inDs:
    :param outData:
    :param outpath:
    :param noValue:
    :param datatype: GDT_Float32/GDT_Byte
    :return:
    '''
    gdal.AllRegister()
    # create the output image
    driver = inDs.GetDriver()

    rows = inDs.RasterYSize
    cols = inDs.RasterXSize

    outDs = driver.Create(outpath, cols, rows, 1, datatype)
    if outDs is None:
        print('Could not create ')
        sys.exit(1)
    outBand = outDs.GetRasterBand(1)
    outBand.SetNoDataValue(noValue)

    # write the output data
    outBand.WriteArray(outData, 0, 0)

    # flush data to disk, set the NoData value and calculate stats
    outBand.FlushCache()
    stats = outBand.GetStatistics(0, 1)

    # georeference the image and set the projection
    outDs.SetGeoTransform(inDs.GetGeoTransform())
    outDs.SetProjection(inDs.GetProjection())

    # inDs = None
    # outDs = None
    del outDs

# 输出多波段栅格数据
def outputBands(inDs, outData, outpath, cols, rows, bandcount):
    # create the output image
    driver = inDs.GetDriver()

    outDs = driver.Create(outpath, cols, rows, bandcount, GDT_Float32)
    if outDs is None:
        print('Could not create tif')
        sys.exit(1)

    for i in range(0, bandcount):
        outBand = outDs.GetRasterBand(i + 1)
        # write the output data
        outBand.WriteArray(outData[:, :, i], 0, 0)
        outBand.FlushCache()
    outDs.SetGeoTransform(inDs.GetGeoTransform())
    outDs.SetProjection(inDs.GetProjection())
    outDs = None

def changeNoDataValue(inpath,outpath,new_noValue):
    inDs, inBand, inData, noValue = getRasterData(inpath)
    outputResult_setNoDataValue(inDs, inData, outpath, new_noValue)

'''DataFrame的列输出成栅格'''    
def saveDataFrameColAsRaster(dataframe,colname,inDs,outpath):
    rows = inDs.RasterYSize
    cols = inDs.RasterXSize
    resultData = dataframe[colname]
    resultData = resultData.reshape(rows,cols)
    outputResult(inDs,resultData,outpath)

#根据行号列号得到转为一维数组后的索引号
#rowIndex：行号，从0开始。  colIndex：列号，从0开始。  
#rows：总行数，1~n。  cols：总列数，1~n
#index:一维数组的索引号，从0开始
def getIndexByRowsAndCols(rowIndex,colIndex,rows,cols):
    index = rowIndex*cols+colIndex
    return index

#根据转为一维数组后的索引号计算原二维数组的行号和列号
#rowIndex：行号，从0开始。  colIndex：列号，从0开始。  
#rows：总行数，1~n。  cols：总列数，1~n
#Index:一维数组的索引号，从0开始
def getRowColByIndex(Index,rows,cols):
    ys = (Index+1)%cols
    ss = math.floor((Index+1)/cols)
    if ys == 0:
        colIndex = cols-1
        rowIndex = ss-1
    else:
        colIndex = ys -1
        rowIndex = ss
    return rowIndex,colIndex
 
#indexArray:二维图像reshape成1维数组，得到索引(索引列表需转化为数组)
#cols:二维图像的列数
#返回：根据索引号计算在二维图像中的行号和列号,X,Y分别为行数组和列数组
#索引、行号、列号都从0开始
def batch_getRowColByIndex(indexArray,cols):
    X = np.trunc(indexArray/cols)#相除取整
    Y = indexArray - X*cols
    return X,Y
    
#获取周边像素，得到一维数组的索引列表
#currowIndex,curcolIndex：当前的行号、列号
#rows,cols：总行数，列数
#radius：邻域范围半径，如3*3格网，则邻域半径填1
#nearbyType:通道类型，4 四通道， 8 八通道
#indexLst:一维数组的索引列表
def getSurroundPixels(currowIndex,curcolIndex,rows,cols,radius,nearbyType):
    indexLst = []
    #四通道
    if nearbyType == 4:
        for i in range(1,radius+1):
            if currowIndex+i<rows:
                indexLst.append(getIndexByRowsAndCols(currowIndex+i,curcolIndex,rows,cols))
            if currowIndex-i>-1:
                indexLst.append(getIndexByRowsAndCols(currowIndex-i,curcolIndex,rows,cols))
            if curcolIndex+i<cols:
                indexLst.append(getIndexByRowsAndCols(currowIndex,curcolIndex+i,rows,cols))
            if curcolIndex-i>-1:
                indexLst.append(getIndexByRowsAndCols(currowIndex,curcolIndex-i,rows,cols))
    else :
        #获取八通道最大最小行号
        if currowIndex - radius >=0:
            rmin = currowIndex - radius
        else:
            rmin = 0
        if currowIndex + radius < rows:
            rmax = currowIndex + radius
        else:
            rmax = rows-1
        #获取八通道最大最小列号   
        if curcolIndex - radius >=0:
            cmin = curcolIndex - radius
        else:
            cmin = 0
        if curcolIndex + radius < cols:
            cmax = curcolIndex + radius
        else:
            cmax = cols-1
        #遍历
        for i in range(rmin,rmax+1):
            for j in range(cmin,cmax+1):
                if((i!= currowIndex)|(j!=curcolIndex)):
                    index = getIndexByRowsAndCols(i,j,rows,cols)
                    if(index == rows*cols):
                        print('错误的行列：\n')
                        print(index,i,j)
                    indexLst.append(index)
    return indexLst
 
#获取以currowIndex,curcolIndex为中心的滑动窗口像素的索引列表
def getSlideWindowIndexs(curIndex,rows,cols,windowhalfRows,windowhalfColumns):
    indexLst=[]
    minRow = 0
    mincol = 0
    maxRow = rows - 1
    maxCol = cols - 1
    currowIndex,curcolIndex = getRowColByIndex(curIndex,rows,cols)
    if currowIndex-windowhalfRows > minRow:
        minRow = currowIndex - windowhalfRows
    if currowIndex + windowhalfColumns <maxRow:
        maxRow =  currowIndex + windowhalfColumns
    if curcolIndex - windowhalfColumns >mincol:
        mincol = curcolIndex - windowhalfColumns 
    if curcolIndex + windowhalfColumns <maxCol:
        maxCol = curcolIndex + windowhalfColumns
    for i in range(minRow,maxRow+1):
        for j in range(mincol,maxCol+1):
            index = getIndexByRowsAndCols(i,j,rows,cols)
            indexLst.append(index)
    return indexLst
    
'''
栅格转为DataFrame的某一列
path:栅格文件路径
colName:栅格名称或列名称
'''
def rasterTransferDataFrame(path,colName):
    inDs = readRasterFile(path)
    inBand = getBand(inDs,1)
    inData = getBandData(inDs,inBand)
    noValue = inBand.GetNoDataValue()
    inflattenData = inData.reshape(-1)
    rows = inDs.RasterYSize
    cols = inDs.RasterXSize
    dataframe = pd.DataFrame()  
    dataframe[colName] = inflattenData
    return inDs,inData,dataframe,rows,cols,noValue,colName

# 将pathLst栅格路径列表中的栅格转为1维后存在dataframe中，colNameLst为栅格在Dataframe中对应的列名   
def batch_rasterTransferDataFrame(pathLst,colNameLst):
    dataframe = pd.DataFrame() 
    for i in range(0,len(pathLst)):
        path = pathLst[i]
        colName = colNameLst[i]
        inDs = readRasterFile(path)
        inBand = getBand(inDs,1)
        inData = getBandData(inDs,inBand)        
        inflattenData = inData.reshape(-1)
        dataframe[colName] = inflattenData
        print(colName)
    rows = inDs.RasterYSize
    cols = inDs.RasterXSize 
    noValue = inBand.GetNoDataValue()
    return inDs,inData,dataframe,rows,cols,noValue
    
    
'''获取栅格周边像素'''
#3*3
def getPixels_round8(inData,rows,cols,isPadding,isincludeCenter):
    if isincludeCenter:
        t1 = createNewRasterData(rows,cols,9,np.float)
        t1[:,:,8] = inData
    else:
        t1 = createNewRasterData(rows,cols,8,np.float)
    t1[1:-1,1:-1,0] = inData[0:-2,0:-2]#左上
    t1[1:-1,0:-1,1] = inData[0:-2,0:-1]#上方
    t1[1:-1,0:-2,2] = inData[0:-2,1:-1]#右上
    t1[0:-1,1:-1,3] = inData[0:-1,0:-2]#左边
    t1[0:-1,0:-2,4] = inData[0:-1,1:-1]#右边
    t1[0:-2,1:-1,5] = inData[1:-1,0:-2]#左下
    t1[0:-2,0:-1,6] = inData[1:-1,0:-1]#下方
    t1[0:-2,0:-2,7] = inData[1:-1,1:-1]#右下
    if isPadding == False:
        t1[0,:,0] = np.nan
        t1[-1,:,0] = np.nan
        t1[:,0,0] = np.nan
        t1[:,-1,0] = np.nan
    return t1

'''R*R全部像元,R为与中心像元相邻的距离，如3*3窗口，R为1'''    
def getSurroundsPixels_roundR(inData,rows,cols,R):
    tRows = rows+2*R
    tCols = cols+2*R
    bandcount = (2*R+1)*(2*R+1)
    transData = np.zeros((tRows, tCols), np.float)
    transData[:,:] = np.nan
    transData[R:R+rows,R:R+cols]=inData[:,:]
    newData = np.zeros((rows, cols,bandcount), np.float)
    for p in range(0,2*R+1):
        for q in range(0,2*R+1):
            l = p*(2*R+1)+q
            newData[:,:,l] = transData[p:p+rows,q:q+cols]
    return newData
    
    '''（2R+1）*(2R+1)横向像元,R为与中心像元相邻的距离，如3*3窗口，R为1'''    
def getSurroundsPixels_horizonalR(inData,rows,cols,R):
    tRows = rows+2*R
    tCols = cols+2*R
    bandcount = (2*R+1)
    transData = np.zeros((tRows, tCols), np.float)
    transData[:,:] = np.nan
    transData[R:R+rows,R:R+cols]=inData[:,:]
    newData = np.zeros((rows, cols,bandcount), np.float)
    
    for q in range(0,2*R+1):
        l = q
        newData[:,:,l] = transData[R:R+rows,q:q+cols]
    return newData
    
    '''（2R+1）*(2R+1)纵向像元,R为与中心像元相邻的距离，如3*3窗口，R为1'''    
def getSurroundsPixels_vertialR(inData,rows,cols,R):
    tRows = rows+2*R
    tCols = cols+2*R
    bandcount = (2*R+1)
    transData = np.zeros((tRows, tCols), np.float)
    transData[:,:] = np.nan
    transData[R:R+rows,R:R+cols]=inData[:,:]
    newData = np.zeros((rows, cols,bandcount), np.float)
    for p in range(0,2*R+1):
        l = p
        newData[:,:,l] = transData[p:p+rows,R:R+cols]
    return newData
    
    '''（2R+1）*(2R+1)空洞像元。卷积大小为3*3.R为与中心像元相邻的距离，如3*3窗口，R为1'''    
def getSurroundsPixels_dilatedR(inData,rows,cols,R):
    tRows = rows+2*R
    tCols = cols+2*R
    bandcount = 9
    transData = np.zeros((tRows, tCols), np.float)
    transData[:,:] = np.nan
    transData[R:R+rows,R:R+cols]=inData[:,:]
    newData = np.zeros((rows, cols,bandcount), np.float)
    #左上
    newData[:,:,0] = transData[0:rows,0:cols]
    #左中
    newData[:,:,1] = transData[R:R+rows,0:cols]
    #左下
    newData[:,:,2] = transData[2*R:2*R+rows,0:cols]
    #正上
    newData[:,:,3] = transData[0:rows,R:R+cols]
    #正中
    newData[:,:,4] = transData[R:R+rows,R:R+cols]
    #正下
    newData[:,:,5] = transData[2*R:2*R+rows,R:R+cols]
    #右上
    newData[:,:,6] = transData[0:rows,2*R:2*R+cols]
    #右中
    newData[:,:,7] = transData[R:R+rows,2*R:2*R+cols]
    #右下
    newData[:,:,8] = transData[2*R:2*R+rows,2*R:2*R+cols]
    return newData
    
def getSurroundsPixels_Mean_dilatedR(inData,rows,cols,R):#3*3窗口，R=1
    tRows = rows+2*R
    tCols = cols+2*R
    bandcount = 9
    transData = np.zeros((tRows, tCols), np.float)
    transData[:,:] = np.nan
    transData[R:R+rows,R:R+cols]=inData[:,:]
    newData = np.zeros((rows, cols,bandcount), np.float)
    #左上
    newData[:,:,0] = transData[0:rows,0:cols]
    #左中
    newData[:,:,1] = transData[R:R+rows,0:cols]
    #左下
    newData[:,:,2] = transData[2*R:2*R+rows,0:cols]
    #正上
    newData[:,:,3] = transData[0:rows,R:R+cols]
    #正中
    newData[:,:,4] = transData[R:R+rows,R:R+cols]
    #正下
    newData[:,:,5] = transData[2*R:2*R+rows,R:R+cols]
    #右上
    newData[:,:,6] = transData[0:rows,2*R:2*R+cols]
    #右中
    newData[:,:,7] = transData[R:R+rows,2*R:2*R+cols]
    #右下
    newData[:,:,8] = transData[2*R:2*R+rows,2*R:2*R+cols]
    #平均
    result = np.nanmean(newData,axis=2)
    del newData,transData
    gc.collect()
    return result
 
#分别计算八个方向与中心像元的距离和标准差，选择与中心像元距离最近、标准差最小的方向,返回该方向的均值与最小值
def calPixelDirection_Surrounds(series,curIndex,rows,cols,r):#3*3窗口，r=1
    minRow = 0
    mincol = 0
    maxRow = rows - 1
    maxCol = cols - 1
    currowIndex,curcolIndex = getRowColByIndex(curIndex,rows,cols)
    centerValue = series[curIndex]
    std0= float('-inf')
    dis0= float('-inf')
    values0=[]
    std1= float('inf')
    dis1= float('inf')
    values1=[]
    #分别计算八个方向与中心像元的距离和标准差，选择与中心像元距离最近、标准差最小的方向
    x=-2
    y=-2
    for xi in [-1,0,1]:
        for yi in [-1,0,1]:
            if (xi==0) & (yi==0):
                continue
            surroundvalues = []
            surdis = 0
            #获取某一方向上的值，计算与中心像元距离以及标准差
            for i in range(1,r+1):
                surRow = currowIndex+yi*i
                surCol = curcolIndex+xi*i
                if (mincol<=surCol<=maxCol)&(minRow<=surRow<=maxRow):
                    surindex = getIndexByRowsAndCols(surRow,surCol,rows,cols)
                    survalue=series[surindex]
                    if survalue >= 0:
                        surroundvalues.append(survalue)
                        surdis = surdis+(centerValue-survalue)**2
            if len(surroundvalues) == 0:
                continue
            surroundvalues.append(centerValue)
            surstd = np.nanstd(surroundvalues)
            #比较距离与标准差，更新
            #与中心差异小的方向
            if (dis1 > surdis) | ((dis1 == surdis) & (std1 > surstd)):
                dis1 = surdis
                std1 = surstd
                values1 = surroundvalues
                x=xi
                y=yi
            #与中心差异大的方向
            if (dis0 < surdis) | ((dis0 == surdis) & (std0 < surstd)):
                dis0 = surdis
                std0 = surstd
                values0 = surroundvalues
                x=xi
                y=yi
    #返回均值与最大值
    max_meanValue = np.nanmean(values0)
    max_maxValue = np.nanmax(values0)
    max_medianValue = np.nanmedian(values0)
    min_meanValue = np.nanmean(values1)
    min_maxValue = np.nanmax(values1)
    min_medianValue = np.nanmedian(values1)
    return max_meanValue,max_maxValue,max_medianValue,min_meanValue,min_maxValue,min_medianValue

#选择与中心像元差异最小的方向的数据作为邻域信息，将均值与最大值作为特征          
def  getSurroundsPixels_byDirection(df,rows,cols,R,colname): #3*3窗口，R=1              
    series = df[colname]
    df['max_mean'] = np.zeros(rows*cols,np.float)
    df['max_max'] = np.zeros(rows*cols,np.float)
    df['max_median'] = np.zeros(rows*cols,np.float)
    df['min_mean'] = np.zeros(rows*cols,np.float)
    df['min_max'] = np.zeros(rows*cols,np.float)
    df['min_median'] = np.zeros(rows*cols,np.float)
    for ite in series.items():
        max_meanValue,max_maxValue,max_medianValue,min_meanValue,min_maxValue,min_medianValue = calPixelDirection_Surrounds(series,ite[0],rows,cols,R)
        df.loc[ite[0],'max_mean']=max_meanValue
        df.loc[ite[0],'max_max']=max_maxValue
        df.loc[ite[0],'max_median']=max_medianValue
        df.loc[ite[0],'min_mean']=min_meanValue
        df.loc[ite[0],'min_max']=min_maxValue
        df.loc[ite[0],'min_median']=min_medianValue
    return df

#分别计算八个方向与中心像元的距离和标准差，选择与中心像元距离最近、标准差最小的方向,返回该方向的均值与最小值
#不同的判定方式
def calPixelDirection_Surrounds_DifferCalMethod(series,curIndex,rows,cols,r):#3*3窗口，r=1
    minRow = 0
    mincol = 0
    maxRow = rows - 1
    maxCol = cols - 1
    currowIndex,curcolIndex = getRowColByIndex(curIndex,rows,cols)
    centerValue = series[curIndex]
    #1、比较中心点与方向上每个像元的距离平方和
    std1= float('inf')
    dis1= float('inf')
    values1=[]#记录距离小的
    values2=[]#记录标准差小的
    std2= float('inf')
    dis2= float('inf')
    #2、比较中心点与方向上的均值距离
    dis3= float('inf')
    values3=[]#记录距离小的
    #3、掉一个最高点和最低点，比较中心像元与剩下像元的距离平方和
    dis4= float('inf')
    values4=[]#记录距离小的
    #分别计算八个方向与中心像元的距离和标准差，选择与中心像元距离最近、标准差最小的方向
    for xi in [-1,0,1]:
        for yi in [-1,0,1]:
            if (xi==0) & (yi==0):
                continue
            surroundvalues = []
            surdis = 0
            #获取某一方向上的值，计算与中心像元距离以及标准差
            for i in range(1,r+1):
                surRow = currowIndex+yi*i
                surCol = curcolIndex+xi*i
                if (mincol<=surCol<=maxCol)&(minRow<=surRow<=maxRow):
                    surindex = getIndexByRowsAndCols(surRow,surCol,rows,cols)
                    survalue=series[surindex]
                    if survalue >= 0:
                        surroundvalues.append(survalue)
                        surdis = surdis+(centerValue-survalue)**2
            if len(surroundvalues) < r:
                continue
            surroundvalues.append(centerValue)
            surstd = np.nanstd(surroundvalues)
            #1、比较中心点与方向上每个像元的距离平方和
            #与中心差异小的方向，距离优先
            if (dis1 > surdis) | ((dis1 == surdis) & (std1 > surstd)):
                dis1 = surdis
                std1 = surstd
                values1 = surroundvalues[:]
            #标准差优先
            if (std2 >surstd)|((std1==surstd)&(dis2 > surdis)):
                dis2 = surdis
                std2 = surstd
                values2 = surroundvalues[:]
            #2、比较中心点与方向上的均值距离
            surroundvalues.pop(-1)
            mean = np.mean(surroundvalues)
            surdis = (centerValue-mean)**2
            if(dis3 > surdis):
                dis3 = surdis
                surroundvalues.append(centerValue)
                values3 = surroundvalues[:]
                surroundvalues.pop(-1)
            #3、去掉一个最高点和最低点，比较中心像元与剩下像元的距离平方和
            surroundvalues.sort()
            surroundvalues.pop(0)
            if(len(surroundvalues)==0):
                continue
            surroundvalues.pop(-1)
            if(len(surroundvalues)==0):
                continue
            surdis=0
            for i in range(0,len(surroundvalues)):
                surdis = surdis+(centerValue-surroundvalues[i])**2
            if dis4 > surdis:
                dis4 = surdis
                surroundvalues.append(centerValue)
                values4 = surroundvalues[:]
    #返回特征值
    #均值
    dis_mean = np.nanmean(values1)
    std_mean = np.nanmean(values2)
    noceter_mean = np.nanmean(values3)
    removehighlow_mean= np.nanmean(values4)
    return dis_mean,std_mean,noceter_mean,removehighlow_mean

#分别计算八个方向与中心像元的距离和标准差，选择与中心像元标准差最小的方向,返回该方向的均值与最小值
def calPixelDirection_Surrounds_Std(series,curIndex,rows,cols,r):#3*3窗口，r=1
    minRow = 0
    mincol = 0
    maxRow = rows - 1
    maxCol = cols - 1
    currowIndex, curcolIndex = getRowColByIndex(curIndex, rows, cols)
    centerValue = series[curIndex]
    # 1、比较中心点与方向上每个像元的距离平方和
    std1 = float('inf')
    dis1 = float('inf')
    values1 = []  # 记录距离小的
    values2 = []  # 记录标准差小的
    std2 = float('inf')
    dis2 = float('inf')
    # 2、比较中心点与方向上的均值距离
    dis3 = float('inf')
    values3 = []  # 记录距离小的
    # 3、掉一个最高点和最低点，比较中心像元与剩下像元的距离平方和
    dis4 = float('inf')
    values4 = []  # 记录距离小的
    # 分别计算八个方向与中心像元的距离和标准差，选择与中心像元距离最近、标准差最小的方向
    for xi in [-1, 0, 1]:
        for yi in [-1, 0, 1]:
            if (xi == 0) & (yi == 0):
                continue
            surroundvalues = []
            surdis = 0
            # 获取某一方向上的值，计算与中心像元距离以及标准差
            for i in range(1, r + 1):
                surRow = currowIndex + yi * i
                surCol = curcolIndex + xi * i
                if (mincol <= surCol <= maxCol) & (minRow <= surRow <= maxRow):
                    surindex = getIndexByRowsAndCols(surRow, surCol, rows, cols)
                    survalue = series[surindex]
                    if survalue >= 0:
                        surroundvalues.append(survalue)
                        surdis = surdis + (centerValue - survalue) ** 2
            if len(surroundvalues) < r:
                continue
            surroundvalues.append(centerValue)
            surstd = np.nanstd(surroundvalues)
            # 1、比较中心点与方向上每个像元的距离平方和
            # # 与中心差异小的方向，距离优先
            # if (dis1 > surdis) | ((dis1 == surdis) & (std1 > surstd)):
            #     dis1 = surdis
            #     std1 = surstd
            #     values1 = surroundvalues[:]
            # 标准差优先
            if (std2 > surstd) | ((std2 == surstd) & (dis2 > surdis)):
                dis2 = surdis
                std2 = surstd
                values2 = surroundvalues[:]
            # # 2、比较中心点与方向上的均值距离
            # surroundvalues.pop(-1)
            # mean = np.mean(surroundvalues)
            # surdis = (centerValue - mean) ** 2
            # if (dis3 > surdis):
            #     dis3 = surdis
            #     surroundvalues.append(centerValue)
            #     values3 = surroundvalues[:]
            #     surroundvalues.pop(-1)
            # # 3、去掉一个最高点和最低点，比较中心像元与剩下像元的距离平方和
            # surroundvalues.sort()
            # surroundvalues.pop(0)
            # if (len(surroundvalues) == 0):
            #     continue
            # surroundvalues.pop(-1)
            # if (len(surroundvalues) == 0):
            #     continue
            # surdis = 0
            # for i in range(0, len(surroundvalues)):
            #     surdis = surdis + (centerValue - surroundvalues[i]) ** 2
            # if dis4 > surdis:
            #     dis4 = surdis
            #     surroundvalues.append(centerValue)
            #     values4 = surroundvalues[:]
    # 返回特征值
    # 均值
    # dis_mean = np.nanmean(values1)
    std_mean = np.nanmean(values2)
    # noceter_mean = np.nanmean(values3)
    # removehighlow_mean = np.nanmean(values4)
    return std_mean


def filterBySobel(inData):
    sobel_x = [[-1,0,1],[-2,0,2],[-1,0,1]]
    sobel_y = [[1,2,1],[0,0,0],[-1,-2,-1]]
    sobel_x = np.array(sobel_x)
    sobel_y = np.array(sobel_y)
    rows = inData.shape[0]
    cols = inData.shape[1]
    #数据padding = 1,用边缘行列数据赋值给扩展数据
    originData = np.zeros((rows+2, cols+2), np.float)
    originData[1:-1,1:-1] = inData[:,:]
    originData[0,1:-1] = inData[0,:]
    originData[-1,1:-1] = inData[-1,:]
    originData[1:-1,0] = inData[:,0]
    originData[1:-1,-1] = inData[:,-1]
    originData[0,0] = inData[0,0]
    originData[0,-1] = inData[0,-1]
    originData[-1,0] = inData[-1,0]
    originData[-1,-1] = inData[-1,-1]
    #sobel_x
    x_Data = np.zeros((rows,cols),np.float)
    x_Data[:,:] = originData[0:-2,0:-2]*sobel_x[0,0]+originData[0:-2,1:-1]*sobel_x[0,1]+originData[0:-2,2:]*sobel_x[0,2]+originData[1:-1,0:-2]*sobel_x[1,0]+originData[1:-1,1:-1]*sobel_x[1,1]+originData[1:-1,2:]*sobel_x[1,2]+originData[2:,0:-2]*sobel_x[2,0]+originData[2:,1:-1]*sobel_x[2,1]+originData[2:,2:]*sobel_x[2,2]
    #sobel_y
    y_Data = np.zeros((rows, cols), np.float)
    y_Data[:,:] = originData[0:-2,0:-2] * sobel_y[0, 0] + originData[0:-2, 1:-1] * sobel_y[0, 1] + originData[0:-2,2:] * sobel_y[0, 2] + originData[1:-1, 0:-2] * sobel_y[1, 0] + originData[1:-1, 1:-1] * sobel_y[1, 1] + originData[1:-1,2:] * sobel_y[1, 2] + originData[2:, 0:-2] * sobel_y[2, 0] + originData[2:, 1:-1] * sobel_y[2, 1] + originData[2:, 2:] * sobel_y[2, 2]
    return x_Data,y_Data


#膨胀
def dilate(data):
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    dilate_data = cv2.dilate(data,kernel)
    return dilate_data


'''Kappa精度计算'''
'''
1、创建记录精度的dataframe
'''
def createAccuracyDataFrame(filepath):
    '''
    创建记录精度的dataframe
    :param filepath: 用于存储精度的csv路径
    :return:记录精度的dataframe
    '''
    colLst = ["citynumber",'cityname','method','Jaccard','OA','Kappa','PA','UA','HA','KIA1','OE','CE','NUM','ref_f','ref_t','pre_f',
              'pre_t','tt','preT_refF','preF_refT','ff']
    # accuracy_dataframe = pd.DataFrame(columns=colLst,dtype=object)
    accuracy_dataframe = pd.DataFrame(columns=colLst)
    if (os.path.isfile(filepath) == False):
        accuracy_dataframe.to_csv(filepath,header=True,index=False,encoding='gb2312')
    return accuracy_dataframe
'''
2、计算精度
ref_data:参照数据，n行1列
predict_data:预测数据，n行1列
mask_data：掩膜数据，行政区范围裁剪，n行1列
novalue:mask_data中的空值
'''
def calKappaWithMask(accuracy_dataframe,index,ref_data,predict_data,mask_data,novalue):
    data = pd.DataFrame()
    data['ref'] = ref_data
    data['pre'] = predict_data
    data['mask'] = mask_data
    subdata = data[data['mask']>novalue]
    sub_ref_data = subdata['ref'].values
    sub_pre_data = subdata['pre'].values
    ja, OA, Kappa, PA, UA, HA, kia1, OE, CE, num, ref_f, ref_t, pre_f, pre_t, tt, tf, ft, ff = calKappa(sub_ref_data,sub_pre_data)
    accuracy_dataframe.loc[index, 'Jaccard'] = ja
    accuracy_dataframe.loc[index, 'OA'] = OA
    accuracy_dataframe.loc[index, 'Kappa'] = Kappa
    accuracy_dataframe.loc[index, 'PA'] = PA
    accuracy_dataframe.loc[index, 'UA'] = UA
    accuracy_dataframe.loc[index, 'HA'] = HA
    accuracy_dataframe.loc[index, 'KIA1'] = kia1
    accuracy_dataframe.loc[index, 'OE'] = OE
    accuracy_dataframe.loc[index, 'CE'] = CE
    accuracy_dataframe.loc[index, 'NUM'] = num
    accuracy_dataframe.loc[index, 'ref_f'] = ref_f
    accuracy_dataframe.loc[index, 'ref_t'] = ref_t
    accuracy_dataframe.loc[index, 'pre_f'] = pre_f
    accuracy_dataframe.loc[index, 'pre_t'] = pre_t
    accuracy_dataframe.loc[index, 'tt'] = tt
    accuracy_dataframe.loc[index, 'preT_refF'] = tf
    accuracy_dataframe.loc[index, 'preF_refT'] = ft
    accuracy_dataframe.loc[index, 'ff'] = ff
    return accuracy_dataframe
'''
计算精度
'''
def calKappa(ref_data,predict_data):
    data = np.c_[ref_data,predict_data]
    resultframe = pd.DataFrame(data,columns=['ref','pre'])
    ff = len(resultframe[(resultframe['ref']<0)&(resultframe['pre']<0)])
    tt = len(resultframe[(resultframe['ref']>0)&(resultframe['pre']>0)])
    tf = len(resultframe[(resultframe['ref']<0)&(resultframe['pre']>0)])
    ft = len(resultframe[(resultframe['ref']>0)&(resultframe['pre']<0)])
    ref_f = len(resultframe[resultframe['ref']<0])
    ref_t = len(resultframe[resultframe['ref']>0])
    pre_f = len(resultframe[resultframe['pre']<0])
    pre_t = len(resultframe[resultframe['pre']>0])
    num = len(resultframe)
    OA = (ff+tt)*1.0/num
    if tt+ft == 0:
        PA = 0
    else:
        PA = tt*1.0/(tt+ft)
    if tt+tf == 0:
        UA = 0
    else:
        UA = tt*1.0/(tt+tf)
    if (PA==0)|(UA==0):
        HA=0
    else:
        HA = 2.0/(1.0/PA+1.0/UA)
    Pc = (ref_f*pre_f+ref_t*pre_t)*1.0/(num*num)
    Kappa = (OA - Pc) / (1 - Pc)*1.0
    OE = 1-PA#漏分误差
    CE = 1-UA #错分误差
    #kia1
    p11 = tt/num
    p1 = pre_t/num
    pt1 = ref_t/num
    if p1-p1*pt1 == 0:
        kia1 = 0
    else:
        kia1 = (p11-p1*pt1)/ (p1-p1*pt1)
    #jaccard
    ja = tt/(tt+ft+tf)
    return ja,OA,Kappa,PA,UA,HA,kia1,OE,CE,num,ref_f,ref_t,pre_f,pre_t,tt,tf,ft,ff
def calKappa_numpy(ref_data,predict_data):
    ff = len(ref_data[(ref_data<0)&(predict_data<0)])
    tt = len(ref_data[(ref_data>0)&(predict_data>0)])
    tf = len(ref_data[(ref_data<0)&(predict_data>0)])
    ft = len(ref_data[(ref_data>0)&(predict_data<0)])
    ref_f = len(ref_data[ref_data<0])
    ref_t = len(ref_data[ref_data>0])
    pre_f = len(predict_data[predict_data<0])
    pre_t = len(predict_data[predict_data>0])
    #计算指标
    num = len(predict_data)
    OA = (ff+tt)*1.0/num
    if tt+ft == 0:
        PA = -99
        OE = -99  # 漏分误差
    else:
        PA = tt*1.0/(tt+ft)
        OE = 1 - PA  # 漏分误差
    if tt+tf == 0:
        UA = -99
        CE = -99 # 错分误差
    else:
        UA = tt*1.0/(tt+tf)
        CE = 1 - UA  # 错分误差
    if (PA==0)|(UA==0):
        HA=-99
    else:
        HA = 2.0/(1.0/PA+1.0/UA)
    Pc = (ref_f*pre_f+ref_t*pre_t)*1.0/(num*num)
    if 1 - Pc == 0:
        Kappa = -99
    else:
        Kappa = (OA - Pc) / (1 - Pc)*1.0
    #kia1
    p11 = tt/num
    p1 = pre_t/num
    pt1 = ref_t/num
    if p1-p1*pt1 == 0:
        kia1 = -99
    else:
        kia1 = (p11-p1*pt1)/ (p1-p1*pt1)
    #jaccard
    if tt+ft+tf == 0:
        ja = -99
    else:
        ja = tt/(tt+ft+tf)
    return ja,OA,Kappa,PA,UA,HA,kia1,OE,CE,num,ref_f,ref_t,pre_f,pre_t,tt,tf,ft,ff
def calKappa_confuseMatrix(tt,ff,tf,ft):
    ref_f = ff+tf
    ref_t = tt+ft
    pre_f = ff+ft
    pre_t = tt+tf
    #计算指标
    num = tt+ff+tf+ft
    OA = (ff+tt)*1.0/num
    if tt+ft == 0:
        PA = -99
        OE = -99  # 漏分误差
    else:
        PA = tt*1.0/(tt+ft)
        OE = 1 - PA  # 漏分误差
    if tt+tf == 0:
        UA = -99
        CE = -99 # 错分误差
    else:
        UA = tt*1.0/(tt+tf)
        CE = 1 - UA  # 错分误差
    if (PA==0)|(UA==0):
        HA=-99
    else:
        HA = 2.0/(1.0/PA+1.0/UA)
    Pc = (ref_f*pre_f+ref_t*pre_t)*1.0/(num*num)
    if 1 - Pc == 0:
        Kappa = -99
    else:
        Kappa = (OA - Pc) / (1 - Pc)*1.0
    #kia1
    p11 = tt/num
    p1 = pre_t/num
    pt1 = ref_t/num
    if p1-p1*pt1 == 0:
        kia1 = -99
    else:
        kia1 = (p11-p1*pt1)/ (p1-p1*pt1)
    #jaccard
    if tt+ft+tf == 0:
        ja = -99
    else:
        ja = tt/(tt+ft+tf)
    return ja,OA,Kappa,PA,UA,HA,kia1,OE,CE,num,ref_f,ref_t,pre_f,pre_t,tt,tf,ft,ff

'''
3、记录精度
''';
def appendToAccuracyFile(filepath,accuracy_dataframe):
    accuracy_dataframe.to_csv(filepath, mode='a', header=False,index=False,encoding='gb2312')

def calKappa_appendToAccuracyFile(csvfile,cityname,citynum,method,ref_data,predict_data):
    accuracy_dataframe = createAccuracyDataFrame(csvfile)
    index = accuracy_dataframe.shape[0]
    accuracy_dataframe.loc[index] = 1
    accuracy_dataframe.loc[index, ['citynumber', 'cityname', 'method']] = [citynum, cityname, method]
    ja, OA, Kappa, PA, UA, HA, kia1, OE, CE, num, ref_f, ref_t, pre_f, pre_t, tt, tf, ft, ff = calKappa(ref_data,predict_data)
    accuracy_dataframe.loc[index, 'Jaccard'] = ja
    accuracy_dataframe.loc[index, 'OA'] = OA
    accuracy_dataframe.loc[index, 'Kappa'] = Kappa
    accuracy_dataframe.loc[index, 'PA'] = PA
    accuracy_dataframe.loc[index, 'UA'] = UA
    accuracy_dataframe.loc[index, 'HA'] = HA
    accuracy_dataframe.loc[index, 'KIA1'] = kia1
    accuracy_dataframe.loc[index, 'OE'] = OE
    accuracy_dataframe.loc[index, 'CE'] = CE
    accuracy_dataframe.loc[index, 'NUM'] = num
    accuracy_dataframe.loc[index, 'ref_f'] = ref_f
    accuracy_dataframe.loc[index, 'ref_t'] = ref_t
    accuracy_dataframe.loc[index, 'pre_f'] = pre_f
    accuracy_dataframe.loc[index, 'pre_t'] = pre_t
    accuracy_dataframe.loc[index, 'tt'] = tt
    accuracy_dataframe.loc[index, 'preT_refF'] = tf
    accuracy_dataframe.loc[index, 'preF_refT'] = ft
    accuracy_dataframe.loc[index, 'ff'] = ff
    appendToAccuracyFile(csvfile,accuracy_dataframe)

#数据预处理
def rasterRead_basePrcess(rasterdata,replaceValue=0):
    rasterdata = np.where((rasterdata<0)|(np.isnan(rasterdata)),replaceValue,rasterdata)
    rasterdata = np.where(np.isinf(rasterdata),replaceValue,rasterdata)
    rasterdata = np.where(np.isneginf(rasterdata),replaceValue,rasterdata)
    return rasterdata

