import numpy as np
import  pandas as pd
from osgeo import gdal
from osgeo import ogr,osr,gdalconst
import os
import pickle
import argparse

import matplotlib.pyplot as plt

import rasterOp
import vectorOp
import largeRasterImageOp


#########################实验一 NTL超分#############################
#********************剖面分析数据准备******************************#
def positionPoint_createRowImg_ColumnImg(outdir="",positionPointPath="",namefield="",vnlpath=""):
    '''
    position point 用来定位剖线位置，在ArcMap中选择生成位置点。
    获取VNL影像中position point的行号和列号
    生成一列行号影像和一行列号影像
    示例数据：以在北京地区，VNL影像为例
    :return:
    '''
    if outdir == "":
        outdir = r'D:\04study\00Paper\Dissertation\03report\03论文内容\图表\实验一NTL\04空间格局分析\02剖面分析\data\Beijing\point1'
    if positionPointPath=="":
        positionPointPath = r'D:\04study\00Paper\Dissertation\03report\03论文内容\图表\实验一NTL\04空间格局分析\02剖面分析\data\Beijing\point1\point_position.shp'
    if namefield == "":
        namefield = "Id"
    if vnlpath == "":
        vnlpath = r'D:\04study\00Paper\Dissertation\03report\03论文内容\图表\实验一NTL\04空间格局分析\01不同夜间灯光数据空间格局比较\data\2012\BJH_VNL.tif'
    vnlDs = gdal.Open(vnlpath, gdalconst.GA_ReadOnly)
    geoTrans = vnlDs.GetGeoTransform()
    geoProj = vnlDs.GetProjection()
    rowsize = vnlDs.RasterYSize
    colsize = vnlDs.RasterXSize
    #获得行列号
    locLst = largeRasterImageOp.getPointsInfo_shp(positionPointPath,namefield)
    x,y,_ = locLst[0]
    col_number, row_number = largeRasterImageOp.xy_to_rowcol(geoTrans,x,y)
    new_X,new_Y = largeRasterImageOp.rowcol_to_xy(geoTrans,row_number,col_number)
    #****生成一行列号影像
    #生成当前行数据
    colValues = [i+1 for i in range(0,colsize)]
    data = np.array(colValues).reshape((1,colsize))
    #设置当前行geoTrans
    geoTrans1 = list(geoTrans)
    geoTrans1[3] = new_Y
    #输出当前行
    outpath = os.path.join(outdir,'row'+str(row_number)+'.tif')
    rasterOp.write_img(outpath, geoProj, geoTrans1, data)
    # ****生成一列行号影像
    # 生成当前列数据
    rowValues = [i+1 for i in range(0,rowsize)]
    data = np.array(rowValues).reshape((rowsize,1))
    #设置当前列geoTrans
    geoTrans2 = list(geoTrans)
    geoTrans2[0] = new_X
    #输出当前行
    outpath = os.path.join(outdir,'col'+str(col_number)+'.tif')
    rasterOp.write_img(outpath, geoProj, geoTrans2, data)
def getRasterValues_bySelectedRowImg(outpath = "",startRow=None,endRow=None,startCol=None,endoCol=None,dataDict=None):
    '''
    行影像对应像元点提取数据
    :param outpath:
    :param startRow:
    :param endRow:
    :param startCol:
    :param endoCol:
    :param dataDict:
    :return:
    '''
    if outpath == "":
        outpath = r'D:\04study\00Paper\Dissertation\03report\03论文内容\图表\实验一NTL\04空间格局分析\02剖面分析\data\Beijing\point1\profiles_row192_1.csv'
    if startRow is None:
        startRow = 192
        endRow = 193
        startCol = 23
        endoCol = 336
    if dataDict is None:
        dataDict = {}
        dataDict['2012VNL'] = r'D:\04study\00Paper\Dissertation\03report\03论文内容\图表\实验一NTL\04空间格局分析\01不同夜间灯光数据空间格局比较\data\2012\BJH_VNL.tif'
        dataDict['2013VNL'] = r'D:\04study\00Paper\Dissertation\03report\03论文内容\图表\实验一NTL\04空间格局分析\01不同夜间灯光数据空间格局比较\data\2013\BJH_VNL.tif'
        dataDict['2012srDNL'] = r'D:\04study\00Paper\Dissertation\03report\03论文内容\图表\实验一NTL\04空间格局分析\01不同夜间灯光数据空间格局比较\data\2012\BJH_srDNL.tif'
        dataDict['2013srDNL'] = r'D:\04study\00Paper\Dissertation\03report\03论文内容\图表\实验一NTL\04空间格局分析\01不同夜间灯光数据空间格局比较\data\2013\BJH_srDNL.tif'
        dataDict['2012ENL'] = r'D:\04study\00Paper\Dissertation\03report\03论文内容\图表\实验一NTL\04空间格局分析\01不同夜间灯光数据空间格局比较\data\2012\BJH_ENL.tif'
        dataDict['2012DNL'] = r'D:\04study\00Paper\Dissertation\03report\03论文内容\图表\实验一NTL\04空间格局分析\01不同夜间灯光数据空间格局比较\data\2012\BJH_DNL.tif'
        dataDict['2013DNL'] = r'D:\04study\00Paper\Dissertation\03report\03论文内容\图表\实验一NTL\04空间格局分析\01不同夜间灯光数据空间格局比较\data\2013\BJH_DNL.tif'

    result = {}
    indexes = [i+1 for i in range(startCol,endoCol)]
    result['Index'] = indexes
    for key in dataDict.keys():
        _,_,data,_ = rasterOp.getRasterData(dataDict[key])
        data = np.where(np.isnan(data), 0, data)
        data[(data <= 0)] = 0
        selectedData =data[startRow:endRow,startCol:endoCol]
        selectedData = selectedData.reshape(-1)
        result[key] = selectedData
    df = pd.DataFrame(result)
    df.to_csv(outpath,header=True,index=False)
def getRasterValues_bySelectedColImg(outpath = "",startRow=None,endRow=None,startCol=None,endoCol=None,dataDict=None):
    '''
    列影像对应像元点提取数据
    :param outpath:
    :param startRow:
    :param endRow:
    :param startCol:
    :param endoCol:
    :param dataDict:
    :return:
    '''
    if outpath == "":
        outpath = r'D:\04study\00Paper\Dissertation\03report\03论文内容\图表\实验一NTL\04空间格局分析\02剖面分析\data\Beijing\point1\profiles_col142_1.csv'
    if startRow is None:
        startRow = 55
        endRow = 294
        startCol = 142
        endoCol = 143
    if dataDict is None:
        dataDict = {}
        dataDict['2012VNL'] = r'D:\04study\00Paper\Dissertation\03report\03论文内容\图表\实验一NTL\04空间格局分析\01不同夜间灯光数据空间格局比较\data\2012\BJH_VNL.tif'
        dataDict['2013VNL'] = r'D:\04study\00Paper\Dissertation\03report\03论文内容\图表\实验一NTL\04空间格局分析\01不同夜间灯光数据空间格局比较\data\2013\BJH_VNL.tif'
        dataDict['2012srDNL'] = r'D:\04study\00Paper\Dissertation\03report\03论文内容\图表\实验一NTL\04空间格局分析\01不同夜间灯光数据空间格局比较\data\2012\BJH_srDNL.tif'
        dataDict['2013srDNL'] = r'D:\04study\00Paper\Dissertation\03report\03论文内容\图表\实验一NTL\04空间格局分析\01不同夜间灯光数据空间格局比较\data\2013\BJH_srDNL.tif'
        dataDict['2012ENL'] = r'D:\04study\00Paper\Dissertation\03report\03论文内容\图表\实验一NTL\04空间格局分析\01不同夜间灯光数据空间格局比较\data\2012\BJH_ENL.tif'
        dataDict['2012DNL'] = r'D:\04study\00Paper\Dissertation\03report\03论文内容\图表\实验一NTL\04空间格局分析\01不同夜间灯光数据空间格局比较\data\2012\BJH_DNL.tif'
        dataDict['2013DNL'] = r'D:\04study\00Paper\Dissertation\03report\03论文内容\图表\实验一NTL\04空间格局分析\01不同夜间灯光数据空间格局比较\data\2013\BJH_DNL.tif'
    result = {}
    indexes = [i+1 for i in range(startRow,endRow)]
    result['Index'] = indexes
    for key in dataDict.keys():
        _,_,data,_ = rasterOp.getRasterData(dataDict[key])
        data = np.where(np.isnan(data), 0, data)
        data[(data <= 0)] = 0
        selectedData =data[startRow:endRow,startCol:endoCol]
        selectedData = selectedData.reshape(-1)
        result[key] = selectedData
    df = pd.DataFrame(result)
    df.to_csv(outpath,header=True,index=False)
def getRasterValues_bySelectedPoints(outpath="",pointpath = "",dataDict=None):
    '''
    根据剖线上的点提取影像值
    :param outpath:
    :param pointpath:
    :param dataDict:
    :return:
    '''
    if outpath == "":
        outpath = r'D:\04study\00Paper\Dissertation\03report\03论文内容\图表\实验一NTL\04空间格局分析\02剖面分析\data\LosAngeles\point1\profiles_l3_1.csv'
    if pointpath == "":
        pointpath = r'D:\04study\00Paper\Dissertation\03report\03论文内容\图表\实验一NTL\04空间格局分析\02剖面分析\data\LosAngeles\point1\l3_point.shp'
    if dataDict is None:
        dataDict = {}
        dataDict['2012VNL'] = r'D:\04study\00Paper\Dissertation\03report\03论文内容\图表\实验一NTL\04空间格局分析\01不同夜间灯光数据空间格局比较\data\2012\Log Angeles_VNL.tif'
        dataDict['2013VNL'] = r'D:\04study\00Paper\Dissertation\03report\03论文内容\图表\实验一NTL\04空间格局分析\01不同夜间灯光数据空间格局比较\data\2013\Log Angeles_VNL.tif'
        dataDict['2012srDNL'] = r'D:\04study\00Paper\Dissertation\03report\03论文内容\图表\实验一NTL\04空间格局分析\01不同夜间灯光数据空间格局比较\data\2012\Log Angeles_srDNL.tif'
        dataDict['2013srDNL'] = r'D:\04study\00Paper\Dissertation\03report\03论文内容\图表\实验一NTL\04空间格局分析\01不同夜间灯光数据空间格局比较\data\2013\Log Angeles_srDNL.tif'
        dataDict['2012ENL'] = r'D:\04study\00Paper\Dissertation\03report\03论文内容\图表\实验一NTL\04空间格局分析\01不同夜间灯光数据空间格局比较\data\2012\Log Angeles_ENL.tif'
        dataDict['2012DNL'] = r'D:\04study\00Paper\Dissertation\03report\03论文内容\图表\实验一NTL\04空间格局分析\01不同夜间灯光数据空间格局比较\data\2012\Log Angeles_DNL.tif'
        dataDict['2013DNL'] = r'D:\04study\00Paper\Dissertation\03report\03论文内容\图表\实验一NTL\04空间格局分析\01不同夜间灯光数据空间格局比较\data\2013\Log Angeles_DNL.tif'

    ras_pathLst = []
    rasNameLst = []
    for key in dataDict.keys():
        rasNameLst.append(key)
        ras_pathLst.append(dataDict[key])
    fieldLst = ['POINTID','pid']
    result_df = largeRasterImageOp.getRasterValueByPoints(pointpath,"",ras_pathLst,rasNameLst,fieldLst)
    result_df.to_csv(outpath,header=True,index=False)
#********************区域时间序列空间展示：用于绘制区域1992-2020夜间灯光总强度折线图******************************#
def getTNL_of_regions_Csv():
    regionlst = ['Beijing','GBA','LosAngeles']
    for region in regionlst:
        dir = r'D:\04study\00Paper\Dissertation\03report\03论文内容\图表\实验一NTL\04空间格局分析\03时间序列空间格局'
        tnllst = []
        for y in range(1992,2013):
            path = os.path.join(dir,region,'data',region+'_srDNL_'+str(y)+'.tif')
            _,_,data,_ = rasterOp.getRasterData(path)
            data = np.where(np.isnan(data), 0, data)
            data[(data <= 0)] = 0
            tnl = np.sum(data)
            tnllst.append(tnl)
        for y in range(2013,2021):
            path = os.path.join(dir,region,'data',region+'_VNL_'+str(y)+'.tif')
            _,_,data,_ = rasterOp.getRasterData(path)
            data = np.where(np.isnan(data), 0, data)
            data[(data <= 0)] = 0
            tnl = np.sum(data)
            tnllst.append(tnl)
        ylst = [y for y in range(1992,2021)]
        df = pd.DataFrame(data={'year':ylst,'tnl':tnllst})
        outpath = os.path.join(dir,region,'data','TNL_'+region+'1992-2020_set0.csv')
        df.to_csv(outpath,header=True,index=False)
def getENTL_TNL_of_regions_Csv():
    regionlst = ['Beijing','GBA','LosAngeles']
    for region in regionlst:
        dir = r'D:\04study\00Paper\Dissertation\03report\03论文内容\图表\实验一NTL\04空间格局分析\03时间序列空间格局'
        tnllst = []
        for y in range(2000,2013):
            path = os.path.join(dir,region,'data',region+'_ENL_'+str(y)+'.tif')
            _,_,data,_ = rasterOp.getRasterData(path)
            data = np.where(np.isnan(data), 0, data)
            data[(data <= 0)] = 0
            tnl = np.sum(data)
            tnllst.append(tnl)
        ylst = [y for y in range(2000,2013)]
        df = pd.DataFrame(data={'year':ylst,'tnl':tnllst})
        outpath = os.path.join(dir,region,'data','ENL_TNL_'+region+'2000-2012.csv')
        df.to_csv(outpath,header=True,index=False)






