import os
import numpy as np
from osgeo import gdal,ogr,osr,gdalconst
import bisect
import pandas as pd
import gc

import rasterOp
import vectorOp
from commontools.common import FileOpr
from largeRasterImageOp import clipRaster_byEnvelope_shp_batchGrids,extractRaster_Rectangle_bySamplePoints,extractRaster_byRefRaster,clipRaster_byEnvelope_shp_sigleGrid


'''设置RNTL对应年份'''
def getRNTL_YearDict():
    rntlyeardict = {}
    for year in range(1992,1998):
        rntlyeardict[year] = 1996
    for year in range(1998,2000):
        rntlyeardict[year] = 1999
    for year in range(2000,2002):
        rntlyeardict[year] = 2000
    for year in range(2002,2004):
        rntlyeardict[year] = 2003
    for year in range(2004,2008):
        rntlyeardict[year] = 2005
    for year in range(2008,2014):
        rntlyeardict[year] = 2010
    return rntlyeardict

class GetTrainDataSet():
    def get_region_DataSet(self):
        # eval_gis.py SRImgRegionAccess clip_toGet_input_Data()
        return 0

    #img04
    #裁剪出img04对应的图像块
    def get_img04_DataSet(self):
        # 生成样本 img04
        namefield = 'Name'
        shpdatafile = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\04\shp\rect.shp'
        outRoot = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img04'
        tifpath = []
        tardir = []
        # tifpath.append(r'D:\01data\00整理\02夜间灯光\npp\annualV2\VNL_2013.tif');tardir.append('2013_VNL');
        # tifpath.append(r'D:\01data\00整理\04NDVI\landsatNDVI\NDVI_2013.tif');tardir.append('2013_NDVI');
        # tifpath.append(r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\outdata\02oriGNL_stepSNL\04interp\2013_interp.tif');tardir.append('2013_oriDNL');
        # tifpath.append(r'D:\01data\00整理\02夜间灯光\npp\annualV2\VNL_2012.tif');tardir.append('2012_VNL');
        # tifpath.append(r'D:\01data\00整理\02夜间灯光\npp\Chen2012\resample_align\chenNTL_2012.tif');tardir.append('2012_Chen');
        # tifpath.append(r'D:\01data\00整理\04NDVI\landsatNDVI\NDVI_2012.tif');tardir.append('2012_NDVI');
        # tifpath.append(r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\outdata\02oriGNL_stepSNL\04interp\2012_interp.tif');tardir.append('2012_oriDNL');
        # tifpath.append(r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\outdata\cf_cvg\cf_cvg2012.tif');tardir.append('2012_CfCvg');
        # tifpath.append(r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\outdata\cf_cvg\cf_cvg2013.tif');tardir.append('2013_CfCvg');
        # tifpath.append(r'D:\01data\00整理\02夜间灯光\grc_interp\2010_interp.tif');tardir.append('2010_RNTL');
        # tifpath.append(r'D:\01data\00整理\04NDVI\AVHRR\2013_AVHRR_500m.tif');tardir.append('2013_AVHRR');
        tifpath.append( r'D:\01data\00整理\干旱半干旱区域\Aridity Index\watermask_AridityIndex.tif');tardir.append('Water_Aridity');
        for j in range(len(tifpath)):
            inRasFile = tifpath[j]
            out_folder = os.path.join(outRoot,tardir[j])
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)
            clipRaster_byEnvelope_shp_batchGrids(shpdatafile, inRasFile, out_folder,namefield)

    # # 生成样本 img
    # 裁剪出img对应的图像块
    def get_img_DataSet(self):
        namefield = 'Name'
        shpdatafile = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\selected_1degree_alignGAIA.shp'
        outRoot = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img'
        tifpath = []
        tardir = []
        # tifpath.append(r'D:\01data\00整理\02夜间灯光\npp\annualV2\VNL_2013.tif');tardir.append('2013_VNL');
        # tifpath.append(r'D:\01data\00整理\04NDVI\landsatNDVI\NDVI_2013.tif');tardir.append('2013_NDVI');
        # tifpath.append(r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\outdata\02oriGNL_stepSNL\04interp\2013_interp.tif');tardir.append('2013_oriDNL');
        # tifpath.append(r'D:\01data\00整理\02夜间灯光\npp\annualV2\VNL_2012.tif');tardir.append('2012_VNL');
        # tifpath.append(r'D:\01data\00整理\02夜间灯光\npp\Chen2012\resample_align\chenNTL_2012.tif');tardir.append('2012_Chen');
        # tifpath.append(r'D:\01data\00整理\04NDVI\landsatNDVI\NDVI_2012.tif');tardir.append('2012_NDVI');
        # tifpath.append(r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\outdata\02oriGNL_stepSNL\04interp\2012_interp.tif');tardir.append('2012_oriDNL');
        # tifpath.append(r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\outdata\cf_cvg\cf_cvg2012.tif');tardir.append('2012_CfCvg');
        # tifpath.append(r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\outdata\cf_cvg\cf_cvg2013.tif');tardir.append('2013_CfCvg');
        # tifpath.append(r'D:\01data\00整理\02夜间灯光\grc_interp\2010_interp.tif');tardir.append('2010_RNTL');
        tifpath.append(r'D:\01data\00整理\干旱半干旱区域\Aridity Index\watermask_AridityIndex.tif');tardir.append('Water_Aridity');
        # tifpath.append(r'D:\01data\00整理\04NDVI\AVHRR\2013_AVHRR_500m.tif');tardir.append('2013_AVHRR');
        for j in range(len(tifpath)):
            inRasFile = tifpath[j]
            out_folder = os.path.join(outRoot,tardir[j])
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)
            clipRaster_byEnvelope_shp_batchGrids(shpdatafile, inRasFile, out_folder,namefield)

    #img03
    #初次创建img03
    def create_img03_DataSet_first(self):
        samplePointsfile = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\03\samplePoints.shp'
        outdir = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img03'
        tifpath = []
        tardir = []
        # tifpath.append(r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\outdata\cf_cvg\cf_cvg2012.tif');tardir.append('2012_CfCvg');
        # tifpath.append(r'D:\01data\00整理\02夜间灯光\grc_interp\2010_interp.tif');tardir.append('2010_RNTL');
        # tifpath.append(r'D:\01data\00整理\02夜间灯光\npp\annualV2\VNL_2013.tif');tardir.append('2013_VNL');
        # tifpath.append(r'D:\01data\00整理\04NDVI\landsatNDVI\NDVI_2013.tif');tardir.append('2013_NDVI');
        # tifpath.append(r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\outdata\02oriGNL_stepSNL\04interp\2013_interp.tif');tardir.append('2013_oriDNL');
        # tifpath.append(r'D:\01data\00整理\02夜间灯光\npp\annualV2\VNL_2012.tif');tardir.append('2012_VNL');
        # tifpath.append(r'D:\01data\00整理\02夜间灯光\npp\Chen2012\resample_align\chenNTL_2012.tif');tardir.append('2012_Chen');
        # tifpath.append(r'D:\01data\00整理\04NDVI\landsatNDVI\NDVI_2012.tif');tardir.append('2012_NDVI');
        # tifpath.append(r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\outdata\02oriGNL_stepSNL\04interp\2012_interp.tif');tardir.append('2012_oriDNL');
        tifpath.append(r'D:\01data\00整理\干旱半干旱区域\Aridity Index\water_AridityIndex.tif');tardir.append('Water_Aridity');
        namefield = 'Name'
        xOffset,yOffset,width,height = 128,128,256,256
        for j in range(0, len(tifpath)):
            rasterfile = tifpath[j]
            out_folder = os.path.join(outdir, tardir[j])
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)
            print(out_folder)
            extractRaster_Rectangle_bySamplePoints(samplePointsfile,rasterfile,out_folder,namefield,xOffset,yOffset,width,height)
    #根据img03的图像块裁剪其他数据
    def get_img03_DataSet(self):
        outdir = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img03'
        refdir = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img03\2012_VNL'
        tifpath = []
        tardir = []
        tifpath.append(r'D:\01data\00整理\干旱半干旱区域\Aridity Index\watermask_AridityIndex.tif');tardir.append('Water_Aridity');

        fileOp = FileOpr()
        filelst = fileOp.getFileNames_byExt(refdir,'.tif')
        for j in range(len(tifpath)):
            inpath = tifpath[j]
            out_folder = os.path.join(outdir, tardir[j])
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)
            for file in filelst:
                refpath = os.path.join(refdir,file)
                outpath = os.path.join(out_folder,file)
                if not os.path.exists(outpath):
                    print(file)
                    extractRaster_byRefRaster(inpath, outpath, refpath)

    #合并
    #将12年、13年训练图片合并到一个文件夹
    def combine_diffyears_img04_DataSet(self):
        dirInfolst = []
        # dirInfolst.append((r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img04\2012_2013VNL',
        #                   r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img04\2012_VNL',
        #                   '2012'))
        # dirInfolst.append((r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img04\2012_2013NDVI',
        #                   r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img04\2012_NDVI',
        #                   '2012'))
        # dirInfolst.append((r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img04\2012_2013NDVI',
        #                   r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img04\2013_NDVI',
        #                   '2013'))
        # dirInfolst.append((r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img04\2012_2013oriDNL',
        #                   r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img04\2012_oriDNL',
        #                   '2012'))
        # dirInfolst.append((r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img04\2012_2013oriDNL',
        #                   r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img04\2013_oriDNL',
        #                   '2013'))
        # dirInfolst.append((r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img04\2012_2013CfCvg',
        #                   r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img04\2013_CfCvg',
        #                   '2013'))
        # dirInfolst.append((r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img04\2012_2013CfCvg',
        #                   r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img04\2012_CfCvg',
        #                   '2012'))
        # dirInfolst.append((r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img04\2012_2013VNL',
        #                   r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img04\2012_VNL',
        #                   '2012'))
        fileOp = FileOpr()
        for dirInfo in dirInfolst:
            print(dirInfo)
            targetdir,sourcedir,prefix = dirInfo
            fileOp.copyTiffImg(sourcedir, targetdir, prefix)
    #生成对应的sampleTxt:将12年、13年训练图片合并到一个文件夹
    def getSampleTxt_combine_diffyears_img04_DataSet(self):
        fileOp = FileOpr()
        dirInfolst = []
        dirInfolst.append((r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\04\train_08.txt',
                           r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\04\train_07.txt',
                          '2012'))
        for dirInfo in dirInfolst:
            print(dirInfo)
            targetTxt,srcTxt,prefix = dirInfo
            fileOp.copyContent_addPrefix(srcTxt,targetTxt,prefix)
    def getSampleTxt_combine_diffyears_img03_DataSet(self):
        fileOp = FileOpr()
        dirInfolst = []
        dirInfolst.append((r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\03\train_img03.txt',
                           r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\03\train_04.txt',
                          '2012'))
        dirInfolst.append((r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\03\train_img03.txt',
                           r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\03\train_04.txt',
                          '2013'))
        dirInfolst.append((r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\03\test_img03.txt',
                           r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\03\test_05.txt',
                          '2012'))
        dirInfolst.append((r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\03\test_img03.txt',
                           r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\03\test_05.txt',
                          '2013'))
        dirInfolst.append((r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\03\valid_img03.txt',
                           r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\03\valid_05.txt',
                          '2012'))
        dirInfolst.append((r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\03\valid_img03.txt',
                           r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\03\valid_05.txt',
                          '2013'))
        for dirInfo in dirInfolst:
            print(dirInfo)
            targetTxt,srcTxt,prefix = dirInfo
            fileOp.copyContent_addPrefix(srcTxt,targetTxt,prefix)
    def getSampleTxt_combine_diffyears_img_DataSet(self):
        fileOp = FileOpr()
        dirInfolst = []
        dirInfolst.append((r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\01\train_img.txt',
                           r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\01\train.txt',
                          '2012'))
        dirInfolst.append((r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\01\train_img.txt',
                           r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\01\train.txt',
                          '2013'))
        dirInfolst.append((r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\01\test_img.txt',
                           r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\01\test.txt',
                          '2012'))
        dirInfolst.append((r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\01\test_img.txt',
                           r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\01\test.txt',
                          '2013'))
        dirInfolst.append((r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\01\valid_img.txt',
                           r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\01\valid.txt',
                          '2012'))
        dirInfolst.append((r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\01\valid_img.txt',
                           r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\01\valid.txt',
                          '2013'))
        for dirInfo in dirInfolst:
            print(dirInfo)
            targetTxt,srcTxt,prefix = dirInfo
            fileOp.copyContent_addPrefix(srcTxt,targetTxt,prefix)


    #world
    def get_world_img_Dataset(self):
        namefield = 'Name'
        shpdatafile = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\world\data\shp\grid\landmask_grids_2degree_overlap02.shp'
        outRoot = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\world\data\img'
        tifpath = []
        tardir = []
        tifpath.append(r'D:\01data\00整理\02夜间灯光\npp\annualV2\VNL_2013.tif');tardir.append('2013_VNL');
        tifpath.append(r'D:\01data\00整理\04NDVI\landsatNDVI\NDVI_2013.tif');tardir.append('2013_NDVI');
        tifpath.append(r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\outdata\02oriGNL_stepSNL\04interp\2013_interp.tif');tardir.append('2013_oriDNL');
        tifpath.append(r'D:\01data\00整理\02夜间灯光\npp\annualV2\VNL_2012.tif');tardir.append('2012_VNL');
        tifpath.append(r'D:\01data\00整理\04NDVI\landsatNDVI\NDVI_2012.tif');tardir.append('2012_NDVI');
        tifpath.append(r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\outdata\02oriGNL_stepSNL\04interp\2012_interp.tif');tardir.append('2012_oriDNL');
        tifpath.append(r'D:\01data\00整理\干旱半干旱区域\Aridity Index\watermask_AridityIndex.tif');
        tardir.append('Water_Aridity');
        for j in range(len(tifpath)):
            inRasFile = tifpath[j]
            out_folder = os.path.join(outRoot,tardir[j])
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)
            clipRaster_byEnvelope_shp_batchGrids(shpdatafile, inRasFile, out_folder,namefield)
    def get_singleRect_img_sample(self):
        shpdatafile = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\world\data\shp\grid\landmask_grids_2degree_overlap03.shp'
        outRoot = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\world\data\img'
        tifpath = []
        tardir = []
        tifpath.append(r'D:\01data\00整理\02夜间灯光\npp\annualV2\VNL_2013.tif');tardir.append('2013_VNL');
        tifpath.append(r'D:\01data\00整理\04NDVI\landsatNDVI\NDVI_2013.tif');tardir.append('2013_NDVI');
        tifpath.append(r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\outdata\02oriGNL_stepSNL\04interp\2013_interp.tif');tardir.append('2013_oriDNL');
        tifpath.append(r'D:\01data\00整理\02夜间灯光\npp\annualV2\VNL_2012.tif');tardir.append('2012_VNL');
        tifpath.append(r'D:\01data\00整理\04NDVI\landsatNDVI\NDVI_2012.tif');tardir.append('2012_NDVI');
        tifpath.append(r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\outdata\02oriGNL_stepSNL\04interp\2012_interp.tif');tardir.append('2012_oriDNL');
        tifpath.append(r'D:\01data\00整理\干旱半干旱区域\Aridity Index\watermask_AridityIndex.tif');
        tardir.append('Water_Aridity');
        for j in range(len(tifpath)):
            inRasFile = tifpath[j]
            out_folder = os.path.join(outRoot,tardir[j])
            if not os.path.exists(out_folder):
                os.makedirs(out_folder)
            out_tif = os.path.join(out_folder,'32.2_119.2.tif')
            if not os.path.exists(out_tif):
                clipRaster_byEnvelope_shp_sigleGrid(shpdatafile,inRasFile,out_tif,9303)
            out_tif = os.path.join(out_folder,'32.2_120.9.tif')
            if not os.path.exists(out_tif):
                clipRaster_byEnvelope_shp_sigleGrid(shpdatafile, inRasFile,out_tif, 9371)
    def updateDNVI_missingData(self):
        NDVIRoot = r'D:\01data\00整理\04NDVI\landsatNDVI\update'
        txtpath = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\world\data\shp\grid\NDVI_missing_grids.txt'
        file_list = tuple(open(txtpath, "r"))
        file_list = [id_.rstrip() for id_ in file_list]

        for year in range(1992,1999):
            print(year)
            for filename in file_list:
                path1999 = os.path.join(NDVIRoot,'ori','1999',filename+'.tif')
                path = os.path.join(NDVIRoot,'ori',str(year),filename+'.tif')
                outdir = os.path.join(NDVIRoot,'update',str(year))
                if not os.path.exists(outdir):
                    os.makedirs(outdir)
                outpath = os.path.join(outdir,filename+'.tif')
                inDs1999, inBand1999, inData1999, noValue1999 = rasterOp.getRasterData(path1999)
                inDs,inBand,inData,noValue = rasterOp.getRasterData(path)
                inData = np.where(np.isnan(inData),inData1999,inData)
                inData = np.where(np.isnan(inData)|np.isinf(inData)|np.isneginf(inData)| (inData<0),0,inData)
                rasterOp.outputResult(inDs,inData,outpath)
                del inData,inDs,inBand,noValue
            gc.collect()



    #dbf转为txt 样本
    def dbf_block_name_to_txt(self,dbfpath,saveTxtpath):
        fileop = FileOpr()
        df = fileop.readDBF_asDataFrame(dbfpath)
        subdf = df['Name']
        for index,item in subdf.iteritems():
            ss = str(item).strip()
            with open(saveTxtpath,'a') as f:
                f.write(ss+'\n')




#
getTrainDataSet = GetTrainDataSet()
# # getTrainDataSet.getSampleTxt_combine_diffyears_img03_DataSet()
# # getTrainDataSet.getSampleTxt_combine_diffyears_img_DataSet()
# # getTrainDataSet.get_img_DataSet()
# # getTrainDataSet.get_img03_DataSet()
# # getTrainDataSet.get_img04_DataSet()
# # getTrainDataSet.get_singleRect_imgDataSet()
#  #dbf转为txt 样本
# dir = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\world\data\shp\grid'
# namelst = ['grids_01','grids_02','grids_03','grids_04']
# for name in namelst:
#     dbfpath = os.path.join(dir,name+'.dbf')
#     saveTxtpath = os.path.join(dir,name+'.txt')
#     getTrainDataSet.dbf_block_name_to_txt(dbfpath,saveTxtpath)
# dbfpath = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\world\data\shp\grid\NDVI_missing_grids.dbf'
# saveTxtpath = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\world\data\shp\grid\NDVI_missing_grids.txt'
# getTrainDataSet.dbf_block_name_to_txt(dbfpath,saveTxtpath)
# getTrainDataSet.updateDNVI_missingData()