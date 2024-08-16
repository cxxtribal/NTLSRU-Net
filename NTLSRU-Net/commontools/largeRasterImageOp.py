#裁剪栅格数据、制作样本
import os
import numpy as np
from osgeo import gdal,ogr,osr,gdalconst
import bisect
import pandas as pd
import gc

import rasterOp
import vectorOp

'''栅格行列号与地理空间坐标转换'''
def xy_to_rowcol(extend,x,y):
    '''
    根据GDAL的六参数模型将给定的投影或地理坐标转为影像图上坐标（行列号）
    :param extend: geoTrans
    :param x: 投影坐标x
    :param y: 投影坐标y
    :return: 对应的行列号（col,row)
    '''
    a = np.array([[extend[1],extend[2]],[extend[4],extend[5]]])
    b = np.array([x-extend[0],y - extend[3]])
    row_col = np.linalg.solve(a,b) #进行二元一次方程求解
    row = int(np.floor(row_col[1]))
    col = int(np.floor(row_col[0]))
    return col,row
def rowcol_to_xy(extend,row,col):
    '''
    根据GDAL的六参数模型将影像图上坐标（行列号）转为投影坐标或地理坐标
    :param extend:geoTrans
    :param row:像元的行号
    :param col:像元的列号
    :return:
    '''
    x = extend[0]+col*extend[1]+row*extend[2]
    y = extend[3]+col*extend[4]+row*extend[5]
    return x,y
def world2Pixel(geoMatrix, x, y):
  """
  Uses a gdal geomatrix (gdal.GetGeoTransform()) to calculate
  the pixel location of a geospatial coordinate
  """
  ulX = geoMatrix[0]
  ulY = geoMatrix[3]
  xDist = geoMatrix[1]
  yDist = geoMatrix[5]
  pixel = int(np.floor(((x - ulX) / xDist)))
  line = int(np.floor(((y - ulY) / yDist)))
  return (pixel, line)


'''#矢量裁剪栅格   格网无空隙裁剪  矢量多边形掩膜'''
def clipRaster_byextent(extent,rasterdata,geoTrans,geoProj, save_path):
    minX, maxX, minY, maxY = extent
    colnums = rasterdata.RasterXSize
    rownums = rasterdata.RasterYSize
    rasMinX,rasMaxX,rasMinY,rasMaxY = 0,colnums-1,0,rownums-1
    #计算行数和列数
    ulX, ulY = xy_to_rowcol(geoTrans, minX, maxY)
    lrX, lrY = xy_to_rowcol(geoTrans, maxX, minY)
    #比较是否查出栅格范围
    if ulX<rasMinX:
        ulX = rasMinX
    if ulY <rasMinY:
        ulY = rasMinY
    if lrX > rasMaxX:
        lrX = rasMaxX
    if lrY > rasMaxY:
        lrY = rasMaxY
    if (ulX>rasMaxX) | (ulY>rasMaxY) |(lrX<rasMinX)|(lrY<rasMinY):
        return None,geoProj, geoTrans
    # Calculate the pixel size of the new image
    pxWidth = int(lrX - ulX)
    pxHeight = int(lrY - ulY)
    # clip = srcArray[:, ulY:lrY, ulX:lrX]
    clip = rasterdata.ReadAsArray(ulX, ulY, pxWidth, pxHeight)  # ***只读要的那块***
    # Create a new geomatrix for the image
    new_minX,new_maxY = rowcol_to_xy(geoTrans,ulY,ulX)
    geoTrans = list(geoTrans)
    geoTrans[0] = new_minX
    geoTrans[3] = new_maxY
    #输出
    if save_path != "":
        rasterOp.write_img(save_path, geoProj, geoTrans, clip)
    return clip, geoProj, geoTrans
def clipRaster_byEnvelope(feature,rasterdata,geoTrans,geoProj, save_path):
    geometry = feature.GetGeometryRef()
    minX, maxX, minY, maxY = geometry.GetEnvelope()
    colnums = rasterdata.RasterXSize
    rownums = rasterdata.RasterYSize
    rasMinX,rasMaxX,rasMinY,rasMaxY = 0,colnums-1,0,rownums-1
    #计算行数和列数
    ulX, ulY = xy_to_rowcol(geoTrans, minX, maxY)
    lrX, lrY = xy_to_rowcol(geoTrans, maxX, minY)
    #比较是否查出栅格范围
    if ulX<rasMinX:
        ulX = rasMinX
    if ulY <rasMinY:
        ulY = rasMinY
    if lrX > rasMaxX:
        lrX = rasMaxX
    if lrY > rasMaxY:
        lrY = rasMaxY
    if (ulX>rasMaxX) | (ulY>rasMaxY) |(lrX<rasMinX)|(lrY<rasMinY):
        return None,geoProj, geoTrans
    # Calculate the pixel size of the new image
    pxWidth = int(lrX - ulX)
    pxHeight = int(lrY - ulY)
    # clip = srcArray[:, ulY:lrY, ulX:lrX]
    clip = rasterdata.ReadAsArray(ulX, ulY, pxWidth, pxHeight)  # ***只读要的那块***
    # Create a new geomatrix for the image
    new_minX,new_maxY = rowcol_to_xy(geoTrans,ulY,ulX)
    geoTrans = list(geoTrans)
    geoTrans[0] = new_minX
    geoTrans[3] = new_maxY
    #输出
    if save_path != "":
        rasterOp.write_img(save_path, geoProj, geoTrans, clip)
    return clip, geoProj, geoTrans
def clipRaster_byEnvelope_shp_sigleGrid(shpdatafile, rasterfile, out_tif,fid):
    '''
    用面的外包矩形裁剪矢量。适合格网裁剪。无缝。但面不是矩形时，裁剪得到的是外包框范围，没有掩膜。
    shapefile面矢量裁剪栅格文件并保存，只用指定的一个要素裁剪栅格数据.
    :param shpdatafile: 矢量文件路径
    :param rasterfile:栅格文件路径
    :param out_tif:输出栅格文件路径，若为""，则不输出。
    :return:
    '''
    rasterdata = gdal.Open(rasterfile)
    geoTrans = rasterdata.GetGeoTransform()
    geoProj = rasterdata.GetProjection()

    shpdata = ogr.Open(shpdatafile)
    lyr = shpdata.GetLayer( os.path.split( os.path.splitext( shpdatafile )[0] )[1] )
    feature = lyr.GetFeature(fid)
    clip, geoProj, geoTrans = clipRaster_byEnvelope(feature,rasterdata,geoTrans,geoProj,out_tif)
    return clip,geoProj,geoTrans

def clipRaster_byEnvelope_shp_batchGrids(shpdatafile, rasterfile, out_folder,namefield,postfix = ''):
    '''
    批量格网裁剪
    :param shpdatafile: 格网矢量文件路径
    :param rasterfile: 待裁剪栅格路径
    :param out_folder: 输出目录路径
    :param namefield: 格网矢量文件中的字段，用于作为输出tif的名称
    :return:
    '''
    rasterdata = gdal.Open(rasterfile)
    geoTrans = rasterdata.GetGeoTransform()
    geoProj = rasterdata.GetProjection()

    shpdata = ogr.Open(shpdatafile)
    lyr = shpdata.GetLayer( os.path.split( os.path.splitext( shpdatafile )[0] )[1])
    feature = lyr.GetNextFeature()
    while feature:
        name = feature.GetField(namefield)
        save_path = os.path.join(out_folder,name+postfix+'.tif')
        if not os.path.exists(save_path):
            print(name)
            # Convert the layer extent to image pixel coordinates
            clipRaster_byEnvelope(feature,rasterdata,geoTrans,geoProj,save_path)
        feature = lyr.GetNextFeature()
    print('finish!')
#world
shpdatafile = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\world\data\shp\grid\landmask_grids_2degree_overlap03.shp'
namefield = 'Name'
in_dmsp_dir = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\outdata\02oriGNL_stepSNL\04interp'
in_ndvi_dir = r'D:\01data\00整理\04NDVI\landsatNDVI'
out_root = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\world\data\img'
# yearlst = [1992,1996,2000,2004,2008]
yearlst = [2018]
for i in range(len(yearlst)):
    year = yearlst[i]
    print('DMSP',year)
    out_dmsp_dir = os.path.join(out_root,'oriDNL',str(year))
    if not os.path.exists(out_dmsp_dir):
        os.makedirs(out_dmsp_dir)
    in_dmsp_path = os.path.join(in_dmsp_dir,str(year)+'_interp.tif')
    clipRaster_byEnvelope_shp_batchGrids(shpdatafile, in_dmsp_path, out_dmsp_dir, namefield)
for i in range(len(yearlst)):
    year = yearlst[i]
    print('NDVI', year)
    out_ndvi_dir = os.path.join(out_root, 'NDVI', str(year))
    if not os.path.exists(out_ndvi_dir):
        os.makedirs(out_ndvi_dir)
    in_ndvi_path = os.path.join(in_ndvi_dir,'NDVI_'+str(year)+'.tif')
    clipRaster_byEnvelope_shp_batchGrids(shpdatafile, in_ndvi_path, out_ndvi_dir, namefield)

# #测试单个面裁剪：矢量掩膜用gdal.Warp；栅格用网格分块用clipRaster_byEnvelope_shp_sigleGrid()
# # shpdatafile = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\grid\litmasked_1degree_alignGAIA.shp'
# # rasterfile1 = r'D:\01data\00整理\02夜间灯光\npp\annualV2\VNL_2013.tif'
# # rasterfile2 = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\outdata\02oriGNL_stepSNL\03avg\2013.tif'
# # out_tif1 = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img\2013_VNL\f2_31_120.tif'
# # out_tif2 = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img\2013_DNL\f2_31_120.tif'
# # featIndex = 8932
# # # #方法1：Warp 裁剪，根据矢量形状裁剪，适合做矢量掩膜。但是裁剪得到的栅格与矢量线间有空白。并不能用于格网裁剪，不能得到无缝裁剪结果。
# ds = gdal.Warp(out_tif2, # 裁剪图像保存完整路径（包括文件名）
#                    rasterfile2, # 待裁剪的影像
#                    format='GTiff', # 保存图像的格式
#                    cutlineDSName=shpdatafile, # 矢量文件的完整路径
#                    cropToCutline=True, # 保证裁剪后影像大小跟矢量文件的图框大小一致（设置为False时，结果图像大小会跟待裁剪影像大小一样，则会出现大量的空值区域）,
#                    cutlineWhere = 'FID = 8868',
#                    resampleAlg=gdalconst.GRA_NearestNeighbour,
#                    dstNodata=0)
# # #方法2:适合格网裁剪，不适合矢量掩膜
# # # clipRaster_byEnvelope_shp_sigleGrid(shpdatafile,rasterfile1,out_tif1,featIndex)
# # # clipRaster_byEnvelope_shp_sigleGrid(shpdatafile,rasterfile2,out_tif2,featIndex)
# #


# # 生成样本 img04
# namefield = 'Name'
# shpdatafile = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\04\shp\rect.shp'
# outRoot = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img04'
# tifpath = []
# tardir = []
# # tifpath.append(r'D:\01data\00整理\02夜间灯光\npp\annualV2\VNL_2013.tif');tardir.append('2013_VNL');
# # tifpath.append(r'D:\01data\00整理\04NDVI\landsatNDVI\NDVI_2013.tif');tardir.append('2013_NDVI');
# # tifpath.append(r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\outdata\02oriGNL_stepSNL\04interp\2013_interp.tif');tardir.append('2013_oriDNL');
# # tifpath.append(r'D:\01data\00整理\02夜间灯光\npp\annualV2\VNL_2012.tif');tardir.append('2012_VNL');
# # tifpath.append(r'D:\01data\00整理\02夜间灯光\npp\Chen2012\resample_align\chenNTL_2012.tif');tardir.append('2012_Chen');
# # tifpath.append(r'D:\01data\00整理\04NDVI\landsatNDVI\NDVI_2012.tif');tardir.append('2012_NDVI');
# # tifpath.append(r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\outdata\02oriGNL_stepSNL\04interp\2012_interp.tif');tardir.append('2012_oriDNL');
# # tifpath.append(r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\outdata\cf_cvg\cf_cvg2012.tif');tardir.append('2012_CfCvg');
# tifpath.append(r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\outdata\cf_cvg\cf_cvg2013.tif');tardir.append('2013_CfCvg');
# # tifpath.append(r'D:\01data\00整理\02夜间灯光\grc_interp\2010_interp.tif');tardir.append('2010_RNTL');
# for j in range(len(tifpath)):
#     inRasFile = tifpath[j]
#     out_folder = os.path.join(outRoot,tardir[j])
#     if not os.path.exists(out_folder):
#         os.makedirs(out_folder)
#     clipRaster_byEnvelope_shp_batchGrids(shpdatafile, inRasFile, out_folder,namefield)
#
#
# # 生成样本 img
# namefield = 'Name'
# shpdatafile = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\selected_1degree_alignGAIA.shp'
# outRoot = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img'
# tifpath = []
# tardir = []
# # tifpath.append(r'D:\01data\00整理\02夜间灯光\npp\annualV2\VNL_2013.tif');tardir.append('2013_VNL');
# # tifpath.append(r'D:\01data\00整理\04NDVI\landsatNDVI\NDVI_2013.tif');tardir.append('2013_NDVI');
# # tifpath.append(r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\outdata\02oriGNL_stepSNL\04interp\2013_interp.tif');tardir.append('2013_oriDNL');
# # tifpath.append(r'D:\01data\00整理\02夜间灯光\npp\annualV2\VNL_2012.tif');tardir.append('2012_VNL');
# # tifpath.append(r'D:\01data\00整理\02夜间灯光\npp\Chen2012\resample_align\chenNTL_2012.tif');tardir.append('2012_Chen');
# # tifpath.append(r'D:\01data\00整理\04NDVI\landsatNDVI\NDVI_2012.tif');tardir.append('2012_NDVI');
# # tifpath.append(r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\outdata\02oriGNL_stepSNL\04interp\2012_interp.tif');tardir.append('2012_oriDNL');
# # tifpath.append(r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\outdata\cf_cvg\cf_cvg2012.tif');tardir.append('2012_CfCvg');
# tifpath.append(r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\outdata\cf_cvg\cf_cvg2013.tif');tardir.append('2013_CfCvg');
# # tifpath.append(r'D:\01data\00整理\02夜间灯光\grc_interp\2010_interp.tif');tardir.append('2010_RNTL');
# for j in range(len(tifpath)):
#     inRasFile = tifpath[j]
#     out_folder = os.path.join(outRoot,tardir[j])
#     if not os.path.exists(out_folder):
#         os.makedirs(out_folder)
#     clipRaster_byEnvelope_shp_batchGrids(shpdatafile, inRasFile, out_folder,namefield)

#生成区域输入数据patches
# tifpath = []
# tardir = []
# tifpath.append(r'D:\01data\00整理\02夜间灯光\npp\annualV2\VNL_2013.tif');tardir.append('2013_VNL');
# tifpath.append(r'D:\01data\00整理\04NDVI\landsatNDVI\NDVI_2013.tif');tardir.append('2013_NDVI');
# tifpath.append(r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\outdata\02oriGNL_stepSNL\04interp\2013_interp.tif');tardir.append('2013_oriDNL');
# tifpath.append(r'D:\01data\00整理\02夜间灯光\npp\annualV2\VNL_2012.tif');tardir.append('2012_VNL');
# tifpath.append(r'D:\01data\00整理\02夜间灯光\npp\Chen2012\resample_align\chenNTL_2012.tif');tardir.append('2012_Chen');
# tifpath.append(r'D:\01data\00整理\04NDVI\landsatNDVI\NDVI_2012.tif');tardir.append('2012_NDVI');
# tifpath.append(r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\outdata\02oriGNL_stepSNL\04interp\2012_interp.tif');tardir.append('2012_oriDNL');
# namefield = 'Name'
# dirnameLst = ['01','02','03','04','05']
# shpnameLst = ['BJH','NewYork','cairo','YRD','GBA']
# outRoot = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\out\\'
# for i in range(len(dirnameLst)):
#     print(shpnameLst[i])
#     shpdatafile = outRoot+dirnameLst[i]+'\\'+shpnameLst[i]+'_1degree_overlap04.shp'
#     for j in range(len(tifpath)):
#         inRasFile = tifpath[j]
#         out_folder = outRoot+dirnameLst[i]+'\\inputs\\'+tardir[j]
#         if not os.path.exists(out_folder):
#             os.makedirs(out_folder)
#         clipRaster_byEnvelope_shp_batchGrids(shpdatafile, inRasFile, out_folder,namefield)

# #world
# namefield = 'Name'
# dmsp_rasterfile = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\outdata\02oriGNL_stepSNL\03snapToVIIRS\2012.tif'
# ndvi_rasterfile = r'D:\01data\00整理\04NDVI\landsatNDVI\NDVI_2012.tif'
# viirs_rasterfile = r'D:\01data\00整理\02夜间灯光\npp\annualV2\VNL_2012.tif'
# shpdatafile = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\world\imgIDs\litmaskselected_grids_1degree_overlap04.shp'
# dmsp_out_folder = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\world\inputs\2012\oriDNL'
# viirs_out_folder = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\world\inputs\2012\VNL'
# ndvi_out_folder = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\world\inputs\2012\NDVI'
# infile_lst = [dmsp_rasterfile,ndvi_rasterfile,viirs_rasterfile]
# outdir_lst = [dmsp_out_folder,ndvi_out_folder,viirs_out_folder]
# for i in range(0,len(infile_lst)):
#     if not os.path.exists(outdir_lst[i]):
#         os.makedirs(outdir_lst[i])
#     clipRaster_byEnvelope_shp_batchGrids(shpdatafile, infile_lst[i], outdir_lst[i],namefield)

# # #测试批量格网裁剪
# outdir = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img02'
# shpdit = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\02'
# shpname = ['train_shp','test_shp','gt50_valid','lt10_gt0_valid','lt50_gt10_valid']
# tifpath = []
# tardir = []
# tifpath.append(r'D:\01data\00整理\02夜间灯光\npp\annualV2\VNL_2013.tif');tardir.append('2013_VNL');
# tifpath.append(r'D:\01data\00整理\04NDVI\landsatNDVI\NDVI_2013.tif');tardir.append('2013_NDVI');
# tifpath.append(r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\outdata\02oriGNL_stepSNL\04interp\2013_interp.tif');tardir.append('2013_oriDNL');
# tifpath.append(r'D:\01data\00整理\02夜间灯光\npp\annualV2\VNL_2012.tif');tardir.append('2012_VNL');
# tifpath.append(r'D:\01data\00整理\02夜间灯光\npp\Chen2012\resample_align\chenNTL_2012.tif');tardir.append('2012_Chen');
# tifpath.append(r'D:\01data\00整理\04NDVI\landsatNDVI\NDVI_2012.tif');tardir.append('2012_NDVI');
# tifpath.append(r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\outdata\02oriGNL_stepSNL\04interp\2012_interp.tif');tardir.append('2012_oriDNL');
# namefield = 'Name'
# for j in range(0, len(tifpath)):
#     rasterfile = tifpath[j]
#     out_folder = os.path.join(outdir, tardir[j])
#     for i in range(0,len(shpname)):
#         print(tardir[j],shpname[i])
#         shpdatafile = os.path.join(shpdit, shpname[i] + '.shp')
#         clipRaster_byEnvelope_shp_batchGrids(shpdatafile, rasterfile, out_folder, namefield)

# shpdatafile = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\selected_litLargeThan10_1degree_alignGAIA.shp'
# # rasterfile1 = r'D:\01data\00整理\02夜间灯光\npp\annualV2\VNL_2013.tif'
# # out_folder1 = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img\2013_VNL'
# # rasterfile2 = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\outdata\02oriGNL_stepSNL\03avg\2013.tif'
# # out_folder2 = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img\2013_DNL'
# namefield = 'Name'
# # clipRaster_byEnvelope_shp_batchGrids(shpdatafile, rasterfile1, out_folder1,namefield)
# # clipRaster_byEnvelope_shp_batchGrids(shpdatafile, rasterfile2, out_folder2,namefield)
# #裁剪年度辐射校正但未经饱和校正的DNL  oriDNL   裁剪landsat生成的NDVI 500m
# namefield = 'Name'
# shpdatafile = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\selected_litLargeThan10_1degree_alignGAIA.shp'
# rasterfile = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\outdata\02oriGNL_stepSNL\01stepwise\F182013.tif'
# out_folder = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img\2013_oriDNL'
# clipRaster_byEnvelope_shp_batchGrids(shpdatafile, rasterfile, out_folder,namefield)
# rasterfile = r'D:\01data\00整理\04NDVI\landsatNDVI\NDVI_2013.tif'
# out_folder = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img\2013_NDVI'
# clipRaster_byEnvelope_shp_batchGrids(shpdatafile, rasterfile, out_folder,namefield)
# # #裁剪Chen_2012 NTL 作为训练集的对比
# shpdatafile = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\selected_litLargeThan10_1degree_alignGAIA.shp'
# rasterfile3 = r'D:\01data\00整理\02夜间灯光\npp\Chen2012\LongNTL_2012.tif'
# out_folder3 = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img\2012_Chen'
# clipRaster_byEnvelope_shp_batchGrids(shpdatafile, rasterfile3, out_folder3,namefield)
# ## #裁剪2012 NTL 作为验证集
# # shpdatafile = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\valid_selected_litLargeThan50_1degree_alignGAIA.shp'
# # rasterfile1 = r'D:\01data\00整理\02夜间灯光\npp\annualV2\VNL_2012.tif'
# # out_folder1 = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img\2012_VNL'
# # rasterfile2 = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\outdata\02oriGNL_stepSNL\03avg\2012.tif'
# # out_folder2 = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img\2012_DNL'
# # namefield = 'Name'
# # clipRaster_byEnvelope_shp_batchGrids(shpdatafile, rasterfile1, out_folder1,namefield)
# # clipRaster_byEnvelope_shp_batchGrids(shpdatafile, rasterfile2, out_folder2,namefield)
# # #裁剪Chen_2012 NTL ,用于与本文方法生成的2012年类VNL数据对比
# # rasterfile3 = r'D:\01data\00整理\02夜间灯光\npp\Chen2012\LongNTL_2012.tif'
# # out_folder3 = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\out\2012_Chen'
# # clipRaster_byEnvelope_shp_batchGrids(shpdatafile, rasterfile3, out_folder3,namefield)
# # # #裁剪AVHRR_1km ,得到AVHRR数据
# rasterfile3 = r'D:\01data\00整理\04NDVI\AVHRR\2013_AVHRR_1KM.tif'
# out_folder3 = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img\2013_AVHRR'
# clipRaster_byEnvelope_shp_batchGrids(shpdatafile, rasterfile3, out_folder3,namefield)
# #裁剪格网间部分重叠的BJH
# namefield = 'Name'
# shpdatafile = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\out\01\BJH_1degree_overlap04.shp'
# rasterfile = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\outdata\02oriGNL_stepSNL\01stepwise\F182013.tif'
# out_folder = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\out\01\inputs\2013_oriDNL'
# if not os.path.exists(out_folder):
#     os.makedirs(out_folder)
# clipRaster_byEnvelope_shp_batchGrids(shpdatafile, rasterfile, out_folder,namefield)
# rasterfile = r'D:\01data\00整理\04NDVI\landsatNDVI\NDVI_2013.tif'
# out_folder = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\out\01\inputs\2013_NDVI'
# if not os.path.exists(out_folder):
#     os.makedirs(out_folder)
# clipRaster_byEnvelope_shp_batchGrids(shpdatafile, rasterfile, out_folder,namefield)
# rasterfile = r'D:\01data\00整理\02夜间灯光\npp\annualV2\VNL_2013.tif'
# out_folder = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\out\01\inputs\2013_VNL'
# if not os.path.exists(out_folder):
#     os.makedirs(out_folder)
# clipRaster_byEnvelope_shp_batchGrids(shpdatafile, rasterfile, out_folder,namefield)

# # 空间格局分析 数据
# for year in [2000,2010]:
#     shpdatafile = r'D:\04study\00Paper\Dissertation\03report\03论文内容\图表\实验一NTL\04空间格局分析\01不同夜间灯光数据空间格局比较\data\citySHP\city.shp'
#     rasterfile = r'D:\01data\00整理\02夜间灯光\grc_calib\\'+str(year)+'.tif'
#     out_folder = r'D:\04study\00Paper\Dissertation\03report\03论文内容\图表\实验一NTL\04空间格局分析\01不同夜间灯光数据空间格局比较\data\\'+str(year)
#     namefield = 'Name'
#     postfix = '_RNTL'
#     clipRaster_byEnvelope_shp_batchGrids(shpdatafile, rasterfile, out_folder,namefield,postfix=postfix)

# 时间序列空间格局
# shpdatafile =  r'D:\04study\00Paper\Dissertation\03report\03论文内容\图表\实验一NTL\04空间格局分析\01不同夜间灯光数据空间格局比较\data\citySHP\city.shp'
# # fid = 5
# # name = "Beijing"
# # fid = 4
# # name = "GBA"
# fid = 0
# name = "LosAngeles"
# out_root_dir = r'D:\04study\00Paper\Dissertation\03report\03论文内容\图表\实验一NTL\04空间格局分析\03时间序列空间格局'
# ##srDNL
# raster_dir = r'F:\study\dissertation\NTLtimwseries\world\result'
# for y in range(1992,2013):
#     print(y)
#     rasterfile = os.path.join(raster_dir,str(y),'UNet_labelNoProcess_12_13_1571','UNet_labelNoProcess_12_13_1571_world_'+str(y)+'.tif')
#     out_dir = os.path.join(out_root_dir,name,'data')
#     if(not os.path.exists(out_dir)):
#         os.makedirs(out_dir)
#     out_tif = os.path.join(out_dir,name+'_srDNL_'+str(y)+'.tif')
#     clipRaster_byEnvelope_shp_sigleGrid(shpdatafile, rasterfile, out_tif, fid)
# # ##VNL
# # raster_dir = r'D:\01data\00整理\02夜间灯光\npp\annualV2'
# # for y in range(2013,2021):
# #     print(y)
# #     rasterfile = os.path.join(raster_dir,'VNL_'+str(y)+'.tif')
# #     out_dir = os.path.join(out_root_dir,name,'data')
# #     if(not os.path.exists(out_dir)):
# #         os.makedirs(out_dir)
# #     out_tif = os.path.join(out_dir,name+'_VNL_'+str(y)+'.tif')
# #     clipRaster_byEnvelope_shp_sigleGrid(shpdatafile, rasterfile, out_tif, fid)
# # ##ENL
# # raster_dir = r'F:\study\data\00整理\02夜间灯光\npp-like'
# # for y in range(2000,2013):
# #     print(y)
# #     rasterfile = os.path.join(raster_dir,str(y),'LongNTL_'+str(y)+'.tif')
# #     out_dir = os.path.join(out_root_dir,name,'data')
# #     if(not os.path.exists(out_dir)):
# #         os.makedirs(out_dir)
# #     out_tif = os.path.join(out_dir,name+'_ENL_'+str(y)+'.tif')
# #     clipRaster_byEnvelope_shp_sigleGrid(shpdatafile, rasterfile, out_tif, fid)



'''根据随机样本点生成以该样本点为锚点的矩形框，裁剪相应的图像，得到样本图像集'''
def getPointsInfo_shp(samplePointsfile,namefield,sql=''):
    # 读取矢量
    # #############获取矢量点位的经纬度
    # 设置driver
    driver = ogr.GetDriverByName("ESRI Shapefile")
    # 打开矢量
    ds = driver.Open(samplePointsfile, 0)
    if ds is None:
        print('Could not open ' + 'sites.shp')
    # 获取图层
    layer = ds.GetLayer()
    #查询
    if sql != "":
        layer.SetAttributeFilter(sql)
    # 获取要素及要素地理位置
    locLst = []
    feature = layer.GetNextFeature()
    while feature:
        geometry = feature.GetGeometryRef()
        x = geometry.GetX()
        y = geometry.GetY()
        name = feature.GetField(namefield)
        locLst.append((x,y,name))
        feature = layer.GetNextFeature()
    return locLst
def createRectangleShp_bySamplePoints(samplePointsfile,rasterfile,namefield,outdir,shpname,xOffset,yOffset,width,height,sql=''):
    locLst = getPointsInfo_shp(samplePointsfile,namefield,sql)
    #栅格信息
    rasterdata = gdal.Open(rasterfile)
    geoTrans = rasterdata.GetGeoTransform()
    colnums = rasterdata.RasterXSize
    rownums = rasterdata.RasterYSize
    minCol,maxCol,minRow,maxRow = 0,colnums-1,0,rownums-1
    #注册驱动驱动，这里是ESRI Shapefile类型
    driver = ogr.GetDriverByName("ESRI Shapefile")
    ds = driver.Open(samplePointsfile, 0)
    if ds is None:
        print('Could not open ' + 'sites.shp')
    # 获取图层
    layer = ds.GetLayer()
    # 投影信息
    srs = layer.GetSpatialRef()

    # 创建数据源
    outfilepath = outdir + "\\" + shpname + ".shp"
    data_source = driver.CreateDataSource(outfilepath)
    # 创建图层，图层名称和上面注册驱动的shp名称一致
    layer = data_source.CreateLayer(shpname, srs, ogr.wkbPolygon)
    # 创建字段
    field_name = ogr.FieldDefn("Name", ogr.OFTString)
    field_name.SetWidth(50)
    layer.CreateField(field_name)
    for i in range(len(locLst)):
        print(locLst[i])
        x, y, name = locLst[i]
        #根据地理坐标得到图像行列号
        ci, ri = xy_to_rowcol(geoTrans,x , y)
        #得到左上角坐标
        cl = ci-xOffset
        rl = ri-yOffset
        #右下角(因为计算的是左上角的坐标，所以多加一格，得到外包矩形框的右下角坐标)
        cr,rr = cl+width,rl+height
        if (cl<minCol)|(rl<minRow)|(cl+width-1 > maxCol)|(rl+height-1> maxRow):
            continue
        #得到对应图像的geotrans
        xl, yl = rowcol_to_xy(geoTrans, rl, cl)
        xr,yr = rowcol_to_xy(geoTrans, rr, cr)
        x1 = '%.8f' % xl
        y1 = '%.8f' % yl
        x2 = '%.8f' % xr
        y2 = '%.8f' % yr
        # 左下角
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetField("Name", 'P'+str(name))
        # 创建几何
        wkt = 'POLYGON((' + x1 + ' ' + y1 + ',' + x2 + ' ' + y1 + ',' + x2 + ' ' + y2 + ',' + x1 + ' ' + y2 + ',' + x1 + ' ' + y1 + '))'
        poly = ogr.CreateGeometryFromWkt(wkt)
        feature.SetGeometry(poly)
        layer.CreateFeature(feature)
    feature = None
    data_source = None
# rasterfile = r'D:\01data\00整理\02夜间灯光\npp\annualV2\VNL_2012.tif'
# namefield = 'PID'
# outdir = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\outdata\02oriGNL_stepSNL\04interp\sample'
# # # 点在中间
# # samplePointsfile = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\outdata\02oriGNL_stepSNL\04interp\sample\point_onRectCenter.shp'
# # shpname = 'rect_center'
# # xOffset,yOffset,width,height = 50,50,100,100
# # createRectangleShp_bySamplePoints(samplePointsfile,rasterfile,namefield,outdir,shpname,xOffset,yOffset,width,height)
# # #点在左上角
# samplePointsfile = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\outdata\02oriGNL_stepSNL\04interp\sample\point_onRectRight.shp'
# shpname = 'rect_left1'
# xOffset,yOffset,width,height = -1,-1,100,100
# createRectangleShp_bySamplePoints(samplePointsfile,rasterfile,namefield,outdir,shpname,xOffset,yOffset,width,height)
# # #点在右下角point_onRectRight.shp
# samplePointsfile = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\outdata\02oriGNL_stepSNL\04interp\sample\point_onRectLeft.shp'
# shpname = 'rect_right1'
# xOffset,yOffset,width,height = 105,101,100,100
# createRectangleShp_bySamplePoints(samplePointsfile,rasterfile,namefield,outdir,shpname,xOffset,yOffset,width,height)

# samplePointsfile = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\03\samplePoints.shp'
# rasterfile = r'D:\01data\00整理\02夜间灯光\npp\annualV2\VNL_2013.tif'
# namefield = 'Name'
# outdir = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\03'
# shpname = 'samplePoints_rect'
# xOffset,yOffset,width,height = 128,128,256,256
# createRectangleShp_bySamplePoints(samplePointsfile,rasterfile,namefield,outdir,shpname,xOffset,yOffset,width,height)
def extractRaster_Rectangle_bySamplePoints(samplePointsfile,rasterfile,out_folder,namefield,xOffset,yOffset,width,height,sql=''):
    locLst = getPointsInfo_shp(samplePointsfile,namefield,sql)
    #从格网中裁剪对应点为中心的矩形部分
    #读取栅格
    rasterdata = gdal.Open(rasterfile)
    geoTrans = rasterdata.GetGeoTransform()
    geoProj = rasterdata.GetProjection()
    colnums = rasterdata.RasterXSize
    rownums = rasterdata.RasterYSize
    minCol,maxCol,minRow,maxRow = 0,colnums-1,0,rownums-1
    for i in range(len(locLst)):
        print(locLst[i])
        x,y,name = locLst[i]
        #根据地理坐标得到图像行列号
        ci, ri = xy_to_rowcol(geoTrans,x , y)
        #得到左上角坐标
        cl = ci-xOffset
        rl = ri-yOffset
        if (cl<minCol)|(rl<minRow)|(cl+width-1 > maxCol)|(rl+height-1> maxRow):
            continue
        #读取对应图像数据
        clip = rasterdata.ReadAsArray(cl, rl, width, height)  # ***只读要的那块***
        #得到对应图像的geotrans
        xl, yl = rowcol_to_xy(geoTrans, rl, cl)
        cl1,rl1 = xy_to_rowcol(geoTrans,xl, yl)
        newgeoTrans = list(geoTrans)
        newgeoTrans[0] = xl
        newgeoTrans[3] = yl
        # 输出
        save_path = os.path.join(out_folder,name+'.tif')
        rasterOp.write_img(save_path, geoProj, newgeoTrans, clip)
# samplePointsfile = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\03\samplePoints.shp'
# outdir = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img03'
# tifpath = []
# tardir = []
# tifpath.append(r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\outdata\cf_cvg\cf_cvg2012.tif');tardir.append('2012_CfCvg');
# # tifpath.append(r'D:\01data\00整理\02夜间灯光\grc_interp\2010_interp.tif');tardir.append('2010_RNTL');
# # tifpath.append(r'D:\01data\00整理\02夜间灯光\npp\annualV2\VNL_2013.tif');tardir.append('2013_VNL');
# # tifpath.append(r'D:\01data\00整理\04NDVI\landsatNDVI\NDVI_2013.tif');tardir.append('2013_NDVI');
# # tifpath.append(r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\outdata\02oriGNL_stepSNL\04interp\2013_interp.tif');tardir.append('2013_oriDNL');
# # tifpath.append(r'D:\01data\00整理\02夜间灯光\npp\annualV2\VNL_2012.tif');tardir.append('2012_VNL');
# # tifpath.append(r'D:\01data\00整理\02夜间灯光\npp\Chen2012\resample_align\chenNTL_2012.tif');tardir.append('2012_Chen');
# # tifpath.append(r'D:\01data\00整理\04NDVI\landsatNDVI\NDVI_2012.tif');tardir.append('2012_NDVI');
# # tifpath.append(r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\outdata\02oriGNL_stepSNL\04interp\2012_interp.tif');tardir.append('2012_oriDNL');
# namefield = 'Name'
# xOffset,yOffset,width,height = 128,128,256,256
# for j in range(0, len(tifpath)):
#     rasterfile = tifpath[j]
#     out_folder = os.path.join(outdir, tardir[j])
#     if not os.path.exists(out_folder):
#         os.makedirs(out_folder)
#     print(out_folder)
#     extractRaster_Rectangle_bySamplePoints(samplePointsfile,rasterfile,out_folder,namefield,xOffset,yOffset,width,height)

def extractRaster_byRefRaster(inpath,outpath,refpath):
    '''用一个栅格裁剪另一个栅格'''
    resampleType = gdalconst.GRA_NearestNeighbour
    inputrasfile1 = gdal.Open(refpath, gdal.GA_ReadOnly)
    inputProj1 = inputrasfile1.GetProjection()
    geotrans = inputrasfile1.GetGeoTransform()
    rows = inputrasfile1.RasterYSize
    cols = inputrasfile1.RasterXSize
    x2, y2 = rowcol_to_xy(geotrans, rows, cols)
    minX, minY, maxX, maxY = geotrans[0], y2, x2, geotrans[3]
    outputBounds = (minX, minY, maxX, maxY)
    options = gdal.WarpOptions(srcSRS=inputProj1, dstSRS=inputProj1, format='GTiff',
                               resampleAlg=resampleType, outputBounds=outputBounds)
    newds = gdal.Warp(outpath, inpath, options=options)
# inpath =  r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\outdata\cf_cvg\cf_cvg2013.tif'
# outdir = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img03\2013_CfCvg'
# refdir = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img03\2012_VNL'
# fileOp = FileOpr()
# filelst = fileOp.getFileNames_byExt(refdir,'.tif')
# for file in filelst:
#     if not os.path.exists(outdir):
#         os.makedirs(outdir)
#     refpath = os.path.join(refdir,file)
#     outpath = os.path.join(outdir,file)
#     if not os.path.exists(outpath):
#         print(file)
#         extractRaster_byRefRaster(inpath, outpath, refpath)





# #各大洲
# regionlst = []
# regionlst.append('Africa')
# regionlst.append('Oceania')
# regionlst.append('Asia')
# regionlst.append('SouthAmerica')
# regionlst.append('NorthAmerica')
# regionlst.append('Europe')
# outdir = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\world\inputs\img\regionImg'
# refdir = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\world\inputs\img\regionImg'
# year = 2013
# inpath = r'D:\01data\00整理\04NDVI\landsatNDVI\NDVI_'+str(year)+'.tif'
# for region in regionlst:
#     outpath = os.path.join(outdir,'NDVI_'+str(year)+region+'.tif')
#     refpath = os.path.join(refdir,'VNL_2013'+region+'.tif')
#     if not os.path.exists(outpath):
#         print(year,region)
#         extractRaster_byRefRaster(inpath, outpath, refpath)

'''World_Goode_Homolosine_Land  54052'''
def createShp_ByRasterfilesExtent(outshppath,featureNamelst,rasterpathlst):
    '''根据栅格影像范围创建shp'''
    # ####用已存在的空的shapefile##########
    # driver = ogr.GetDriverByName('ESRI Shapefile')  # 查找一个特定的驱动程序
    # data_source = driver.Open(outshppath, 1)  # 0只读，1可写
    # layer = data_source.GetLayer()
    #############用栅格数据指定投影####################
    rasterdata = gdal.Open(rasterpathlst[0])
    project = rasterdata.GetProjection()
    srs = osr.SpatialReference()
    srs.ImportFromProj4(project)
    rasterdata = None
    # 注册驱动驱动，这里是ESRI Shapefile类型
    driver = ogr.GetDriverByName("ESRI Shapefile")
    # 创建数据源
    data_source = driver.CreateDataSource(outshppath)
    # # 注入投影信息，这里使用54052，表示World_Goode_Homolosine_Land
    # srs = osr.SpatialReference()
    # srs.ImportFromEPSG(54052)  # EPSG没有World_Goode_Homolosine_Land
    # 创建图层，图层名称和上面注册驱动的shp名称一致
    shpname = os.path.basename(outshppath).split('.')[0]
    layer = data_source.CreateLayer(shpname, srs, ogr.wkbPolygon)
    # 创建字段
    field_name = ogr.FieldDefn("Name_ID", ogr.OFTInteger64)
    layer.CreateField(field_name)
    field_name = ogr.FieldDefn("Name", ogr.OFTString)
    field_name.SetWidth(15)
    layer.CreateField(field_name)
    for i in range(len(featureNamelst)):
        featureName = featureNamelst[i]
        rasterpath = rasterpathlst[i]
        inDs = gdal.Open(rasterpath, gdal.GA_ReadOnly)
        geotrans = inDs.GetGeoTransform()
        rows = inDs.RasterYSize
        cols = inDs.RasterXSize
        x2, y2 = rowcol_to_xy(geotrans, rows, cols)
        minX, minY, maxX, maxY = geotrans[0], y2, x2, geotrans[3]
        #创建要素
        x1 = format(minX, '.6f')
        y1 = format(minY, '.6f')
        x2 = format(maxX, '.6f')
        y2 = format(maxY, '.6f')
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetField("Name_ID", str(int(i+1)))
        # 左下角
        feature.SetField("Name", featureName)
        # 创建几何
        wkt = 'POLYGON((' + x1 + ' ' + y1 + ',' + x2 + ' ' + y1 + ',' + x2 + ' ' + y2 + ',' + x1 + ' ' + y2 + ',' + x1 + ' ' + y1 + '))'
        poly = ogr.CreateGeometryFromWkt(wkt)
        feature.SetGeometry(poly)
        layer.CreateFeature(feature)
        inDs = None
        geotrans = None
    feature = None
    layer = None
    data_source = None


'''矢量要素ID输出成一个txt'''
def getSingleAttributeToTxt(shpdatafile,txtfile,fieldname):
    driver = ogr.GetDriverByName("ESRI Shapefile")
    # 打开矢量
    ds = driver.Open(shpdatafile, 0)
    if ds is None:
        print('Could not open ' + 'sites.shp')
    attrValueLst = []
    # 获取图层
    layer = ds.GetLayer()
    #待记录字段
    feature = layer.GetNextFeature()
    while feature:
        attrValue = feature.GetField(fieldname)
        with open(txtfile, 'a') as f:
            f.write(attrValue.strip() + '\n')
        attrValueLst.append(attrValue)
        feature = layer.GetNextFeature()
    ds = None
    return attrValueLst
# rootdir = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\world\inputs\img\grid\litmasked_grids_2degree_05'
# namelst = ['Africa','Antarctica','Asia','Europe','NorthAmerica','Oceania','SouthAmerica']
# fieldname = 'Name'
# for name in namelst:
#     path = os.path.join(rootdir,name+'.shp')
#     txtfile = os.path.join(rootdir,'Ids_'+name+'.txt')
#     getSingleAttributeToTxt(path, txtfile, fieldname)

# path = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\world\inputs\img\grid\litmaskselected_grids_2degree_overlap05.shp'
# txtfile = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\world\inputs\img\grid\Id.txt'
# fieldname = 'Name'
# getSingleAttributeToTxt(path,txtfile,fieldname)

# dir = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\03'
# namelst = ['test_05','train_05','valid_05']
# for i in range(len(namelst)):
#     name = namelst[i]
#     print(name)
#     path = os.path.join(dir,name+'.shp')
#     fieldname = 'Name'
#     txtfile = os.path.join(dir,name+'_05.txt')
#     getSingleAttributeToTxt(path,txtfile,fieldname)

# dirnameLst = ['02','03','04','05']
# shpnameLst = ['NewYork','cairo','YRD','GBA']
# for i in range(len(dirnameLst)):
#     print(shpnameLst[i])
#     shpdatafile = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\out\\'+dirnameLst[i]+'\\'+shpnameLst[i]+'_1degree_overlap04.shp'
#     fieldname = 'Name'
#     txtfile = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\out\\'+dirnameLst[i]+'\\valid.txt'
#     getSingleAttributeToTxt(shpdatafile,txtfile,fieldname)

#world
# shpdatafile = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\world\imgIDs\litmaskselected_grids_1degree_overlap04.shp'
# fieldname = 'Name'
# txtfile = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\world\imgIDs\imgIds_whole.txt'
# getSingleAttributeToTxt(shpdatafile,txtfile,fieldname)
def getMultiAttrsToCsv(shpdatafile,csvpath,fieldlst):
    driver = ogr.GetDriverByName("ESRI Shapefile")
    # 打开矢量
    ds = driver.Open(shpdatafile, 0)
    if ds is None:
        print('Could not open ' + 'sites.shp')
    attrValueLstDict = {}
    for i in range(len(fieldlst)):
        attrValueLstDict[fieldlst[i]] = []
    # 获取图层
    layer = ds.GetLayer()
    #待记录字段
    feature = layer.GetNextFeature()
    while feature:
        for field in fieldlst:
            attrValueLstDict[field].append(feature.GetField(field))
        feature = layer.GetNextFeature()
    df = pd.DataFrame(data=attrValueLstDict)
    df.to_csv(csvpath,header=True,index=False)
    ds = None
    return attrValueLstDict




'''根据矢量要素提取栅格数据'''
'''点'''
def getPointXY(shp_path,fieldLst,sql):
    '''
    根据sql筛选相应数据，并读取shapefile的x,y坐标，以及fieldLst对应的字段值，
    :param shp_path: shapefile路径
    :param fieldLst: 字段名列表
    :param sql: 属性查询语句。如选择有标注的记录 sql= 'label1 = 1 or label1 = -1'
    :return: x列表，y列表，字段值列表
    '''
    # 读取矢量
    # #############获取矢量点位的经纬度
    # 设置driver
    driver = ogr.GetDriverByName("ESRI Shapefile")
    # 打开矢量
    ds = driver.Open(shp_path, 0)
    if ds is None:
        print('Could not open ' + 'sites.shp')
    # 获取图层
    layer = ds.GetLayer()
    #查询
    if sql != "":
        layer.SetAttributeFilter(sql)
    # 获取要素及要素地理位置
    xValues = []
    yValues = []
    fieldValues = []
    for i in range(0,len(fieldLst)):
        fieldValues.append([])
    #待记录字段
    feature = layer.GetNextFeature()
    while feature:
        geometry = feature.GetGeometryRef()
        x = geometry.GetX()
        y = geometry.GetY()
        for i in range(0,len(fieldLst)):
            v = feature.GetField(fieldLst[i])
            fieldValues[i].append(v)
        xValues.append(x)
        yValues.append(y)
        feature = layer.GetNextFeature()
    ds = None
    return xValues,yValues,fieldValues
def getRasterValueByPoints(shp_path,sql,ras_pathLst,rasNameLst,fieldLst):
    '''
    基于矢量点提取影像值，这里影像为单波段
    :param shp_path:矢量点图层shp路径
    :param sql:sql查询语句
    :param ras_pathLst:多个影像路径组成地列表
    :param rasNameLst:多个影像名列表，无后缀
    :param fieldLst: 需要提取地字段列表
    :return:result_df,需提取的字段值及栅格值转为dataframe范围，字段名为fieldLst+rasNameLst
    '''
    xValues, yValues, fieldValues = getPointXY(shp_path,fieldLst,sql)
    #############获取点位所在像元的栅格值
    multiValues = []
    for i in range(0,len(ras_pathLst)):
        ras_path = ras_pathLst[i]
        #读取栅格
        ras_inDs, ras_inBand, ras_inData, ras_vnoValue = rasterOp.getRasterData(ras_path)
        # 获取行列、波段
        rows = ras_inDs.RasterYSize
        cols = ras_inDs.RasterXSize
        # 获取放射变换信息
        ras_transform = ras_inDs.GetGeoTransform()
        ras_xOrigin = ras_transform[0]
        ras_yOrigin = ras_transform[3]
        ras_pixelWidth = ras_transform[1]
        ras_pixelHeight = ras_transform[5]
        values = []
        for i in range(len(xValues)):
            x = xValues[i]
            y = yValues[i]
            # 获取点位所在栅格的位置
            xOffset = int((x - ras_xOrigin) / ras_pixelWidth)
            yOffset = int((y - ras_yOrigin) / ras_pixelHeight)
            s = str(int(x)) + ' ' + str(int(y)) + ' ' + str(xOffset) + ' ' + str(yOffset) + ' '
            # 提取每个波段上对应的像元值
            data = ras_inBand.ReadAsArray(xOffset, yOffset, 1, 1)
            value = data[0, 0]
            # s = s + str(value) + ' '
            # print(s)
            values.append(value)
        ras_inDs = None
        multiValues.append(values)
    #转为pd.dataframe
    d = {}
    for i in range(0,len(fieldLst)):
        d[fieldLst[i]] = fieldValues[i]
    for i in range(0,len(rasNameLst)):
        d[rasNameLst[i]] = multiValues[i]
    result_df = pd.DataFrame(d)
    # #调整列顺序
    # colNames = []
    # for i in range(0, len(fieldLst)):
    #     colNames.append(fieldLst[i])
    # for i in range(0, len(rasNameLst)):
    #     colNames.append(rasNameLst[i])
    # result_df = result_df[colNames]
    return result_df

'''面'''
def getMaskTifByShp(shp_path,tifpath,outTifpath,sql="",cropToCutline=False):
    '''选择矢量文件中的部分要素，裁剪栅格，生成与输入栅格同等大小的mask.tif。
    生成结果中，像元值1为目标像元，像元值0为掩膜像元。'''
    #获取栅格信息
    inDs = gdal.Open(tifpath)
    rows = inDs.RasterYSize
    cols = inDs.RasterXSize
    geotrans = inDs.GetGeoTransform()
    proj = inDs.GetProjection()
    # #创建内存栅格
    # mem = gdal.GetDriverByName('MEM')
    # mid_ds = mem.Create('', cols, rows, 1, gdal.GDT_Byte)
    # mid_ds.GetRasterBand(1).WriteArray(np.ones((rows, cols), dtype=np.bool))
    # mid_ds.SetGeoTransform(geotrans)
    # mid_ds.SetProjection(proj)
    # #裁剪生成内存mask
    # mask_ds = gdal.Warp('', mid_ds, format='MEM', cutlineDSName=shp_path,cropToCutline=False,cutlineWhere=sql)
    # #输出
    # gtiff = gdal.GetDriverByName('GTiff')
    # result = gtiff.CreateCopy(outTifpath, mask_ds)
    # result.FlushCache()
    # del result,inDs,mid_ds,mask_ds
    #裁剪生成mask
    if sql == "":
        mask_ds = gdal.Warp(outTifpath, tifpath, format='GTiff', cutlineDSName=shp_path, cropToCutline=cropToCutline)
    else:
        mask_ds = gdal.Warp(outTifpath, tifpath, format='GTiff', cutlineDSName=shp_path,cropToCutline=cropToCutline,cutlineWhere=sql)
    #输出
    del inDs,mask_ds
def getMaskDataByShp(shp_path,tifpath,sql):
    '''选择矢量文件中的部分要素，裁剪栅格，在内存中生成与输入栅格同等大小的mask data。
    生成结果中，像元值1为目标像元，像元值0为掩膜像元。
    若原始图像较大，则速度较慢
    范围numpy'''
    #获取栅格信息
    inDs = gdal.Open(tifpath)
    rows = inDs.RasterYSize
    cols = inDs.RasterXSize
    geotrans = inDs.GetGeoTransform()
    proj = inDs.GetProjection()
    #创建内存栅格
    mem = gdal.GetDriverByName('MEM')
    mid_ds = mem.Create('', cols, rows, 1, gdal.GDT_Byte)
    mid_ds.GetRasterBand(1).WriteArray(np.ones((rows, cols), dtype=np.bool))
    mid_ds.SetGeoTransform(geotrans)
    mid_ds.SetProjection(proj)
    #裁剪生成内存mask
    mask_ds = gdal.Warp('', mid_ds, format='MEM', cutlineDSName=shp_path,cropToCutline=False,cutlineWhere=sql)
    maskdata = mask_ds.GetRasterBand(1).ReadAsArray(0, 0, cols, rows).astype(np.bool)
    del inDs,mid_ds,mask_ds
    return maskdata
def sumOfDN_RasterValue_maskByPolygon(shp_path,ras_pathLst,sql):
    '''统计矢量文件裁剪区域的DN值总和,mask.tif的大小与原始图像大小相同。
        若原始图像很大，则速度会较慢。
    '''
    maskdata = getMaskDataByShp(shp_path,ras_pathLst[0],sql)
    stat_result = []
    for i in range(len(ras_pathLst)):
        _, _, inData, noValue = rasterOp.getRasterData(ras_pathLst[i])
        inData[(inData == noValue)|(inData<0)]=0
        resultdata = inData[maskdata]
        sumOfDN = np.nansum(resultdata)
        stat_result.append(sumOfDN)
    return stat_result
def getClipDataByShp_cropToCutline_True(shp_path,tifpath,sql,xRes=500,yRes=500):
    #裁剪出与矢量图形框相同大小的栅格
    mask_ds = gdal.Warp('', tifpath, format='MEM', cutlineDSName=shp_path, cropToCutline=True, cutlineWhere=sql,xRes=xRes,yRes=yRes)
    rows = mask_ds.RasterYSize
    cols = mask_ds.RasterXSize
    maskdata = mask_ds.GetRasterBand(1).ReadAsArray(0, 0, cols, rows).astype(np.float)
    del mask_ds
    return maskdata
def sumOfDN_RasterValue_maskByPolygon_cropToCutline_True(shp_path,ras_pathLst,sql):
    '''统计矢量文件裁剪区域的DN值总和,mask.tif的大小与矢量图形框相同。这样比与原始图像相同的速度快。
    '''
    stat_result = []
    for i in range(len(ras_pathLst)):
        maskdata = getClipDataByShp_cropToCutline_True(shp_path, ras_pathLst[i], sql)
        maskdata[(maskdata<0)]=0
        sumOfDN = np.nansum(maskdata)
        stat_result.append(sumOfDN)
    return stat_result

# shp_path = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\out\01\data\part1_poly.shp'
# dir = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\out\01\data'
# outdir =  r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\out\01\data\part1'
# namelst = ['BJH_VNL2012.tif','BJH_rnl2010.tif','BJH_cfcvg2012.tif']
# sql = ""
# for name in namelst:
#     tifpath = os.path.join(dir,name)
#     outTifpath = os.path.join(outdir,name)
#     getMaskTifByShp(shp_path,tifpath,outTifpath,sql,cropToCutline=True)



'''栅格对其'''
def getRasterData_bySize(data,geoTrans,ul_x,ul_y,xRastersize,yRastersize):
    ulcol, ulrow = xy_to_rowcol(geoTrans, ul_x, ul_y)
    drcol = ulcol+xRastersize-1
    drrow = ulrow+yRastersize-1
    #原始数据大小
    rownums,colnums = data.shape
    rasMinX,rasMaxX,rasMinY,rasMaxY = 0,colnums-1,0,rownums-1
    #获取截取范围
    if (ulcol>rasMaxX) | (ulrow>rasMaxY) |(drcol<rasMinX)|(drrow<rasMinY):
        return None,geoTrans
    if ulcol < rasMinX:
        ulcol = rasMinX
    if ulrow < rasMinY:
        ulrow = rasMinY
    if drcol > rasMaxX:
        drcol = rasMaxX
    if drrow > rasMaxY:
        drrow = rasMaxY
    # Calculate the pixel size of the new image
    result = data[ulrow:drrow+1,ulcol:drcol+1]
    new_minX,new_maxY = rowcol_to_xy(geoTrans,ulrow,ulcol)
    newgeoTrans = list(geoTrans)
    newgeoTrans[0] = new_minX
    newgeoTrans[3] = new_maxY
    return result,newgeoTrans

def alignRaster(indata,inGeoTrans,refRasterPath):
    refDs = gdal.Open(refRasterPath)
    refGeoTrans = refDs.GetGeoTransform()
    refXrastersize = refDs.RasterXSize
    refYrastersize = refDs.RasterYSize
    result = np.zeros((refYrastersize,refXrastersize))
    #用ref的范围截取indata数据
    clipdata,newgeoTrans = getRasterData_bySize(indata,inGeoTrans,refGeoTrans[0],refGeoTrans[3],refXrastersize,refYrastersize)
    if clipdata is None:
        return result,refGeoTrans
    #截取的数据在参照影像中的范围
    clip_rownum,clip_colnum = clipdata.shape
    ulcol, ulrow = xy_to_rowcol(refGeoTrans, newgeoTrans[0], newgeoTrans[3])

    #获取截取范围
    rasMinX, rasMaxX, rasMinY, rasMaxY = 0, refXrastersize - 1, 0, refYrastersize - 1
    if ulcol < rasMinX:
        ulcol = rasMinX
    if ulrow < rasMinY:
        ulrow = rasMinY
    result[ulrow:clip_rownum,ulcol:clip_colnum] = clipdata
    return result,refGeoTrans


'''统计栅格最大最小值'''
def maxStatis_Raster(rasterfile,readsize):
    rasterdata = gdal.Open(rasterfile)
    band =  rasterdata.GetRasterBand(1)
    rows = rasterdata.RasterYSize
    cols = rasterdata.RasterXSize
    num_rows =  int(rows//readsize)
    num_cols = int(cols/readsize)
    rest_rows = rows - readsize*num_rows
    rest_cols = cols - readsize*num_cols
    maxValue = 0
    for i in range(0,num_rows):
        for j in range(0,num_cols):
            data = band.ReadAsArray(j*readsize,i*readsize,readsize,readsize).astype(np.float)
            value = data.max()
            if value > maxValue:
                maxValue = value
    for i in range(0,num_rows):
        data = band.ReadAsArray(readsize*num_cols, i * readsize, rest_cols, readsize).astype(np.float)
        value = data.max()
        if value > maxValue:
            maxValue = value
    for j in range(0,num_cols):
        data = band.ReadAsArray(j*readsize,readsize*num_rows,readsize,rest_rows).astype(np.float)
        value = data.max()
        if value > maxValue:
            maxValue = value
    data = band.ReadAsArray(readsize*num_cols, readsize * num_rows, rest_cols, rest_rows).astype(np.float)
    value = data.max()
    if value > maxValue:
        maxValue = value
    return maxValue
# rasterfile = r'D:\01data\00整理\02夜间灯光\npp\annualV2\VNL_2013.tif'
# readsize = 10000
# maxvalue = maxStatis_Raster(rasterfile,readsize)
# print(maxvalue)

def histStatis_Raster(rasterfile,readsize,histlst):
    rasterdata = gdal.Open(rasterfile)
    band =  rasterdata.GetRasterBand(1)
    rows = rasterdata.RasterYSize
    cols = rasterdata.RasterXSize
    num_rows =  int(rows//readsize)
    num_cols = int(cols/readsize)
    rest_rows = rows - readsize*num_rows
    rest_cols = cols - readsize*num_cols
    resultLst = [0 for i in histlst]
    for i in range(0,num_rows):
        for j in range(0,num_cols):
            data = band.ReadAsArray(j*readsize,i*readsize,readsize,readsize).astype(np.float)
            data = data.reshape(-1)
            for da in data:
                pos = bisect.bisect(histlst,da)
                resultLst[pos-1] += 1
    for i in range(0,num_rows):
        data = band.ReadAsArray(readsize*num_cols, i * readsize, rest_cols, readsize).astype(np.float)
        data = data.reshape(-1)
        for da in data:
            pos = bisect.bisect(histlst, da)
            resultLst[pos - 1] += 1
    for j in range(0,num_cols):
        data = band.ReadAsArray(j*readsize,readsize*num_rows,readsize,rest_rows).astype(np.float)
        data = data.reshape(-1)
        for da in data:
            pos = bisect.bisect(histlst, da)
            resultLst[pos - 1] += 1
    data = band.ReadAsArray(readsize*num_cols, readsize * num_rows, rest_cols, rest_rows).astype(np.float)
    data = data.reshape(-1)
    for da in data:
        pos = bisect.bisect(histlst, da)
        resultLst[pos - 1] += 1
    return resultLst
# histlst = [0,1,10,100,500,1000,2000,3000,4000,5000,10000,20000]
# rasterfile = r'D:\01data\00整理\02夜间灯光\npp\annualV2\VNL_2013.tif'
# readsize = 10000
# bincount = histStatis_Raster(rasterfile,readsize,histlst)
# print(bincount)


"""
重投影时，重采样

重投影，投影过程中可以设置以下信息
gdal.Warp(
    xRes,yRes:两个方向上的分辨率；
    srcNodata：原来数据的Nodata值
    dstNodata：输出数据的Nodata值
    dstSRS：输出的投影坐标系，可以读取影像的：
            Raster = gdal.Open(InputImage, gdal.GA_ReadOnly)
            Projection = Raster.GetProjectionRef()
    resampleAlg:重采样方式,算法包括：
            import gdalconst
            gdalconst.GRA_NearestNeighbour：near
            gdalconst.GRA_Bilinear:bilinear
            gdalconst.GRA_Cubic:cubic
            gdalconst.GRA_CubicSpline:cubicspline
            gdalconst.GRA_Lanczos:lanczos
            gdalconst.GRA_Average:average
            gdalconst.GRA_Mode:mode
    )

"""

'''分块统计'''
'''1 GAIA 转为 VNL 分辨率'''
def searchFiles_byExtension(dirpath):
    result_filepname_list = []
    for root, dirs, files in os.walk(dirpath):  # 遍历该文件夹
        for file in files:  # 遍历刚获得的文件名files
            (filename, extension) = os.path.splitext(file)  # 将文件名拆分为文件名与后缀
            if (extension == '.tif'):  # 判断该后缀是否为.c文件
                result_filepname_list.append(filename)
    return result_filepname_list
def GAIA_filename(filename):
    pr = 'GAIA_1985_2018_'
    file_numbers = list(map(int,filename.split('_')))
    if file_numbers[0] >=0:
        pr += str(file_numbers[0]).zfill(2)
    else:
        pr += str(file_numbers[0]).zfill(3)
    pr += '_'
    if file_numbers[1] >=0:
        pr += str(file_numbers[1]).zfill(3)
    else:
        pr += str(file_numbers[1]).zfill(4)
    return pr
def calIMPSpercent_GAIA_to_VNL():
    gaia_folder_path = r'D:\01data\00整理\14城市边界\GAIA'
    vnl_folder_path = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img\2013_VNL'
    out_block_folder_path = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img\2013_GAIA_B'
    out_GAIA_folder_path = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img\2013_GAIA'
    if not os.path.exists(out_block_folder_path):
        os.mkdir(out_block_folder_path)
    if not os.path.exists(out_GAIA_folder_path):
        os.mkdir(out_GAIA_folder_path)
    opGDAL = vectorOp.OP_gdal_raster()
    filenamelst = searchFiles_byExtension(vnl_folder_path)
    for filename in filenamelst:
        gaia_filename = GAIA_filename(filename)
        gaia_path = os.path.join(gaia_folder_path,gaia_filename+'.tif')
        nvl_path = os.path.join(vnl_folder_path,filename+'.tif')
        out_block_path = os.path.join(out_block_folder_path,filename+'.tif')
        out_imps_path = os.path.join(out_GAIA_folder_path,filename+'.tif')
        if not os.path.exists(gaia_path):
            print(filename)
            continue
        if os.path.exists(out_imps_path):
            continue
        inDs, inBand, inData, noValue = rasterOp.getRasterData(gaia_path)
        rows,cols = inData.shape
        #get 2013 IMPS
        inData[inData<=5] = 0
        inData[inData>5] = 1
        #block statistic 16&16
        height = 16
        width = 16
        row_num = rows//height
        col_num = cols // width
        rows_rest =  rows - height*row_num
        cols_rest = cols - width*col_num
        result = np.zeros((rows,cols))
        for i in range(row_num):
            for j in range(col_num):
                start_rIndex = i*height
                end_rIndex = (i+1)*height
                start_cIndex = j*width
                end_cIndex = (j+1)*width
                data = inData[start_rIndex:end_rIndex,start_cIndex:end_cIndex]
                sumValue = data.sum()
                result[start_rIndex:end_rIndex, start_cIndex:end_cIndex] = sumValue / (height * width)
        if (cols_rest >0) | (rows_rest >0):
            #the rest rows
            for j in range(col_num):
                start_rIndex = rows-height
                end_rIndex = rows
                start_cIndex = j * width
                end_cIndex = (j + 1) * width
                data = inData[start_rIndex:end_rIndex, start_cIndex:end_cIndex]
                sumValue = data.sum()
                result[rows-rows_rest:rows, start_cIndex:end_cIndex] = sumValue / (height * width)
            #the rest cols
            for i in range(row_num):
                start_rIndex = i * height
                end_rIndex = (i + 1) * height
                start_cIndex = cols-width
                end_cIndex = cols
                data = inData[start_rIndex:end_rIndex, start_cIndex:end_cIndex]
                sumValue = data.sum()
                result[start_rIndex:end_rIndex, cols-cols_rest:cols] = sumValue / (height * width)
            #the rest grid
            data = inData[rows-height:rows, cols-width:cols]
            sumValue = data.sum()
            result[rows-rows_rest:rows, cols - cols_rest:cols] = sumValue / (height * width)
        #resample
        rasterOp.outputResult(inDs, result, out_block_path)
        opGDAL.ReprojectImages(out_block_path, nvl_path, out_imps_path)

# calIMPSpercent_GAIA_to_VNL()


# import datetime
# t1 = datetime.datetime.now()
# a = np.where((np.isnan(inData)| (inData<0)),0,inData)
# sum_a = np.sum(a)
# t2 = datetime.datetime.now()
# print(sum_a,t2-t1)
#
# import torch
# t1 = datetime.datetime.now()
# b = torch.where(torch.isnan(inData_torch)| (inData_torch < 0),torch.full_like(inData_torch, 0) ,inData_torch )
# sum_torch = torch.sum(inData_torch)
# t2 = datetime.datetime.now()
# print(sum_torch.data,t2-t1)



# dir = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\out\data'
# year = 2010
# region = 'BJH'
# ndvipath = os.path.join(dir,'NDVI',region+'_'+str(year)+'_NDVI.tif')
# dnlpath = os.path.join(dir,'DNL',region+'_'+str(year)+'_DNL.tif')
# vanuipath = os.path.join(dir,'VANUI',region+'_'+str(year)+'_VANUI.tif')
# _,_,ndvi_inData,_ = rasterOp.getRasterData(ndvipath)
# dnl_inDs,_,dnl_inData,_ = rasterOp.getRasterData(dnlpath)
# ndvi_inData = np.where(np.isnan(ndvi_inData), 0, ndvi_inData)
# ndvi_inData[(ndvi_inData <= 0)] = 0
# dnl_inData = np.where(np.isnan(dnl_inData), 0, dnl_inData)
# dnl_inData[(dnl_inData <= 0)] = 0
# vanuiData = (1-ndvi_inData)*dnl_inData
# rasterOp.outputResult(dnl_inDs,vanuiData,vanuipath)


'''栅格转矢量'''
