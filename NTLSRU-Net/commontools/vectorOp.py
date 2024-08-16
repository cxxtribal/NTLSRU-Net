import numpy as np
import  pandas as pd
from osgeo import gdal
from osgeo import ogr,osr,gdalconst
import os
import os
import os
# 对应自己的python包的安装地址
# os.environ['PROJ_LIB'] = r'C:\ProgramData\Anaconda3\Lib\site-packages\pyproj\proj_dir\share\proj'
os.environ['PROJ_LIB'] = r'C:\ProgramData\Anaconda3\Lib\site-packages\osgeo\data\proj'

import rasterOp

class OP_gdal_raster:
    def RasterMosaic(self,inputfilePathLst,referencefilefilePath,outputfilePath,resampleType,nodata):
        '''
        栅格镶嵌
        :param inputfilePathLst:
        :param referencefilefilePath:
        :param outputfilePath:
        :param resampleType:
        GRA_NearestNeighbour=0      最近邻法，算法简单并能保持原光谱信息不变；缺点是几何精度差，灰度不连续，边缘会出现锯齿状
        GRA_Bilinear=1              双线性法，计算简单，图像灰度具有连续性且采样精度比较精确；缺点是会丧失细节；默认
        GRA_Cubic=2                 三次卷积法，计算量大，图像灰度具有连续性且采样精度高；
        GRA_CubicSpline=3           三次样条法，灰度连续性和采样精度最佳；
        GRA_Lanczos=4               分块兰索斯法，由匈牙利数学家、物理学家兰索斯法创立，实验发现效果和双线性接近；
        :return:
        '''
        inputrasfile1 = gdal.Open(referencefilefilePath, gdal.GA_ReadOnly)
        inputProj1 = inputrasfile1.GetProjection()
        inputRasteLst = []
        srcnovalue = None
        for inpath in inputfilePathLst:
            inRaster = gdal.Open(inpath, gdal.GA_ReadOnly)
            inputRasteLst.append(inRaster)
            if not srcnovalue:
                inBand = inRaster.GetRasterBand(1)
                srcnovalue = inBand.GetNoDataValue()
        options = gdal.WarpOptions(srcSRS=inputProj1, dstSRS=inputProj1, format='GTiff',
                                   resampleAlg = resampleType,srcNodata = srcnovalue,dstNodata=nodata)
        gdal.Warp(outputfilePath, inputRasteLst, options=options)
    def RasterMosaic_tast(self):
        in_folder = r"D:\01data\00整理\04NDVI\MOD13A1\2013"
        referencefilefilePath = in_folder+'\\115_35.tif'
        inIDs = ["115_35", "115_40"]
        inputfilePathLst = []
        for id in inIDs:
            inputfilePathLst.append(in_folder+"\\"+id+'.tif')
        outputfilePath = r'D:\01data\00整理\04NDVI\temp\01merge.tif'
        resampleType = gdalconst.GRA_Bilinear
        self.RasterMosaic(inputfilePathLst,referencefilefilePath,outputfilePath,resampleType,-9999)

    def RasterReclassify(self,remap,rasterpath,outpath,nodata):
        '''重分类'''
        # 加载影像
        # 为了支持中文路径，请添加下面这句代码
        gdal.SetConfigOption("GDAL_FILENAME_IS_UTF8", "NO")
        # 为了使属性表字段支持中文，请添加下面这句
        gdal.SetConfigOption("SHAPE_ENCODING", "")
        # 注册所有的驱动
        ogr.RegisterAll()
        #打开栅格
        ds = gdal.Open(rasterpath)
        # 判断数据读取是否成功
        if ds is None:
            print('打开数据' + rasterpath + '失败！')
        #获取影像信息，用于创建新栅格
        # 读取栅格数据集的x方向像素数，y方向像素数
        cols = ds.RasterXSize
        rows = ds.RasterYSize
        # bands = ds.RasterCount
        band = ds.GetRasterBand(1)
        data = band.ReadAsArray(0, 0, cols, rows)
        # 进行影像重分类并创建输出影像
        img = data.copy()
        remapLst = remap.split(';')
        for re in remapLst:
            reValueLst = re.split(' ')
            if len(reValueLst) == 2:
                value1 = float(reValueLst[0])
                if reValueLst[-1] == 'NODATA':
                    img[img==value1] = nodata
                else:
                    img[img==value1] = int(reValueLst[-1])
            else:
                value1 = float(reValueLst[0])
                value2 = float(reValueLst[1])
                if reValueLst[-1] == 'NODATA':
                    img[(img>value1)&(img<=value2)] = nodata
                else:
                    img[(img>value1)&(img<=value2)] = int(reValueLst[-1])
        # 创建并输出图像
        rasterOp.outputResult(ds, img, outpath)
    def RasterReclassify_tast(self):
        remap = '0 0;0 34 1'
        rasterpath = r'D:\04study\00Paper\Dissertation\01experiment\00data\04validation\02samplepoints\01beijing\samples\01GAIA.tif'
        outpath = r'D:\04study\00Paper\Dissertation\01experiment\00data\04validation\02samplepoints\01beijing\samples\01GAIA_2018.tif'
        nodata = -99
        self.RasterReclassify(remap,rasterpath,outpath,nodata)

    def RaseterAggregate(self,rasterpath,cellFactor,outpath):
        inDs, inBand, inData, vnoValue = rasterOp.getRasterData(rasterpath)
        geotrans = list(inDs.GetGeoTransform())
        geotrans[1] *= cellFactor  # 像元宽度变为原来的两倍
        geotrans[5] *= cellFactor  # 像元高度也变为原来的两倍
        xsize = inBand.XSize
        ysize = inBand.YSize
        x_resolution = int(xsize / cellFactor)  # 影像的行列都变为原来的一半
        y_resolution = int(ysize / cellFactor)
        out_ds = inDs.GetDriver().Create(outpath, x_resolution, y_resolution, 1,
                                          inBand.DataType)  # 创建一个构建重采样影像的句柄
        out_ds.SetProjection(inDs.GetProjection())  # 设置投影信息
        out_ds.SetGeoTransform(geotrans)  # 设置地理变换信息
        data = np.empty((y_resolution, x_resolution), np.int)  # 设置一个与重采样影像行列号相等的矩阵去接受读取所得的像元值
        inBand.ReadAsArray(buf_obj=data)
        out_band = out_ds.GetRasterBand(1)
        out_band.WriteArray(data)
        out_band.FlushCache()
        out_band.ComputeStatistics(False)  # 计算统计信息
        out_ds.BuildOverviews('average', [1, 2, 4, 8, 16, 32])  # 构建金字塔
        del inDs  # 删除句柄
        del out_ds

    def ReprojectImages(self,inputfilePath,referencefilefilePath,outputfilePath):
        '''
        重采样
        :param inputfilePath:
        :param referencefilefilePath:
        :param outputfilePath:
        :return:
        '''
        # 获取输出影像信息
        inputrasfile = gdal.Open(inputfilePath, gdal.GA_ReadOnly)
        inputProj = inputrasfile.GetProjection()
        bandinputfile = inputrasfile.GetRasterBand(1)
        # 获取参考影像信息
        referencefile = gdal.Open(referencefilefilePath, gdal.GA_ReadOnly)
        referencefileProj = referencefile.GetProjection()
        referencefileTrans = referencefile.GetGeoTransform()
        bandreferencefile = referencefile.GetRasterBand(1)
        Width = referencefile.RasterXSize
        Height = referencefile.RasterYSize
        nbands = referencefile.RasterCount
        # 创建重采样输出文件（设置投影及六参数）
        driver = gdal.GetDriverByName('GTiff')
        output = driver.Create(outputfilePath, Width, Height, nbands, bandinputfile.DataType)
        output.SetGeoTransform(referencefileTrans)
        output.SetProjection(referencefileProj)
        # 参数说明 输入数据集、输出文件、输入投影、参考投影、重采样方法(最邻近内插\双线性内插\三次卷积等)、回调函数
        gdal.ReprojectImage(inputrasfile, output, inputProj, referencefileProj, gdalconst.GRA_NearestNeighbour, 0.0, 0.0)
        return output

#create Fishnet shp
def example_createFishnet():
    '''创建经纬格网例子'''
    shp_path = r'D:\04study\00Paper\Dissertation\01experiment\00data\girds\grids_5degree.shp'
    #注册驱动驱动，这里是ESRI Shapefile类型
    driver = ogr.GetDriverByName("ESRI Shapefile")
    #创建数据源
    data_source = driver.CreateDataSource(shp_path)
    #注入投影信息，这里使用4326，表示WGS84经纬坐标
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326) #这是WGS84
    #创建图层，图层名称和上面注册驱动的shp名称一致
    layer = data_source.CreateLayer("grids_5degree",srs,ogr.wkbPolygon)
    #创建字段
    field_name = ogr.FieldDefn("Name_ID",ogr.OFTInteger64)
    layer.CreateField(field_name)
    field_name = ogr.FieldDefn("Name", ogr.OFTString)
    field_name.SetWidth(15)
    layer.CreateField(field_name)
    #插入要素
    feature = ogr.Feature(layer.GetLayerDefn())
    feature.SetField("Name_ID","1")
    feature.SetField("Name","100_50")
    #创建几何
    wkt = 'POLYGON((100 50,105 50,105 45,100 45,100 50))'
    poly = ogr.CreateGeometryFromWkt(wkt)
    feature.SetGeometry(poly)
    layer.CreateFeature(feature)
    #插入要素
    feature = ogr.Feature(layer.GetLayerDefn())
    feature.SetField("Name_ID","2")
    feature.SetField("Name","105_50")
    #创建几何
    wkt = 'POLYGON((105 50,110 50,110 45,105 45,105 50))'
    poly = ogr.CreateGeometryFromWkt(wkt)
    feature.SetGeometry(poly)
    layer.CreateFeature(feature)
    #插入要素
    feature = ogr.Feature(layer.GetLayerDefn())
    feature.SetField("Name_ID","3")
    feature.SetField("Name","100_45")
    #创建几何
    wkt = 'POLYGON((100 45,105 45,105 40,100 40,100 45))'
    poly = ogr.CreateGeometryFromWkt(wkt)
    feature.SetGeometry(poly)
    layer.CreateFeature(feature)
    #插入要素
    feature = ogr.Feature(layer.GetLayerDefn())
    feature.SetField("Name_ID","4")
    feature.SetField("Name","105_45")
    #创建几何
    wkt = 'POLYGON((105 45,110 45,110 40,105 40,105 45))'
    poly = ogr.CreateGeometryFromWkt(wkt)
    feature.SetGeometry(poly)
    layer.CreateFeature(feature)
    #清空缓存
    feature = None
    data_source = None

def createReg(outdir,shpname,minLng=-180,maxLng=180,minLat=-65,maxLat=75):
    # shp_path = r'D:\04study\00Paper\Dissertation\01experiment\00data\01extent\NTLext\NTL_Ext.shp'
    shp_path = outdir + "\\"+shpname+".shp"
    #注册驱动驱动，这里是ESRI Shapefile类型
    driver = ogr.GetDriverByName("ESRI Shapefile")
    #创建数据源
    data_source = driver.CreateDataSource(shp_path)
    #注入投影信息，这里使用4326，表示WGS84经纬坐标
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326) #这是WGS84
    #创建图层，图层名称和上面注册驱动的shp名称一致
    layer = data_source.CreateLayer(shpname,srs,ogr.wkbPolygon)
    #创建字段
    field_name = ogr.FieldDefn("Name_ID",ogr.OFTInteger64)
    layer.CreateField(field_name)
    field_name = ogr.FieldDefn("Name", ogr.OFTString)
    field_name.SetWidth(15)
    layer.CreateField(field_name)
    #插入要素
    id = 1
    x1 = str(int(minLng))
    y1 = str(int(minLat))
    x2 = str(int(maxLng))
    y2 = str(int(maxLat))
    feature = ogr.Feature(layer.GetLayerDefn())
    feature.SetField("Name_ID", str(int(id)))
    # 左下角
    name = y2 + "_" + x1
    feature.SetField("Name", name)
    # 创建几何
    wkt = 'POLYGON((' + x1 + ' ' + y1 + ',' + x2 + ' ' + y1 + ',' + x2 + ' ' + y2 + ',' + x1 + ' ' + y2 + ',' + x1 + ' ' + y1 + '))'
    poly = ogr.CreateGeometryFromWkt(wkt)
    feature.SetGeometry(poly)
    layer.CreateFeature(feature)
    feature = None
    data_source = None

def createFishnet_5degree_WGS84(outdir,shpname,minLng=-180,maxLng=180,minLat=-65,maxLat=75,step=5):
    '''
    创建经纬格网shapefile,夜间灯光覆盖范围。
    :param outdir: 输出目录
    :param shpname: shp名称，不包括后缀名
    :param minLng: 覆盖范围的最小经度
    :param maxLng:  覆盖范围的最大经度
    :param minLat:  覆盖范围的最小纬度
    :param maxLat:  覆盖范围的最大纬度
    :param step: 格网大小
    :return: shapefile
    '''
    # shp_path = r'D:\04study\00Paper\Dissertation\01experiment\00data\02girds\grids_5degree.shp'
    shp_path = outdir + "\\"+shpname+".shp"
    #注册驱动驱动，这里是ESRI Shapefile类型
    driver = ogr.GetDriverByName("ESRI Shapefile")
    #创建数据源
    data_source = driver.CreateDataSource(shp_path)
    #注入投影信息，这里使用4326，表示WGS84经纬坐标
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326) #这是WGS84
    #创建图层，图层名称和上面注册驱动的shp名称一致
    layer = data_source.CreateLayer(shpname,srs,ogr.wkbPolygon)
    #创建字段
    field_name = ogr.FieldDefn("Name_ID",ogr.OFTInteger64)
    layer.CreateField(field_name)
    field_name = ogr.FieldDefn("Name", ogr.OFTString)
    field_name.SetWidth(15)
    layer.CreateField(field_name)
    #插入要素
    xlst = np.arange(minLng,maxLng+step,step)
    ylst = np.arange(minLat,maxLat+step,step)
    xlen = len(xlst)
    ylen = len(ylst)
    id = 0
    for i in range(0,xlen-1):
        for j in range(0,ylen-1):
            id = id +1
            x1 = str(int(xlst[i]))
            y1 = str(int(ylst[j]))
            x2 = str(int(xlst[i+1]))
            y2 = str(int(ylst[j+1]))
            feature = ogr.Feature(layer.GetLayerDefn())
            feature.SetField("Name_ID",str(int(id)))
            #左下角
            name = y2+"_"+x1
            feature.SetField("Name",name)
            #创建几何
            wkt = 'POLYGON(('+x1+' '+y1+','+x2+' '+y1+','+x2+' '+y2+','+x1+' '+y2+','+x1+' '+y1+'))'
            poly = ogr.CreateGeometryFromWkt(wkt)
            feature.SetGeometry(poly)
            layer.CreateFeature(feature)
    feature = None
    data_source = None
# outdir = r'D:\04study\00Paper\Dissertation\01experiment\00data\02girds'
# shpname = 'grids_2degree'
# createFishnet_5degree_WGS84(outdir,shpname,minLng=-180,maxLng=180,minLat=-65,maxLat=75,step=2)
# print('2 degree finish!')
# shpname = 'grids_4degree'
# createFishnet_5degree_WGS84(outdir,shpname,minLng=-180,maxLng=180,minLat=-65,maxLat=75,step=4)
# print('4 degree finish!')
# outdir = r'D:\04study\00Paper\Dissertation\01experiment\00data\02girds'
# shpname = 'grids_1degree'
# createFishnet_5degree_WGS84(outdir,shpname,minLng=-180,maxLng=180,minLat=-65,maxLat=75,step=1)

def createFishnet_5degree_WGS84_overlap(outdir,shpname,minLng=-180,maxLng=180,minLat=-65,maxLat=75,step=5,overlapnum=0.4):
    '''
    创建经纬格网shapefile,夜间灯光覆盖范围。
    :param outdir: 输出目录
    :param shpname: shp名称，不包括后缀名
    :param minLng: 覆盖范围的最小经度
    :param maxLng:  覆盖范围的最大经度
    :param minLat:  覆盖范围的最小纬度
    :param maxLat:  覆盖范围的最大纬度
    :param step: 格网大小
    :param overlapnum: 重叠大小
    :return: shapefile
    '''
    # shp_path = r'D:\04study\00Paper\Dissertation\01experiment\00data\02girds\grids_5degree.shp'
    shp_path = outdir + "\\"+shpname+".shp"
    #注册驱动驱动，这里是ESRI Shapefile类型
    driver = ogr.GetDriverByName("ESRI Shapefile")
    #创建数据源
    data_source = driver.CreateDataSource(shp_path)
    #注入投影信息，这里使用4326，表示WGS84经纬坐标
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326) #这是WGS84
    #创建图层，图层名称和上面注册驱动的shp名称一致
    layer = data_source.CreateLayer(shpname,srs,ogr.wkbPolygon)
    #创建字段
    field_name = ogr.FieldDefn("Name_ID",ogr.OFTInteger64)
    layer.CreateField(field_name)
    field_name = ogr.FieldDefn("Name", ogr.OFTString)
    field_name.SetWidth(15)
    layer.CreateField(field_name)
    #stride
    stride = step-overlapnum
    #插入要素
    # x1_lst = np.arange(minLng,maxLng-step+stride*0.001,stride)
    # x2_lst = x1_lst+step
    # y1_lst = np.arange(minLat,maxLat-step+stride*0.001,stride)
    # y2_lst = y1_lst+step
    x1_lst = np.arange(minLng,maxLng,stride)
    x2_lst = x1_lst+step
    y1_lst = np.arange(minLat,maxLat,stride)
    y2_lst = y1_lst+step
    if x2_lst[-1]>maxLng:
        x2_lst[-1] = maxLng
        x1_lst[-1] = maxLng-step
    elif x2_lst[-1] < maxLng:
        x2_lst.append(maxLng)
        x1_lst.append(maxLng-step)
    if y2_lst[-1]>maxLat:
        y2_lst[-1] = maxLat
        y1_lst[-1] = maxLat-step
    elif y2_lst[-1]<maxLat:
        y2_lst.append(maxLat)
        y1_lst.append(maxLat-step)

    # xlst = np.arange(minLng,maxLng+step,step)
    # ylst = np.arange(minLat,maxLat+step,step)
    xlen = len(x1_lst)
    ylen = len(y1_lst)
    id = 0
    for i in range(0,xlen):
        for j in range(0,ylen):
            id = id +1
            x1 = '%.1f' % x1_lst[i]
            y1 = '%.1f' % y1_lst[j]
            x2 = '%.1f' % x2_lst[i]
            y2 = '%.1f' % y2_lst[j]
            feature = ogr.Feature(layer.GetLayerDefn())
            feature.SetField("Name_ID",str(int(id)))
            #左下角
            name = y2+"_"+x1
            feature.SetField("Name",name)
            #创建几何
            wkt = 'POLYGON(('+x1+' '+y1+','+x2+' '+y1+','+x2+' '+y2+','+x1+' '+y2+','+x1+' '+y1+'))'
            poly = ogr.CreateGeometryFromWkt(wkt)
            feature.SetGeometry(poly)
            layer.CreateFeature(feature)
    feature = None
    data_source = None
# outdir = r'D:\04study\00Paper\Dissertation\01experiment\00data\02girds'
# shpname = 'grids_2degree_overlap03'
# createFishnet_5degree_WGS84_overlap(outdir,shpname,minLng=-180,maxLng=180,minLat=-65,maxLat=75,step=2,overlapnum=0.3)
# print('grids_2degree_overlap02 finish!')

def createFishnet_WGS84_lt1(outdir,shpname,minLng=-180,maxLng=180,minLat=-65,maxLat=75,step=0.5):
    '''
    创建经纬格网shapefile,夜间灯光覆盖范围。
    :param outdir: 输出目录
    :param shpname: shp名称，不包括后缀名
    :param minLng: 覆盖范围的最小经度
    :param maxLng:  覆盖范围的最大经度
    :param minLat:  覆盖范围的最小纬度
    :param maxLat:  覆盖范围的最大纬度
    :param step: 格网大小
    :return: shapefile
    '''
    # shp_path = r'D:\04study\00Paper\Dissertation\01experiment\00data\02girds\grids_5degree.shp'
    shp_path = outdir + "\\"+shpname+".shp"
    #注册驱动驱动，这里是ESRI Shapefile类型
    driver = ogr.GetDriverByName("ESRI Shapefile")
    #创建数据源
    data_source = driver.CreateDataSource(shp_path)
    #注入投影信息，这里使用4326，表示WGS84经纬坐标
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326) #这是WGS84
    #创建图层，图层名称和上面注册驱动的shp名称一致
    layer = data_source.CreateLayer(shpname,srs,ogr.wkbPolygon)
    #创建字段
    field_name = ogr.FieldDefn("Name_ID",ogr.OFTInteger64)
    layer.CreateField(field_name)
    field_name = ogr.FieldDefn("Name", ogr.OFTString)
    field_name.SetWidth(15)
    layer.CreateField(field_name)
    #插入要素
    xlst = np.arange(minLng,maxLng+step,step)
    ylst = np.arange(minLat,maxLat+step,step)
    xlen = len(xlst)
    ylen = len(ylst)
    id = 0
    for i in range(0,xlen-1):
        for j in range(0,ylen-1):
            id = id +1
            x1 = format(xlst[i],'.1f')
            y1 = format(ylst[j],'.1f')
            x2 = format(xlst[i+1],'.1f')
            y2 = format(ylst[j+1],'.1f')
            feature = ogr.Feature(layer.GetLayerDefn())
            feature.SetField("Name_ID",str(int(id)))
            #左下角
            lf = round(ylst[j+1]*10)
            rg = round(xlst[i]*10)
            strNameId = str(int(lf))+'_'+str(int(rg))
            feature.SetField("Name",strNameId)
            #创建几何
            wkt = 'POLYGON(('+x1+' '+y1+','+x2+' '+y1+','+x2+' '+y2+','+x1+' '+y2+','+x1+' '+y1+'))'
            poly = ogr.CreateGeometryFromWkt(wkt)
            feature.SetGeometry(poly)
            layer.CreateFeature(feature)
    feature = None
    data_source = None
# outdir = r'D:\04study\00Paper\Dissertation\01experiment\00data\02girds'
# shpname = 'grids_0-5degree'
# createFishnet_WGS84_lt1(outdir,shpname)


# multi value to points
def multiValuesToPints(shp_path,ras_path):
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
    # 获取要素及要素地理位置
    xValues = []
    yValues = []
    labelValues = []
    pidValues = []
    labelField = "label1"
    pidField =  "POINTID"
    feature = layer.GetNextFeature()
    while feature:
        geometry = feature.GetGeometryRef()
        x = geometry.GetX()
        y = geometry.GetY()
        label = feature.GetField(labelField)
        pid = feature.GetField(pidField)
        xValues.append(x)
        yValues.append(y)
        labelValues.append(label)
        pidValues.append(pid)
        print(pid)
        feature = layer.GetNextFeature()
    #############获取点位所在像元的栅格值
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
        s = s + str(value) + ' '
        print(s)
        values.append(s)
    return values,labelValues,pidValues

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
            s = s + str(value) + ' '
            # print(s)
            values.append(value)
        ras_inDs = None
        multiValues.append(values)
    #转为pd.dataframe
    colNames = []
    result_dict = {}
    for i in range(0,len(fieldLst)):
        colNames.append(fieldLst[i])
        result_dict[fieldLst[i]] = fieldValues[i]
    for i in range(0,len(rasNameLst)):
        colNames.append(rasNameLst[i])
        result_dict[rasNameLst[i]] = multiValues[i]

    result_df = pd.DataFrame(data=result_dict,columns=colNames) #固定列顺序
    return result_df

# shp_path = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\2000\sample\sample_Points\01\randomPoints.shp'
# ras_path1 = r'D:\01data\00整理\02夜间灯光\dmsp\avg_stepwise\2000.tif'
# ras_path2 = r'D:\01data\00整理\02夜间灯光\dmsp\grc\F12-F15_2000.tif'
# ras_pathLst = [ras_path1,ras_path2]
# rasNameLst = ['2000','F12-F15_2000']
# csvpath = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\2000\sample\sample_Points\01\2000_randomPoints.csv'
# fieldLst = ['CID']
# result_df = getRasterValueByPoints(shp_path,'',ras_pathLst,rasNameLst,fieldLst)
# result_df.to_csv(csvpath)



#vector clip raster
def world2Pixel(geoMatrix, x, y):
  """
  Uses a gdal geomatrix (gdal.GetGeoTransform()) to calculate
  the pixel location of a geospatial coordinate
  """
  ulX = geoMatrix[0]
  ulY = geoMatrix[3]
  xDist = geoMatrix[1]
  yDist = geoMatrix[5]
  rtnX = geoMatrix[2]
  rtnY = geoMatrix[4]
  pixel = int((x - ulX) / xDist)
  line = int((ulY - y) / xDist)
  return (pixel, line)

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

    dataset.SetGeoTransform(im_geotrans)
    dataset.SetProjection(im_proj)
    if im_bands == 1:
        dataset.GetRasterBand(1).WriteArray(im_data)
    else:
        for i in range(im_bands):
            dataset.GetRasterBand(i+1).WriteArray(im_data[i])

    del dataset

def shpClipRaster_LyrExtent(shapefile_path, raster_path, save_path):
    # Load the source data as a gdalnumeric array
    # srcArray = gdalnumeric.LoadFile(raster_path)

    # Also load as a gdal image to get geotransform
    # (world file) info
    srcImage = gdal.Open(raster_path)
    geoTrans = srcImage.GetGeoTransform()
    geoProj = srcImage.GetProjection()

    # Create an OGR layer from a boundary shapefile
    shapef = ogr.Open(shapefile_path)
    lyr = shapef.GetLayer( os.path.split( os.path.splitext( shapefile_path )[0] )[1] )
    poly = lyr.GetNextFeature()
    geometry = poly.GetGeometryRef()
    # minX, maxX, minY, maxY = geometry.GetEnvelope()
    # Convert the layer extent to image pixel coordinates
    minX, maxX, minY, maxY = lyr.GetExtent()
    ulX, ulY = world2Pixel(geoTrans, minX, maxY)
    lrX, lrY = world2Pixel(geoTrans, maxX, minY)

    # Calculate the pixel size of the new image
    pxWidth = int(lrX - ulX)
    pxHeight = int(lrY - ulY)

    # clip = srcArray[:, ulY:lrY, ulX:lrX]
    clip = srcImage.ReadAsArray(ulX,ulY,pxWidth,pxHeight)   #***只读要的那块***

    #
    # EDIT: create pixel offset to pass to new image Projection info
    #
    xoffset =  ulX
    yoffset =  ulY
    print ("Xoffset, Yoffset = ( %f, %f )" % ( xoffset, yoffset ))

    # Create a new geomatrix for the image
    geoTrans = list(geoTrans)
    geoTrans[0] = minX
    geoTrans[3] = maxY

    write_img(save_path, geoProj, geoTrans, clip)
    gdal.ErrorReset()

def envelopeClipRaster(envelopeCoords,raster_path, save_path):
    # Also load as a gdal image to get geotransform
    # (world file) info
    srcImage = gdal.Open(raster_path)
    geoTrans = srcImage.GetGeoTransform()
    geoProj = srcImage.GetProjection()

    minX, maxX, minY, maxY = envelopeCoords[0],envelopeCoords[1],envelopeCoords[2],envelopeCoords[3]
    ulX, ulY = world2Pixel(geoTrans, minX, maxY)
    lrX, lrY = world2Pixel(geoTrans, maxX, minY)
    # Calculate the pixel size of the new image
    pxWidth = int(lrX - ulX)
    pxHeight = int(lrY - ulY)
    # clip = srcArray[:, ulY:lrY, ulX:lrX]
    clip = srcImage.ReadAsArray(ulX, ulY, pxWidth, pxHeight)  # ***只读要的那块***
    #
    # EDIT: create pixel offset to pass to new image Projection info
    #
    xoffset = ulX
    yoffset = ulY
    print("Xoffset, Yoffset = ( %f, %f )" % (xoffset, yoffset))
    # Create a new geomatrix for the image
    geoTrans = list(geoTrans)
    geoTrans[0] = minX
    geoTrans[3] = maxY

    rasterOp.write_img(save_path, geoProj, geoTrans, clip)

def shpClipRaster_EachFeature(shapefile_path, raster_path, save_dir,nameField):
    # Create an OGR layer from a boundary shapefile
    shapef = ogr.Open(shapefile_path)
    lyr = shapef.GetLayer( os.path.split( os.path.splitext( shapefile_path )[0] )[1] )
    poly = lyr.GetNextFeature()

    while poly:
        geometry = poly.GetGeometryRef()
        minX, maxX, minY, maxY = geometry.GetEnvelope()
        envelopeCoords = [minX, maxX, minY, maxY]
        name = poly.GetField(nameField)
        save_path = os.path.join(save_dir,name+'.tif')

        if not os.path.exists(save_path):
            print(name)
            # Convert the layer extent to image pixel coordinates
            envelopeClipRaster(envelopeCoords,raster_path, save_path)
        poly = lyr.GetNextFeature()

    print('finish!')

# raster_path = r'D:\01data\00整理\02夜间灯光\dmsp\grc\F12-F15_2000.tif'
# shapefile_path = r'D:\04study\00Paper\Dissertation\01experiment\00data\02girds\GAIA_nameID_1deg.shp'
# save_dir = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\2000\grc'
# nameField = 'fName_ID'
# # gdal.UseExceptions()
# shpClipRaster_EachFeature(shapefile_path, raster_path, save_dir,nameField)
# # gdal.ErrorReset()



'''矢量图层基本操作'''
def openShp(shppath):
    driver = ogr.GetDriverByName('ESRI Shapefile')
    shp_ds = ogr.Open(shppath, 0)
    lyr = shp_ds.GetLayer(os.path.split(os.path.splitext(shppath)[0])[1])
    return shp_ds,lyr

#空间查询示例
def spatialFilter(layer,geometry):
    layer.SetSpatialFilter(geometry)
    # # 打印id
    # feat_inNibley = layer.GetNextFeature()
    # while feat_inNibley:
    #     print(feat_inNibley.GetField('ID'))
    #     feat_inNibley.Destroy()
    #     feat_inNibley = layer.GetNextFeature()
    return layer

#属性查询示例
def attributeFilter(layer,where):
    layer.SetAttributeFilter(where)
    # #筛选name是Nibley的要素
    # layer.SetAttributeFilter("name = 'Nibley'")
    # # lay_Nibley = ds_town.ExecuteSQL("select * from cache_towns where name = 'Nibley'")
    # print("name 是Nibley的要素个数： %d"%layer.GetFeatureCount())
    # Nibleyfeat = layer.GetFeature(0)
    # # 扩大范围
    # geom = Nibleyfeat.GetGeometryRef()
    # buffergeom = geom.Buffer(1500)
    return layer

