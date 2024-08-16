import numpy as np
import pandas as pd
from osgeo import ogr,osr
from scipy.optimize import  curve_fit
import os
import gc
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
os.environ['PROJ_LIB'] = r'C:\ProgramData\Anaconda3\Lib\site-packages\osgeo\data\proj'

import rasterOp
import common

'''stable DMSP NTL 逐步年际校正'''
class Stable_DMSP_Stepwise:
    def __init__(self):
        # self.raw_dir = r'D:\01data\00整理\02夜间灯光\dmsp\raw'
        # self.stepwise_dir = r'D:\01data\00整理\02夜间灯光\dmsp\stepwise'
        self.raw_dir = r'D:\01data\00整理\02夜间灯光\dmsp\raw'
        self.stepwise_dir = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\outdata\02oriGNL_stepSNL\01stepwise'
        self.avg_stepwise_dir_year = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\outdata\02oriGNL_stepSNL\02avg'
        self.F10_yearLst = [1992,1993,1994]
        self.F12_yearLst = [1994, 1995, 1996,1997,1998,1999]
        self.F14_yearLst = [1997, 1998, 1999,2000,2001,2002,2003]
        self.F15_yearLst = [2000,2001,2002,2003,2004,2005,2006,2007]
        self.F16_yearLst = [2004,2005,2006,2007,2008,2009]
        self.F18_yearLst = [2010,2011,2012,2013]
    def _adjust_F14(self):
        a0 = -0.078
        a1 = 1.4588
        a2 = -0.0073
        for year in self.F14_yearLst:
            oldpath = os.path.join(self.raw_dir,'F14'+str(year)+'.tif')
            newpath = os.path.join(self.stepwise_dir,'F14'+str(year)+'.tif')
            print(oldpath)
            inDs,inBand, inData, vnoValue = rasterOp.getRasterData(oldpath)
            outData = a0+inData*a1+inData*inData*a2
            rasterOp.outputResult(inDs, outData, newpath)
            del inDs,inBand, inData, vnoValue,outData
            gc.collect()

    def _adjust_F15(self):
        ajYeads = np.arange(2003,2008,1)
        a0 = -0.4597
        a1 = 1.714
        a2 = -0.0114
        for year in ajYeads:
            oldpath = os.path.join(self.raw_dir,'F15'+str(year)+'.tif')
            newpath = os.path.join(self.stepwise_dir,'F15'+str(year)+'.tif')
            print(oldpath)
            inDs,inBand, inData, vnoValue = rasterOp.getRasterData(oldpath)
            outData = a0+inData*a1+inData*inData*a2
            rasterOp.outputResult(inDs, outData, newpath)
            del inDs,inBand, inData, vnoValue,outData
            gc.collect()

    def _adjust_F16(self):
        ajYeads = np.arange(2004,2010,1)
        parasLst = [[0.1194,1.2265,-0.0041],[-0.3209,1.4619,-0.0072],[0.0877,1.1616,-0.0021],
                    [0,1,0],[0.11,1.0513,-0.001],[0.6294,1.1188,-0.0024]]
        a0 = -1.2802
        a1 = 1.3313
        a2 =-0.0055
        for i in range(len(ajYeads)):
            year = ajYeads[i]
            para = parasLst[i]
            oldpath = os.path.join(self.raw_dir,'F16'+str(year)+'.tif')
            newpath = os.path.join(self.stepwise_dir,'F16'+str(year)+'.tif')
            print(oldpath)
            inDs,inBand, inData, vnoValue = rasterOp.getRasterData(oldpath)
            outData1 = para[0]+inData*para[1]+inData*inData*para[2]
            outData = a0 + outData1 * a1 + outData1 * outData1 * a2
            rasterOp.outputResult(inDs, outData, newpath)
            del inDs,inBand, inData, vnoValue,outData1,outData
            gc.collect()

    def _adjust_F18(self):
        year = 2010
        a0 = -0.0861
        a1 = 0.821
        a2 = 0.002
        oldpath = os.path.join(self.raw_dir,'F18'+str(year)+'.tif')
        newpath = os.path.join(self.stepwise_dir,'F18'+str(year)+'.tif')
        print(oldpath)
        inDs,inBand, inData, vnoValue = rasterOp.getRasterData(oldpath)
        outData = a0+inData*a1+inData*inData*a2
        rasterOp.outputResult(inDs, outData, newpath)
        del inDs,inBand, inData, vnoValue,outData
        gc.collect()

    def _avg_each_year(self):
        if not os.path.exists(self.avg_stepwise_dir_year):
            os.makedirs(self.avg_stepwise_dir_year)
        for year in range(1992,2014):
            tarpath = os.path.join(self.avg_stepwise_dir_year, str(year) + '.tif')
            if os.path.exists(tarpath):
                continue
            tif_Files = []
            if year in self.F10_yearLst:
                tif_Files.append("F10"+str(year))
            if year in self.F12_yearLst:
                tif_Files.append("F12"+str(year))
            if year in self.F14_yearLst:
                tif_Files.append("F14"+str(year))
            if year in self.F15_yearLst:
                tif_Files.append("F15"+str(year))
            if year in self.F16_yearLst:
                tif_Files.append("F16"+str(year))
            if year in self.F18_yearLst:
                tif_Files.append("F18"+str(year))
            print(tif_Files)
            if len(tif_Files) == 1:
                srcpath = os.path.join(self.stepwise_dir,tif_Files[0]+'.tif')
                command = 'copy {0} {1}'.format(srcpath,tarpath)
                os.system(command)
            else:
                path1 = os.path.join(self.stepwise_dir,tif_Files[0]+'.tif')
                inDs, inBand, inData, vnoValue = rasterOp.getRasterData(path1)
                path2 = os.path.join(self.stepwise_dir, tif_Files[1] + '.tif')
                inDs2, inBand2, inData2, vnoValue2 = rasterOp.getRasterData(path2)
                outdata = (inData+inData2)*0.5
                outdata[(inData<=0)|(inData2<=0)] = 0
                del inData,inData2
                gc.collect()
                rasterOp.outputResult(inDs, outdata, tarpath)
                del inDs, inBand, vnoValue,inDs2, inBand2, vnoValue2,outdata
                gc.collect()

    def run(self):
        # self._adjust_F14()
        # self._adjust_F15()
        # self._adjust_F16()
        # self._adjust_F18()
        self._avg_each_year()
        print('finish')
# model = Stable_DMSP_Stepwise()
# model.run()

'''grc DMSP NTL 年际校正'''
class DMSP_GRC_Process():
    def interAnnualCalibration(self,in_dir,grc_imgname_lst,out_dir):
        '''
        Global Radiance Calibrated Nighttime Lights Product:年际互校准
        NOAA/NGDC 提供校正参数，参考数据集为F16_20051128-20061224_rad_v4
        https://www.ngdc.noaa.gov/eog/dmsp/download_radcal.html  readme
        :param in_dir: grc 影像所在目录
        :param grc_imgname_lst: 影像名称，不带后缀名，顺序与calib_coe_lst对应
        :param out_dir:校正后输出影像所在目录
        :return:
        '''
        calib_coe_lst = [['1996',4.336,0.915],['1999',1.423,0.780],
                         ['2000',3.658,0.710],['2003',3.736,0.797],
                         ['2004',1.062,0.761],['2010',2.196,1.195]] #[年份，co,c1]
        for i in range(0,len(calib_coe_lst)):
            coes = calib_coe_lst[i]
            print(coes[0])
            imgpath = os.path.join(in_dir,grc_imgname_lst[i]+'.tif')
            out_imgpath = os.path.join(out_dir,coes[0]+'.tif')
            inDs,inBand, inData, vnoValue = rasterOp.getRasterData(imgpath)
            outData = coes[1]+coes[2]*inData
            rasterOp.outputResult(inDs, outData, out_imgpath)
            inDs=None; inBand=None; inData=None; vnoValue=None; outData=None;
            gc.collect()

    def run(self):
        in_dir = r'D:\01data\00整理\02夜间灯光\dmsp\grc'
        out_dir = r''
        grc_imgname_lst = ['F12_1996','F12_1999','F12-F15_2000','F14-F15_2002','F14_2004','F16_2010']
        self.interAnnualCalibration(in_dir,grc_imgname_lst,out_dir)
# grc_pro = DMSP_GRC_Process()
# grc_pro.run()

class VNL_Process():
    '''小于1的VNL设为0'''
    def set0_DNlte1(self,inpath,outpath):
        inDs, inBand, inData, vnoValue = rasterOp.getRasterData(inpath)
        inData[inData<=1] = 0
        inData[inData == vnoValue] = 0
        rasterOp.outputResult(inDs,inData,outpath)
# indir = r'D:\01data\00整理\02夜间灯光\npp\annualV2'
# outdir = r'D:\01data\00整理\02夜间灯光\npp\annualV2_set0'
# vnl_proc = VNL_Process()
# for i in range(2012,2014):
#     inpath = os.path.join(indir,'VNL_'+str(i)+'.tif')
#     outpath = os.path.join(outdir, 'VNL_' + str(i) + '_set0.tif')
#     vnl_proc.set0_DNlte1(inpath,outpath)



'''通过记录DMSP图像块的名称来选择训练、测试、验证样本集，用于后续训练
   举例：1°*1°格网裁剪得到DMSP图像块存储在一个文件夹中，然后将选择的图像块名称写在txt文件中，作为数据集。
'''
class Sample_DMSP_Block:
    def _cal_brightPixels_percent(self,img_path,minValue,maxValue):
        inDs, inBand, inData, vnoValue = rasterOp.getRasterData(img_path)
        width = inDs.RasterXSize
        height = inDs.RasterYSize
        geoTrans = inDs.GetGeoTransform()
        geoProj = inDs.GetProjection()
        totalnum = width * height
        zeropixels_num = len(inData[(inData == 0 )])
        # if zeropixels_num/totalnum > 0.9:
        #     return 0,geoTrans,geoProj
        brightpixels_num = len(inData[(inData > minValue)&(inData <= maxValue)])
        percent = brightpixels_num*1.0/totalnum
        inDs = None;inData = None;inBand=None
        return percent,geoTrans,geoProj

    def _create_shp(self,shppath,geometryLst,nameLst):
        shpname = os.path.splitext(os.path.basename(shppath))[0]
        # 注册驱动驱动，这里是ESRI Shapefile类型
        driver = ogr.GetDriverByName("ESRI Shapefile")
        # 创建数据源
        data_source = driver.CreateDataSource(shppath)
        # 注入投影信息，这里使用4326，表示WGS84经纬坐标
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)  # 这是WGS84
        # 创建图层，图层名称和上面注册驱动的shp名称一致
        layer = data_source.CreateLayer(shpname, srs, ogr.wkbPolygon)
        # 创建字段
        field_name = ogr.FieldDefn("Name_ID", ogr.OFTInteger64)
        layer.CreateField(field_name)
        field_name = ogr.FieldDefn("Name", ogr.OFTString)
        field_name.SetWidth(15)
        layer.CreateField(field_name)
        # 插入要素
        id = 0
        for i in range(0, len(geometryLst)):
            id = id + 1
            feature = ogr.Feature(layer.GetLayerDefn())
            feature.SetField("Name_ID", str(int(id)))
            feature.SetField("Name", nameLst[i])
            poly = ogr.CreateGeometryFromWkt(geometryLst[i])
            feature.SetGeometry(poly)
            layer.CreateFeature(feature)
        feature = None
        data_source = None

    def select_DMSP_Block(self,block_dir,minValue,maxValue,threshold,saveTxtpath,shppath,step):
        geoemtryLst = []
        nameLst = []
        for root,dirs,files in os.walk(block_dir):
            for file in files:
                img_path = os.path.join(block_dir,file)
                percent,geoTrans,geoProj = self._cal_brightPixels_percent(img_path,minValue,maxValue)
                if percent >= threshold:
                    name = os.path.splitext(file)[0]
                    nameLst.append(name)
                    x1,y1,x2,y2 = str(geoTrans[0]),str(geoTrans[3]-step),str(geoTrans[0]+step),str(geoTrans[3])
                    wkt = 'POLYGON((' + x1 + ' ' + y1 + ',' + x2 + ' ' + y1 + ',' + x2 + ' ' + y2 + ',' + x1 + ' ' + y2 + ',' + x1 + ' ' + y1 + '))'
                    geoemtryLst.append(wkt)
                    with open(saveTxtpath,'a') as f:
                        f.write(name+'\n')
        # save to shp
        if(len(nameLst)>0):
            self._create_shp(shppath, geoemtryLst, nameLst)

    def img_block_name_to_txt(self,block_dir,saveTxtpath):
        '''
        :param block_dir:
        :param saveTxtpath:
        :return:
        '''
        for root,dirs,files in os.walk(block_dir):
            for file in files:
                name = os.path.splitext(file)[0]
                with open(saveTxtpath, 'a') as f:
                    f.write(name + '\n')

    def dbf_block_name_to_txt(self,dbfpath,saveTxtpath):
        fileop = common.FileOpr()
        df = fileop.readDBF_asDataFrame(dbfpath)
        subdf = df['Name']
        for index,item in subdf.iteritems():
            ss = str(item).strip()
            with open(saveTxtpath,'a') as f:
                f.write(ss+'\n')

    def dbf_block_name_to_train_test_valid_txt(self,dbfpath,savedirpath):
        fileop = common.FileOpr()
        df = fileop.readDBF_asDataFrame(dbfpath)
        trainset = df[df['totalTrain']=='train']
        testset = df[df['totalTrain']=='test']
        validset = df[df['totalTrain']=='valid']
        subdf = trainset['Name']
        saveTxtpath = os.path.join(savedirpath,'train_09_Aridity.txt')
        for index,item in subdf.iteritems():
            ss = str(item).strip()
            with open(saveTxtpath,'a') as f:
                f.write('2012'+ss+'\n')
            with open(saveTxtpath,'a') as f:
                f.write('2013'+ss+'\n')
        subdf = testset['Name']
        saveTxtpath = os.path.join(savedirpath,'test_09_Aridity.txt')
        for index,item in subdf.iteritems():
            ss = str(item).strip()
            with open(saveTxtpath,'a') as f:
                f.write('2012'+ss+'\n')
            with open(saveTxtpath,'a') as f:
                f.write('2013'+ss+'\n')
        subdf = validset['Name']
        saveTxtpath = os.path.join(savedirpath,'valid_09_Aridity.txt')
        for index,item in subdf.iteritems():
            ss = str(item).strip()
            with open(saveTxtpath,'a') as f:
                f.write('2012'+ss+'\n')
            with open(saveTxtpath,'a') as f:
                f.write('2013'+ss+'\n')

    def select_train_test_valid(self,block_txt_path,save_dir,sampleNumber,trainnum,testnum,validnum):
        train_path = os.path.join(save_dir,'train_'+str(sampleNumber)+'.txt')
        test_path = os.path.join(save_dir, 'test_' + str(sampleNumber) + '.txt')
        valid_path = os.path.join(save_dir, 'val_' + str(sampleNumber) + '.txt')
        with open(block_txt_path,'r') as f:
            lines = f.readlines()
        #train
        train_indexArr = np.random.choice(a=len(lines), size=trainnum, replace=False, p=None)
        with open(train_path,'a') as f:
            for ind in train_indexArr:
                f.write(lines[ind])
        #test
        test_lines = [lines[i] for i in range(0,len(lines)) if i not in train_indexArr]
        test_indexArr = np.random.choice(a=len(test_lines), size=testnum, replace=False, p=None)
        with open(test_path,'a') as f:
            for ind in test_indexArr:
                f.write(test_lines[ind])
        #val
        valid_lines = [test_lines[i] for i in range(0,len(test_lines)) if i not in test_indexArr]
        valid_indexArr = np.random.choice(a=len(valid_lines), size=validnum, replace=False, p=None)
        with open(valid_path,'a') as f:
            for ind in valid_indexArr:
                f.write(valid_lines[ind])

    def select_train_test_valid_From_DBF(self,dbfpath,outCSVpath,PidField,tagField,trainnum,testnum,validnum,trainTxtpath,testTxtpath,validTxtpath):
        fileop = common.FileOpr()
        df = fileop.readDBF_asDataFrame(dbfpath)
        subdf = df[df[tagField] == "1"]
        trainset = subdf.sample(n=trainnum, replace=False, random_state=None, axis=0)
        trainset[tagField] = "train"
        namelst = subdf[PidField].values.tolist()
        train_namelst = trainset[PidField].values.tolist()
        for name in train_namelst:
            namelst.remove(name)
        subdf = subdf[subdf[PidField].isin(namelst)]
        testset = subdf.sample(n=testnum, replace=False, random_state=None, axis=0)
        testset[tagField] = "test"
        test_namelst = testset[PidField].values.tolist()
        for name in test_namelst:
            namelst.remove(name)
        subdf = subdf[subdf[PidField].isin(namelst)]
        validset = subdf.sample(n=validnum, replace=False, random_state=None, axis=0)
        validset[tagField] = "valid"
        result = pd.concat([trainset,testset,validset],axis=0)
        result.to_csv(outCSVpath,header=True,index=False)
        train_namelst = trainset['Name'].values.tolist()
        fileop.appendToFile(trainTxtpath, train_namelst)
        test_namelst = testset['Name'].values.tolist()
        fileop.appendToFile(testTxtpath, test_namelst)
        valid_namelst = validset['Name'].values.tolist()
        fileop.appendToFile(validTxtpath, valid_namelst)

    def select_train_test(self, block_txt_path, save_dir, trainnum, testnum):
        train_path = os.path.join(save_dir, 'train.txt')
        test_path = os.path.join(save_dir, 'test.txt')
        with open(block_txt_path, 'r') as f:
            lines = f.readlines()
        # train
        train_indexArr = np.random.choice(a=len(lines), size=trainnum, replace=False, p=None)
        with open(train_path, 'a') as f:
            for ind in train_indexArr:
                f.write(lines[ind])
        # test
        test_lines = [lines[i] for i in range(0, len(lines)) if i not in train_indexArr]
        test_indexArr = np.random.choice(a=len(test_lines), size=testnum, replace=False, p=None)
        with open(test_path, 'a') as f:
            for ind in test_indexArr:
                f.write(test_lines[ind])

# sampleCls = Sample_DMSP_Block()
# dir = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\02'
# dbfpath = os.path.join(dir,'lt50_gt10_valid.dbf')
# # saveTxtpath = os.path.join(dir,'valid_03.txt')
# saveTxtpath = os.path.join(dir,'valid_03_lt50_gt10.txt')
# sampleCls.dbf_block_name_to_txt(dbfpath,saveTxtpath)
# block_dir = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\2000\stl'
# minValue = 1
# maxValue = 60
# threshold = 0.7
# saveTxtpath = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\2000\sample\1\bright_1_60_7.txt'
# shppath = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\2000\sample\1\bright_1_60_7.shp'
# sampleCls.select_DMSP_Block(block_dir,minValue,maxValue,threshold,saveTxtpath,shppath)

# block_txt_path =r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\2000\sample\1\bright_1_60_7.txt'
# save_dir = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\2000\sample\1'
# sampleNumber = 1
# trainnum = 500
# testnum = 150
# validnum = 150
# sampleCls.select_train_test_valid(block_txt_path,save_dir,sampleNumber,trainnum,testnum,validnum)

# sample_block = Sample_DMSP_Block()
# block_dir = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img\2012_DNL'
# block_txt_path = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\valid_selected_litLargeThan50_block.txt'
# # sample_txt_dir = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\01'
# # trainnum = 2100
# # testnum = 500
# sample_block.img_block_name_to_txt(block_dir,block_txt_path)
# sample_block.select_train_test(block_txt_path,sample_txt_dir,trainnum,testnum)


'''SNTL与GNTL进行回归分析，进行饱和校正'''
class Analysis_Gntl_and_Sntl:
    def __init__(self):
        self.global_point_01_csvpath = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\pointAnalysis\ori\01global_sample.csv'

    def getData(self,csvpath):
        df = pd.read_csv(csvpath)
        return df

    '''绘制回归后得到的直线和散点图'''
    def regression_linear(self,df,xcol,ycol,title,pos_x,pos_y,off_y):
        x = np.array(df[xcol].to_list()).reshape(-1, 1)
        y = np.array(df[ycol].to_list()).reshape(-1, 1)

        # seed = 1  # 随机种子
        # validation = 0.25  # 测试与训练比
        # X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=validation)
        model = LinearRegression()
        model.fit(x, y)
        k = model.coef_[0][0]
        b = model.intercept_[0]
        #  print(linreg.intercept_)       常数b
        #  print(linreg.coef_)            系数w

        # 画图比较预测结果
        x0 = np.arange(x.min(),x.max(),1)
        y0 = k * x0 + b
        plt.figure()
        plt.scatter(x, y)
        plt.plot(x0, y0,'r')
        plt.text(pos_x, pos_y+off_y*2, 'k = ' + str(k))
        plt.text(pos_x, pos_y+off_y, 'b = ' + str(b))
        plt.text(pos_x, pos_y, 'score = ' + str(model.score(x, y)))
        plt.title(title)

    '''返回k,b,score'''
    def regression_linear_kbscore(self,df,xcol,ycol):
        x = np.array(df[xcol].to_list()).reshape(-1, 1)
        y = np.array(df[ycol].to_list()).reshape(-1, 1)

        # seed = 1  # 随机种子
        # validation = 0.25  # 测试与训练比
        # X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=1, test_size=validation)
        model = LinearRegression()
        model.fit(x, y)
        k = model.coef_[0][0]
        b = model.intercept_[0]
        score = model.score(x, y)
        return k,b,score

    def different_UL_regression_linear(self,df,xcol,ycol,title,ULlst):
        klst = [];blst = [];scorelst = []
        for ul in ULlst:
            sub = df[df[xcol] <= ul]
            k,b,score = self.regression_linear_kbscore(sub,xcol,ycol)
            klst.append(k); blst.append(b); scorelst.append(score);
        figure = plt.figure()
        plt.suptitle(title)
        plt.subplot(2, 1, 1)
        plt.plot(ULlst,klst,'o-r')
        plt.title('cof')
        plt.subplot(2, 1, 2)
        plt.plot(ULlst, scorelst, '.-')
        plt.title('score')
        plt.show()
        return klst,blst,scorelst

    def analysis_2005(self):
        df = self.getData(self.global_point_01_csvpath)
        df_2005 = df[['pid', 'extent', 's_2005', 'g_2005']]
        #sntl 与 gntl散点图
        df_2005.plot.scatter(x='s_2005',y='g_2005')
        #sntl对应的gntl取平均后做散点图
        mean_gntl_2005 = df_2005.groupby(by=['s_2005'])['g_2005'].mean()
        mean_gntl_2005 = mean_gntl_2005.to_frame()
        mean_gntl_2005['s_2005'] = mean_gntl_2005.index.tolist()
        mean_gntl_2005.plot.scatter(x='s_2005', y='g_2005')
        # sntl对应的gntl取中值后做散点图
        median_gntl_2005 = df_2005.groupby(by=['s_2005'])['g_2005'].median()
        median_gntl_2005 = median_gntl_2005.to_frame()
        median_gntl_2005['s_2005'] = median_gntl_2005.index.tolist()
        median_gntl_2005.plot.scatter(x='s_2005', y='g_2005')
        #回归分析 UL=[10,20,30,40,50,55,60] df, mean, median 的回归关系
        xcol = 's_2005'
        ycol = 'g_2005'
        # #绘制回归后得到的直线和散点图
        # title = 'UL = 10'
        # sub = df_2005[df_2005['s_2005']<=10]
        # sub_mean  = mean_gntl_2005[mean_gntl_2005['s_2005']<=10]
        # sub_median = median_gntl_2005[median_gntl_2005['s_2005'] <= 10]
        # self.regression_linear(sub,xcol,ycol,title,0.8,40,20)
        # self.regression_linear(sub_mean, xcol, ycol, 'mean: '+title,0.8,10,5)
        # self.regression_linear(sub_median, xcol, ycol, 'median: '+title,0.8,10,5)
        # #得到不同UL下的回归结果
        # ULlst = np.arange(5,63,1)
        # klst,blst,scorelst = self.different_UL_regression_linear(df_2005,xcol,ycol,'whole',ULlst)
        # mean_klst, mean_blst, mean_scorelst = self.different_UL_regression_linear(mean_gntl_2005, xcol, ycol, 'mean',ULlst)
        # median_klst, median_blst, median_scorelst = self.different_UL_regression_linear(median_gntl_2005, xcol, ycol, 'median',ULlst)
        # #不同区域的回归结果
        # result = []
        # extentlst = ['Africa','Europe','NorthAmerica','NorthAsia','Oceania','SouthAmerica','SouthEastAsia']
        # extent = extentlst[0]
        # area_df = df[df['extent']==extent]
        # area_df_2005 = area_df[['s_2005','g_2005']]
        # ULlst = np.arange(5, 63, 1)
        # klst, blst, scorelst = self.different_UL_regression_linear(area_df_2005, xcol, ycol, extent, ULlst)
        # scorelst = np.array(scorelst)
        # klst = np.array(klst)
        # blst = np.array(blst)
        # result.append([extent,scorelst,klst,blst])
        #
        # a = result[-1]
        # print(a[0])
        # scorelst = a[1]
        # klst = a[2]
        # blst = a[3]
        # argmax = scorelst.argmax()
        # print(ULlst[argmax])
        # print(scorelst[argmax])

    def linear_func(self,x,a,b):
        return a*x+b

    def target_func1(self,x,a,b,c):
        return a*(x**b)+c

    def target_func3(self,x,a,b,c):
        return (np.log(x)/np.log(a))*b+c

    def target_func4(self,x,a,b,c):
        return np.power(x,a)*b+c

    def target_func5(self,x,a,b,c):
        return a+b*x+c*x**2

    def fit_fun_and_cal_Rsquare(self,target_func,xdata,ydata):
        popt, pcov = curve_fit(target_func, xdata, ydata)
        calc_ydata = target_func(xdata,*popt)
        res_ydata = np.array(ydata) - np.array(calc_ydata)
        ss_res = np.sum(res_ydata ** 2)
        ss_tot = np.sum((ydata - np.mean(ydata)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        return popt,r_squared

    def regression_nonlinear(self,target_func,df,xcol,ycol,title,pos_x,pos_y):
        x = np.array(df[xcol].to_list())
        y = np.array(df[ycol].to_list())
        popt,r_squared = self.fit_fun_and_cal_Rsquare(target_func,x,y)
        # 画图比较预测结果
        x0 = np.arange(x.min(),x.max(),1)
        y0 = target_func(x0,*popt)
        plt.figure()
        plt.scatter(x, y)
        plt.plot(x0, y0,'r')
        plt.text(pos_x, pos_y, 'score = ' + str(r_squared))
        plt.title(title)
        return popt,r_squared

    def regression_nonlinear_subplot(self,target_func,df,xcol,ycol,title,pos_x,pos_y):
        x = np.array(df[xcol].to_list())
        y = np.array(df[ycol].to_list())
        popt,r_squared = self.fit_fun_and_cal_Rsquare(target_func,x,y)
        # 画图比较预测结果
        x0 = np.arange(x.min(),x.max(),1)
        y0 = target_func(x0,*popt)
        plt.scatter(x, y)
        plt.plot(x0, y0,'r')
        plt.text(pos_x, pos_y, 'score = ' + str(r_squared))
        plt.title(title)
        return popt,r_squared

    def meanStl_oriGrc(self,df,xcol,ycol,func,thresholds,titlepfix,xpos,ypos,colnum):
        data = df[(df[xcol] > 0) & (df[ycol] > 0)]
        meandata =  data.groupby(by=[ycol])[xcol].mean()
        meandata = meandata.to_frame()
        meandata[ycol] = meandata.index.to_list()
        plt.figure()
        plt.suptitle(xcol+'~'+ycol)

        for i in range(0,len(thresholds)):
            thre = thresholds[i]
            plt.subplot(2, colnum, i + 1)
            a, b = self.regression_nonlinear_subplot(func, meandata[meandata[ycol] <= thre], xcol,
                                                    ycol,titlepfix + str(thre), xpos, thre - ypos)
            print(thre, a, b)

    def medianStl_oriGrc(self, df, xcol, ycol, func, thresholds, titlepfix, xpos, ypos):
        data = df[(df[xcol] > 0) & (df[ycol] > 0)]
        mediandata = data.groupby(by=[ycol])[xcol].median()
        mediandata = mediandata.to_frame()
        mediandata[ycol] = mediandata.index.to_list()
        plt.figure()
        plt.suptitle(xcol + '~' + ycol)
        colnum = len(thresholds) // 2
        for i in range(0, len(thresholds)):
            thre = thresholds[i]
            plt.subplot(2, colnum, i + 1)
            a, b = self.regression_nonlinear_subplot(func, mediandata[mediandata[ycol] <= thre], xcol,
                                                     ycol, titlepfix + str(thre), xpos, thre - ypos)
            print(thre, a, b)

    '''不同UL下的回归拟合'''
    def diffyear_meanStl_oriGrc(self,df,thre):
        recordfile = r'D:\04study\00Paper\Dissertation\03report\01实验相关\06时间序列NTL\fit.txt'
        grcYears = [1996,1999,2000,2002,2004,2006,2010]
        xcols = ['grc_'+str(i) for i in grcYears]
        F10Years = [1992,1993,1994]
        F12Years = [1994,1995,1996,1997,1998,1999]
        F14Years = [1997,1998,1999,2000,2001,2002,2003]
        F15Years = [2000,2001,2002,2003,2004,2005,2006,2007]
        F16Years = [2004,2005,2006,2007,2008,2009]
        F18Years = [2010,2011,2012,2013]
        ycols = []
        for year in F10Years:
            ycols.append('aF10'+str(year))
            ycols.append('F10' + str(year))
        for year in F12Years:
            ycols.append('aF12'+str(year))
            ycols.append('F12' + str(year))
        for year in F14Years:
            ycols.append('aF14'+str(year))
            ycols.append('F14' + str(year))
        for year in F15Years:
            ycols.append('aF15'+str(year))
            ycols.append('F15' + str(year))
        for year in F16Years:
            ycols.append('aF16'+str(year))
            ycols.append('F16' + str(year))
        for year in F18Years:
            ycols.append('aF18'+str(year))
            ycols.append('F18' + str(year))
        # thre = 50
        for xcol in xcols:
            # plt.figure()
            # plt.suptitle(xcol+'0-'+str(thre))
            i = 1
            for ycol in ycols:
                data = df[(df[xcol] > 0) & (df[ycol] > 0)]
                meandata = data.groupby(by=[ycol])[xcol].mean()
                meandata = meandata.to_frame()
                meandata[ycol] = meandata.index.to_list()
                # plt.subplot(4,9,i)
                # a, b = self.regression_nonlinear_subplot(self.linear_func, meandata[meandata[ycol] <= thre], xcol,ycol,
                #                                 ycol+'~'+xcol, 5, thre - 2)
                a, b = self.regression_nonlinear_subplot(self.linear_func, meandata[meandata[ycol] <= thre], xcol, ycol,
                                                         ycol + '~' + xcol, 5, thre - 2)
                ss = xcol+','+ycol+','+str(thre)+','
                for ai in a:
                    ss += str(ai)+','
                ss += str(b)
                with open(recordfile,'a+') as f:
                    f.write(ss+'\n')
                i = i+1
    # xcol = 'GNTL'
    # ycol = 'SNTL'
    # meandata = regdata.groupby(by=[xcol, ycol])
    # meanr2 = meandata['r2'].mean()
    # meanr2 = meanr2.to_frame()
    # a = meanr2.index.to_list()
    # grclst = [];
    # stllst = []
    # for ai in a:
    #     grclst.append(ai[0])
    #     stllst.append(ai[1])
    # r2 = meanr2['r2'].values
    # for i in range(0,len(r2)):
    #     grc = grclst[i]
    #     stl = stllst[i]
    #     r2_mean = r2[i]
    #     data = regdata[(regdata[xcol] == grc) & (regdata[ycol] == stl)]
    #     b = data[['UL','r2','k','b']].values
    #     b = b[b[:, 0].argsort()]
    #     b = b[::-1]
    #     thre = 99
    #     r2_thre = 0
    #     slope = 0
    #     intercept = 0
    #     lg_mean_lst = []
    #     lt_mean_lst = []
    #     flag = 1
    #     for j in range(0,len(b)):
    #         if b[j,1] <r2_mean:
    #             if flag < 0:
    #                 flag = 1
    #             if flag == 1:
    #                 lt_mean_lst.append(j)
    #             flag +=1
    #         else:
    #             if flag>0:
    #                 flag = -1
    #             if flag == -1:
    #                 lg_mean_lst.append(j)
    #             flag -=1
    #     if len(lg_mean_lst) == 1:
    #         ind = lg_mean_lst[0]
    #     else:
    #         lg_ind1 = lg_mean_lst[0]
    #         lg_ind2 = lg_mean_lst[1]
    #         lt_ind1 = lt_mean_lst[0]
    #         lt_ind2 = lt_mean_lst[1]
    #         d1 = lt_ind2-lg_ind1
    #         d2 = lg_ind2-lt_ind2
    #         d3 = lg_ind2-lg_ind1
    #         if d3 >=10:
    #             ind = lg_ind1
    #         else:
    #             if d1*2 < d2:
    #                 ind = lg_ind2
    #             else:
    #                 ind = lg_ind1
    #     ind = lg_mean_lst[0]
    #     thre = b[ind,0]
    #     r2_thre = b[ind,1]
    #     slope = b[ind,2]
    #     intercept =b[ind,3]
    #     ss = str(grc)+','+str(stl)+','+str(r2_mean)+','+str(thre)+','+str(r2_thre)+','+str(slope)+','+str(intercept)
    #     with open(recordfile,'a+') as f:
    #         f.write(ss+'\n')

    '''根据线性拟合参数进行饱和校正'''
    def update_SaturatedPixels_linear(self,sntlpath,gntlpath,UL,k,b,outpath):
        sinDs, sinBand, sinData, svnoValue = rasterOp.getRasterData(sntlpath)
        ginDs, ginBand, ginData, gvnoValue = rasterOp.getRasterData(gntlpath)
        saturatedIndexs = (sinData> UL)
        saturatedValues = ginData[saturatedIndexs]*k+b
        sinData[saturatedIndexs] = saturatedValues
        sinData[sinData<0]=0
        rasterOp.outputResult(ginDs, sinData, outpath)
        del sinDs, sinBand, sinData, svnoValue,ginDs, ginBand, ginData, gvnoValue,saturatedIndexs,saturatedValues
        gc.collect()
    # sntlfolder = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\outdata\02oriGNL_stepSNL\01stepwise'
    # gntlfolder = r'D:\01data\00整理\02夜间灯光\dmsp\grc'
    # outfolder = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\outdata\02oriGNL_stepSNL\02saturated'
    # ulcsvfile = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\outdata\02oriGNL_stepSNL\02saturated\saturatedPixel_UL.csv'
    # ul_df = pd.read_csv(ulcsvfile)
    # ana = Analysis_Gntl_and_Sntl()
    # gntlcol = 'GNTL';sntlcol='SNTL';ulcol = 'UL';kcol = 'k';bcol = 'b'
    # for index,row in ul_df.iterrows():
    #     print(row[sntlcol])
    #     gntlpath = os.path.join(gntlfolder,str(row[gntlcol])+'.tif')
    #     sntlpath = os.path.join(sntlfolder,row[sntlcol]+'.tif')
    #     outpath = os.path.join(outfolder,row[sntlcol]+'.tif')
    #     if os.path.exists(outpath):
    #         continue
    #     UL = row[ulcol]
    #     k = row[kcol]
    #     b = row[bcol]
    #     ana.update_SaturatedPixels_linear(sntlpath,gntlpath,UL,k,b,outpath)


class Result_Analysis:
    def __init__(self):
        self.recorfDir = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\world\analysis\totalNTLintensity'
        self.recordfile = r'D:\04study\00Paper\Dissertation\03report\03论文内容\图表\02NTL时间序列\03灯光总和.txt'
        self.raw_sntl_folder = r'F:\study\data\00整理\02夜间灯光\dmsp\raw\raw'
        self.raw_gntl_folder = r'F:\study\data\00整理\02夜间灯光\dmsp\grc\grc_calib'
        self.stepwise_sntl_folder = r'F:\study\data\00整理\02夜间灯光\dmsp\stepwise\01stepwise'
        self.saturated_sntl_folder = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\outdata\02oriGNL_stepSNL\02saturated'
        self.avg_saturated_sntl_folder = r'F:\study\data\00整理\02夜间灯光\dmsp\stepwise\02avg'
        self.harmonization_NTL_folder = r'F:\study\data\00整理\02夜间灯光\dmsp\harmonization NTL(1992-2018)'
        self.F10_yearLst = [1992, 1993, 1994]
        self.F12_yearLst = [1994, 1995, 1996, 1997, 1998, 1999]
        self.F14_yearLst = [1997, 1998, 1999, 2000, 2001, 2002, 2003]
        self.F15_yearLst = [2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007]
        self.F16_yearLst = [2004, 2005, 2006, 2007, 2008, 2009]
        self.F18_yearLst = [2010, 2011, 2012, 2013]
        self.grc_yearLst = [1996,1999,2000,2003, 2005,2010]
        # self.grc_yearLst = ['F12_1996','F12_1999','F12-F15_2000','F14_20040118-20041216_rad_v4.avg_vis','F14-F15_2002','F16_2005','F16_2010']

    '''计算累计DNs值'''
    def sum_of_lights(self,ntlpath,ntlname,recordfile):
        sinDs, sinBand, sinData, svnoValue = rasterOp.getRasterData(ntlpath)
        # sinData[sinData < 0 ] = 0
        sinData = np.where((np.isnan(sinData))|(sinData<0),0,sinData)
        sumDNs = np.sum(sinData)
        with open(recordfile,'a+') as f:
            ss = ntlname + ','+str(sumDNs)
            f.write(ss+'\n')
        del sinDs, sinBand, sinData, svnoValue
        gc.collect()

    def sum_of_lights_print(self,ntlpath,ntlname):
        sinDs, sinBand, sinData, svnoValue = rasterOp.getRasterData(ntlpath)
        # sinData[sinData < 0 ] = 0
        sinData = np.where((np.isnan(sinData))|(sinData<0),0,sinData)
        sumDNs = np.sum(sinData)
        print(ntlname,sumDNs)
        del sinDs, sinBand, sinData, svnoValue
        gc.collect()

    def raw_sntl_sumDNs(self):
        outpath = os.path.join(self.recorfDir,'raw_DNL_tnl.txt')
        for year in self.F10_yearLst:
            ntlname = 'F10'+str(year)
            ntlpath = os.path.join(self.raw_sntl_folder,ntlname+'.tif')
            self.sum_of_lights(ntlpath,ntlname,outpath)
        for year in self.F12_yearLst:
            ntlname = 'F12'+str(year)
            ntlpath = os.path.join(self.raw_sntl_folder,ntlname+'.tif')
            self.sum_of_lights(ntlpath,ntlname,outpath)
        for year in self.F14_yearLst:
            ntlname = 'F14'+str(year)
            ntlpath = os.path.join(self.raw_sntl_folder,ntlname+'.tif')
            self.sum_of_lights(ntlpath,ntlname,outpath)
        for year in self.F15_yearLst:
            ntlname = 'F15'+str(year)
            ntlpath = os.path.join(self.raw_sntl_folder,ntlname+'.tif')
            self.sum_of_lights(ntlpath,ntlname,outpath)
        for year in self.F16_yearLst:
            ntlname = 'F16'+str(year)
            ntlpath = os.path.join(self.raw_sntl_folder,ntlname+'.tif')
            self.sum_of_lights(ntlpath,ntlname,outpath)
        for year in self.F18_yearLst:
            ntlname = 'F18'+str(year)
            ntlpath = os.path.join(self.raw_sntl_folder,ntlname+'.tif')
            self.sum_of_lights(ntlpath,ntlname,outpath)

    def raw_gntl_sumDNs(self):
        outpath = os.path.join(self.recorfDir,'calib_GNTL_tnl.txt')
        for year in self.grc_yearLst:
            ntlname = str(year)
            print(year)
            # ntlname = year
            ntlpath = os.path.join(self.raw_gntl_folder,ntlname+'.tif')
            self.sum_of_lights(ntlpath,ntlname,outpath)

    def stepwise_sntl_sumDNs(self):
        outpath = os.path.join(self.recorfDir,'stepwise_DNL_tnl.txt')
        for year in self.F10_yearLst:
            ntlname = 'F10'+str(year)
            ntlpath = os.path.join(self.stepwise_sntl_folder,ntlname+'.tif')
            self.sum_of_lights(ntlpath,ntlname,outpath)
        for year in self.F12_yearLst:
            ntlname = 'F12'+str(year)
            ntlpath = os.path.join(self.stepwise_sntl_folder,ntlname+'.tif')
            self.sum_of_lights(ntlpath,ntlname,outpath)
        for year in self.F14_yearLst:
            ntlname = 'F14'+str(year)
            ntlpath = os.path.join(self.stepwise_sntl_folder,ntlname+'.tif')
            self.sum_of_lights(ntlpath,ntlname,outpath)
        for year in self.F15_yearLst:
            ntlname = 'F15'+str(year)
            ntlpath = os.path.join(self.stepwise_sntl_folder,ntlname+'.tif')
            self.sum_of_lights(ntlpath,ntlname,outpath)
        for year in self.F16_yearLst:
            ntlname = 'F16'+str(year)
            ntlpath = os.path.join(self.stepwise_sntl_folder,ntlname+'.tif')
            self.sum_of_lights(ntlpath,ntlname,outpath)
        for year in self.F18_yearLst:
            ntlname = 'F18'+str(year)
            ntlpath = os.path.join(self.stepwise_sntl_folder,ntlname+'.tif')
            self.sum_of_lights(ntlpath,ntlname,outpath)

    def saturated_sntl_sumDNs(self):
        for year in self.F10_yearLst:
            ntlname = 'F10'+str(year)
            ntlpath = os.path.join(self.saturated_sntl_folder,ntlname+'.tif')
            self.sum_of_lights(ntlpath,ntlname,self.recordfile)
        for year in self.F12_yearLst:
            ntlname = 'F12'+str(year)
            ntlpath = os.path.join(self.saturated_sntl_folder,ntlname+'.tif')
            self.sum_of_lights(ntlpath,ntlname,self.recordfile)
        for year in self.F14_yearLst:
            ntlname = 'F14'+str(year)
            ntlpath = os.path.join(self.saturated_sntl_folder,ntlname+'.tif')
            self.sum_of_lights(ntlpath,ntlname,self.recordfile)
        for year in self.F15_yearLst:
            ntlname = 'F15'+str(year)
            ntlpath = os.path.join(self.saturated_sntl_folder,ntlname+'.tif')
            self.sum_of_lights(ntlpath,ntlname,self.recordfile)
        for year in self.F16_yearLst:
            ntlname = 'F16'+str(year)
            ntlpath = os.path.join(self.saturated_sntl_folder,ntlname+'.tif')
            self.sum_of_lights(ntlpath,ntlname,self.recordfile)
        for year in self.F18_yearLst:
            ntlname = 'F18'+str(year)
            ntlpath = os.path.join(self.saturated_sntl_folder,ntlname+'.tif')
            self.sum_of_lights(ntlpath,ntlname,self.recordfile)

    def avg_saturated_sntl_sumDNs(self):
        outpath = os.path.join(self.recorfDir,'avg_stepwise_DNL_tnl.txt')
        for year in range(1992,2014):
            ntlname = str(year)
            ntlpath = os.path.join(self.avg_saturated_sntl_folder, ntlname + '.tif')
            self.sum_of_lights(ntlpath, ntlname, outpath)

    def harmonization_NTL_sumDNs(self):
        outpath = os.path.join(self.recorfDir, 'Harmonized_DN_NTL_tnl.txt')
        for year in range(2014, 2019):
            ntlname = 'Harmonized_DN_NTL_'+str(year)+'_simVIIRS'
            print(ntlname)
            ntlpath = os.path.join(self.harmonization_NTL_folder, ntlname + '.tif')
            self.sum_of_lights(ntlpath, ntlname, outpath)



result_Analysis = Result_Analysis()
# result_Analysis.raw_sntl_sumDNs()
# result_Analysis.raw_gntl_sumDNs()
# result_Analysis.stepwise_sntl_sumDNs()
# result_Analysis.avg_saturated_sntl_sumDNs()
# result_Analysis.harmonization_NTL_sumDNs()

# # ntlpath = r'D:\01data\00整理\02夜间灯光\dmsp\F15_20180101_20181231.global.stable_lights.avg_vis.tif'
# ntlname = 'F152018'
# # result_Analysis.sum_of_lights_print(ntlpath,ntlname)
# ntlpath=r'D:\01data\00整理\02夜间灯光\dmsp\calibrated_F152018.tif'
# result_Analysis.sum_of_lights_print(ntlpath,ntlname)

# #校正F152018
# a0 = -0.4597
# a1 = 1.714
# a2 = -0.0114
# oldpath = r'D:\01data\00整理\02夜间灯光\dmsp\F15_20180101_20181231.global.stable_lights.avg_vis.tif'
# newpath = r'D:\01data\00整理\02夜间灯光\dmsp\calibrated_F152018.tif'
# inDs,inBand, inData, vnoValue = rasterOp.getRasterData(oldpath)
# filterData = inData[inData>0]
# filterData1 = filterData*a1
# filterData2 = filterData*filterData*a2
# filterData = filterData1+filterData2+a0
# print(filterData.shape)
# del filterData1,filterData2
# gc.collect()
# outData = np.zeros(inData.shape)
# outData[inData>0] = filterData
# outData[inData==0] = a0
# rasterOp.outputResult(inDs, outData, newpath)
# del inDs, inBand, inData, vnoValue, outData
# gc.collect()












