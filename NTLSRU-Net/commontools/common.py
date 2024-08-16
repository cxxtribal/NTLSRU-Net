import os
import dbfread
import pandas as pd
import shutil

'''
文件操作
'''
class FileOpr:
    #创建文件夹
    def createfolder(self,path):
        isExists = os.path.exists(path)
        if not isExists:
            os.makedirs(path)
#添加到文档
    def appendToFile(self,saveFilepath, listRow):
        with open(saveFilepath, 'a+') as f:
            for i in range(0, len(listRow)):
                f.write(listRow[i])
                f.write('\n')
#获取文本最后一行
    def get_last_line(self,filename):
        """
        get last line of a file
        :param filename: file name
        :return: last line or None for empty file
        """
        try:
            filesize = os.path.getsize(filename)
            if filesize == 0:
                return None
            else:
                with open(filename, 'rb') as fp:  # to use seek from end, must use mode 'rb'
                    offset = -8  # initialize offset
                    while -offset < filesize:  # offset cannot exceed file size
                        fp.seek(offset,
                                2)  # read # fp.seek(offset[, where])中where=0,1,2分别表示从文件头，当前指针位置，文件尾偏移，缺省值为0，但是如果要指定where=2，文件打开的方式必须是二进制打开，即使用’rb’模式
                        lines = fp.readlines()  # read from fp to eof
                        if len(lines) >= 2:  # if contains at least 2 lines
                            return lines[-1]  # then last line is totally included
                        else:
                            offset *= 2  # enlarge offset
                    fp.seek(0)
                    lines = fp.readlines()
                    return lines[-1]
        except FileNotFoundError:
            print(filename + ' not found!')
            return None
    def readDBF_asDataFrame(self,path):
        '''
        读取dfb并转为pandas.DataFrame
        :param path: dbf文件路径
        :return: pandas.DataFrame
        '''
        table = dbfread.DBF(path,load=True)
        df = pd.DataFrame(table)
        return df
    def getFilepaths_byExt(self,dirpath,ext):
        '''遍历目录下的文件，返回符合后缀要求的文件路径列表'''
        filelst = []
        for root, dirs, files in os.walk(dirpath):
            for file in files:
                # if file.endswith('.tif'):
                if file.endswith(ext):
                    filelst.append(os.path.join(root, file))
        return filelst
    def getFileNames_byExt(self,dirpath,ext):
        '''遍历目录下的文件，返回符合后缀要求的文件路径列表'''
        filenamelst = []
        for root, dirs, files in os.walk(dirpath):
            for file in files:
                # if file.endswith('.tif'):
                if file.endswith(ext):
                    filenamelst.append(file)
        return filenamelst
    #拷贝
    def copyTiffImg(self,sourcedir,targetdir,prefix):
        if not os.path.exists(targetdir):
            os.makedirs(targetdir)
        filelist = self.getFileNames_byExt(sourcedir,'.tif')
        for i in filelist:
            print(i)
            srcpath = os.path.join(sourcedir,i)
            tarpath = os.path.join(targetdir,prefix+i)
            shutil.copy(srcpath, tarpath)

    def copyContent_addPrefix(self,srcTxt,targetTxt,prefix):
        file_list = tuple(open(srcTxt, "r"))
        file_list = [id_.rstrip() for id_ in file_list]
        for i in range(len(file_list)):
            with open(targetTxt,'a+') as f:
                f.write(prefix+file_list[i])
                f.write('\n')

#拷贝
# dirInfolst = []
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
# fileOp = FileOpr()
# for dirInfo in dirInfolst:
#     print(dirInfo)
#     targetdir,sourcedir,prefix = dirInfo
#     fileOp.copyTiffImg(sourcedir, targetdir, prefix)


# #修改文本
# fileOp = FileOpr()
# srcTxt = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\04\train_07.txt'
# targetTxt = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\04\train_08.txt'
# prefix = '2013'
# fileOp.copyContent_addPrefix(srcTxt,targetTxt,prefix)
# srcTxt = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\04\test_07.txt'
# targetTxt = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\04\test_08.txt'
# prefix = '2013'
# fileOp.copyContent_addPrefix(srcTxt,targetTxt,prefix)
# prefix = '2012'
# fileOp.copyContent_addPrefix(srcTxt,targetTxt,prefix)
# srcTxt = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\04\valid_07.txt'
# targetTxt = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\04\valid_08.txt'
# prefix = '2013'
# fileOp.copyContent_addPrefix(srcTxt,targetTxt,prefix)
# prefix = '2012'
# fileOp.copyContent_addPrefix(srcTxt,targetTxt,prefix)

class CommonTools():
    def transformStrNum2Numer(self,strnum):
        '''
        数字字符串转化为数字。strnum有多个空格，不能直接用int强制类型转化。如'   5 639'
        :param strnum:
        :return:
        '''
        data = strnum.strip().split(' ')
        data = ''.join(data)
        return int(data)