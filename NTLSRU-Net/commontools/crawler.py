import os
import requests
from bs4 import BeautifulSoup
from urllib.request import  urlretrieve
import re

import tarfile
import rarfile
import zipfile
import  gzip

def un_tgz(outdir,filename):
    '''
    解压缩tgz文件
    :param filename: 压缩文件路径
    :return:
    '''
    tar = tarfile.open(filename)
    if os.path.isdir(outdir):
        pass
    else:
        os.mkdir(outdir)
    tar.extractall(outdir)
    tar.close()


def un_tgz_dmsp():
    outdir = r'E:\study\data\00整理\02夜间灯光\dmsp\raw\rar'
    outdir_temp = r'E:\study\data\00整理\02夜间灯光\dmsp\raw\rar'
    if os.path.isdir(outdir_temp):
        pass
    else:
        os.mkdir(outdir_temp)
    srcdir = r'F:\study\data\00整理\02夜间灯光\dmsp'
    # for root,dirs,files in os.walk(srcdir):
    #     for file in files:
    #         tarfilename = file.split('.')[0]
    #         tarfile_path = os.path.join(outdir,tarfilename+'.tif')
    #         if os.path.exists(tarfile_path):
    #             continue
    #         tarpath = os.path.join(srcdir,file)
    #         tar = tarfile.open(tarpath)
    #         names = tar.getnames()
    #         tname = names[0]
    #         for name in names:
    #             if str.endswith(name,'stable_lights.avg_vis.tif.gz'):
    #                 tname = name
    #         tar.extract(tname,outdir_temp)
    #         tar.close()
    for root, dirs, files in os.walk(outdir_temp):
        for file in files:
            gzpath = os.path.join(outdir_temp,file)
            f_gzip = gzip.GzipFile(gzpath,"rb")
            ungzfilename = file.split('.')[0]
            ungzpath = os.path.join(outdir,ungzfilename+'.tif')
            f_in = open(ungzpath, "wb")
            f_in.write(f_gzip.read())
            f_in.close()
            f_gzip.close()
    print('finish')

# un_tgz_dmsp()

def un_rar(filename):
    '''
    解压缩rar文件
    :param filename:
    :return:
    '''
    rar = rarfile.RarFile(filename)
    # 判断同名文件夹是否存在，若不存在则创建同名文件夹
    if os.path.isdir(os.path.splitext(filename)[0]):
        pass
    else:
        os.mkdir(os.path.splitext(filename)[0])
    rar.extractall(os.path.splitext(filename)[0])
def un_zip(filename):
    '''
    解压缩zip文件
    :param filename:
    :return:
    '''
    zip_file = zipfile.ZipFile(filename)
    # 判断同名文件夹是否存在，若不存在则创建同名文件夹
    if os.path.isdir(os.path.splitext(filename)[0]):
        pass
    else:
        os.mkdir(os.path.splitext(filename)[0])
    for names in zip_file.namelist():
        zip_file.extract(names, os.path.splitext(filename)[0])
    zip_file.close()

def download_gas_flaring_mask():
    '''
    下载gas flaring mask shapefiles
    :return:
    '''
    weburl = 'https://www.ngdc.noaa.gov/eog/interest/gas_flares_countries_shapefiles.html'
    outDir = r'D:\01data\00整理\02夜间灯光\GlobalGasFlaring'
    r = requests.get(weburl)
    html = r.text
    bf = BeautifulSoup(html)
    # get the data url
    texts = bf.find_all('li')
    for text in texts:
        dataurl = text.a.attrs.get('href')
        name = text.a.contents[0]
        filepath = os.path.join(outDir,name+'.tgz')
        if not os.path.exists(filepath):
            print(name)
            urlretrieve(dataurl, filepath)
            # un_tgz(outDir,filepath)

def download_Harmonization_NTLs():
    '''
    下载gas flaring mask shapefiles
    :return:
    '''
    weburl = 'https://figshare.com/articles/dataset/Harmonization_of_DMSP_and_VIIRS_nighttime_light_data_from_1992-2018_at_the_global_scale/9828827'
    outDir = r'D:\01data\00整理\02夜间灯光\harmonization NTL(1992-2018)'
    r = requests.get(weburl)
    html = r.text
    bf = BeautifulSoup(html)
    # get the data url
    texts = bf.find_all('div', class_=re.compile('h8Jn9'))
    for text in texts:
        href = text.find_all('a')[0]
        dataurl = href.attrs.get('href')
        filename = text.find_all('span',class_=re.compile('_3fcjv'))[0].attrs['title']
        filepath = os.path.join(outDir,filename)
        if not os.path.exists(filepath):
            print(filename)
            urlretrieve(dataurl, filepath)

def unzip_DMSP_NTLs():
    dir = r'E:\study\data\00整理\02夜间灯光\dmsp\raw\rar'
    yearInfoLst = []
    yearInfoLst.append((1992,'F10'))
    yearInfoLst.append((1993,'F10'))
    yearInfoLst.append((1994,'F10'))
    yearInfoLst.append((1994,'F12'))
    yearInfoLst.append((1995,'F12'))
    yearInfoLst.append((1996,'F12'))
    yearInfoLst.append((1997,'F12'))
    yearInfoLst.append((1998,'F12'))
    yearInfoLst.append((1999,'F12'))
    yearInfoLst.append((1997,'F14'))
    yearInfoLst.append((1998,'F14'))
    yearInfoLst.append((1999,'F14'))
    yearInfoLst.append((2000,'F14'))
    yearInfoLst.append((2001,'F14'))
    yearInfoLst.append((2002,'F14'))
    yearInfoLst.append((2003,'F14'))
    yearInfoLst.append((2000,'F15'))
    yearInfoLst.append((2001,'F15'))
    yearInfoLst.append((2002,'F15'))
    yearInfoLst.append((2003,'F15'))
    yearInfoLst.append((2004,'F15'))
    yearInfoLst.append((2005,'F15'))
    yearInfoLst.append((2006,'F15'))
    yearInfoLst.append((2007,'F15'))
    yearInfoLst.append((2004,'F16'))
    yearInfoLst.append((2005,'F16'))
    yearInfoLst.append((2006,'F16'))
    yearInfoLst.append((2007,'F16'))
    yearInfoLst.append((2008,'F16'))
    yearInfoLst.append((2009,'F16'))
    yearInfoLst.append((2010,'F18'))
    yearInfoLst.append((2011,'F18'))

    for yearInfo in yearInfoLst:
        year,id = yearInfo
        print(yearInfo)
        path = os.path.join(dir,id+str(year)+'.v4.tar')
        un_zip(path)

# unzip_DMSP_NTLs()