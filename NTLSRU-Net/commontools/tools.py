import numpy as np
import csv


def to_csv(saveFilepath,listRow):
    with open(saveFilepath,"a+") as f:
        cw = csv.writer(f,lineterminator='\n',delimiter=',')
        for s in listRow:
            cw.writerow(s)
            
def filesave(saveFilepath,listRow):
    with open(saveFilepath,'a+') as f:
        for s in listRow:
            f.write(s)
            f.write('\n')

def normalize(arr):
    minvalue = np.nanmin(arr)
    maxvalue = np.nanmax(arr)
    arr = (arr-minvalue)/(maxvalue-minvalue)*100
    return arr
    
AttrColNameLst = ['ntl','ndvi','lst']
ResultColNameLst = ['growthtime','result','IsUrban','result_gro']
#SpatialColNameLst = ['mean','median','var','std','sum','largerThenMean','inOneStd','ntlIndex','ptp','max','min','bcxs']
SpatialColNameLst = ['mean','median','std','sum','ntlIndex','ptp','max','min']

'''
城市列表，对应文件名
'''
def getCityLst():
    cityLst = []
    cityLst.append(['01','beijing'])
    cityLst.append(['02','huhhot'])
    cityLst.append(['03','jinhua'])
    cityLst.append(['04','shenyang'])
    cityLst.append(['05','jilin'])
    cityLst.append(['06','haerbin'])
    cityLst.append(['07','xian'])
    cityLst.append(['08','wulumuqi'])
    cityLst.append(['09','lanzhou'])
    cityLst.append(['10','tianjin'])
    cityLst.append(['11','shijiazhuang'])
    cityLst.append(['12','qingdao'])
    cityLst.append(['13','shanghai'])
    cityLst.append(['14','nanjing'])
    cityLst.append(['15','hangzhou'])
    cityLst.append(['16','ningbo'])
    cityLst.append(['17','jiaxing'])
    cityLst.append(['18','yiwu'])
    cityLst.append(['19','wenling'])
    cityLst.append(['20','wuhan'])
    cityLst.append(['21','zhengzhou'])
    cityLst.append(['22','guangzhou'])
    cityLst.append(['23','shenzhou'])
    cityLst.append(['24','xiamen'])
    cityLst.append(['25','lasa'])
    cityLst.append(['26','chongqing'])
    cityLst.append(['27','chengdu'])
    cityLst.append(['28','kunming'])
    return cityLst

def getProvinceLst():
    proviceLst = []
    proviceLst.append("fujian")
    proviceLst.append("gansu")
    proviceLst.append("guangdong")
    proviceLst.append("hebei")
    proviceLst.append("heilongjiang")
    proviceLst.append("henan")
    proviceLst.append("hubei")
    proviceLst.append("jiangsu")
    proviceLst.append("jilin")
    proviceLst.append("liaoning")
    proviceLst.append("menggu")
    proviceLst.append("shandong")
    proviceLst.append("shanxi")
    proviceLst.append("sichuang")
    proviceLst.append("xinjiang")
    proviceLst.append("xizang")
    proviceLst.append("yunnan")
    proviceLst.append("zhejiang")
    return proviceLst

def getRegionLst():
    regionLst=[]
    regionLst.append("VIIRS_EN")
    regionLst.append("VIIRS_HC")
    regionLst.append("VIIRS_HE")
    regionLst.append("VIIRS_HN")
    regionLst.append("VIIRS_HS")
    regionLst.append("VIIRS_WN")
    regionLst.append("VIIRS_WS")
    return regionLst

'''统计列表
'''
def getStatisticTypeList():
    statisticTypeLst =[]
    statisticTypeLst.append('mean')
    statisticTypeLst.append('median')
#    statisticTypeLst.append('var')
    statisticTypeLst.append('std')
#    statisticTypeLst.append('lcVar')
    statisticTypeLst.append('sum')
#    statisticTypeLst.append('largerThenMean')
#    statisticTypeLst.append('inOneStd')
    statisticTypeLst.append('ntlIndex')
    statisticTypeLst.append('ptp')
    statisticTypeLst.append('max')
    statisticTypeLst.append('min')
#    statisticTypeLst.append('bcxs')
#    statisticTypeLst.append('gmi')
    return statisticTypeLst
    

'''
数组统计
calArray:数组
statisticType:统计类型，字符串
返回：统计值，若不存在，则返回np.nan
'''    
def calstatistic(calArray,statisticTypeLst):
    if len(calArray) == 0:
        return np.nan   
    #个数
    calArray=calArray[calArray==calArray]#去除np.nan
    num = len(calArray)
    mean = np.nanmean(calArray)
    std = np.nanstd(calArray)
    var = np.nanvar(calArray)   
    #大于均值像元个数占比
    n1=0
    n2=0
    s2=0
    for t in range(0,num):
        if (calArray[t]>=mean):
            n1=n1+1
        if (calArray[t]<=(mean + std)) and (calArray[t] >= (mean-std)):
            n2=n2+1
            s2 = s2+calArray[t]
    result_stats = []
    for statisticType in statisticTypeLst:
        #统计
        if statisticType == 'mean':
            #均值       
            result_stats.append(mean)
        elif statisticType == 'median':
            #中位数
            result_stats.append(np.nanmedian(calArray))
        elif statisticType == 'var':
             #方差
             result_stats.append(var)
        elif statisticType == 'std':
             #标准差
             result_stats.append(std)
        elif statisticType == 'lcVar':
             #离差平方和
             result_stats.append(var*num)
        elif statisticType == 'sum':
            result_stats.append(s2)
        elif statisticType == 'largerThenMean':
             #大于均值的个数比
            result_stats.append(n1/num)
        elif statisticType == 'inOneStd':
             #1个标准差内的像元个数比
            result_stats.append(n2/num)
        elif statisticType == 'ntlIndex':
                    #平均灯光指数
            if n2 == 0:
                value = 0
            else:
                value = s2/n2
            result_stats.append(value)
        elif statisticType == 'ptp':
            #极差
            result_stats.append(np.ptp(calArray))
        elif statisticType == 'max':
             #最大值
             result_stats.append(np.nanmax(calArray))
        elif statisticType == 'min':
            #最小值
            result_stats.append(np.nanmin(calArray))
        elif statisticType == 'bcxs':
            #变差系数
            if std == 0:
                 value = 0
            else:
                value = mean/std
            result_stats.append(value)
        else:
            result_stats.append(np.nan)
    return result_stats


def getfeaturegroup(origLst,resultLst):
    if len(origLst)==0:
            return resultLst
    if(origLst not in resultLst):
        resultLst.append(origLst)
#    print(len(origLst))
    for i in range(0,len(origLst)):
        newLst = origLst.copy()
        newLst.remove(newLst[i])
        newLst.sort()
        getfeaturegroup(newLst,resultLst)
    return resultLst
  

#自适应阈值分割   
#orign_data为  pandas.core.series.Series格式 
#函数中指定了循环步长为0.01
def otsu_threshold(orign_data):
    minValue = min(orign_data) #最小值
    maxValue = max(orign_data) #最大值
    step=0.01
    threshold=minValue+step
    s_max=[threshold,0]
    #循环计算不同阈值下的类间方差，得到最大的类间方差对应的阈值
    while threshold < maxValue:
        #小于阈值的个数
        w_0 = len(orign_data[orign_data<threshold])
        #大于阈值的个数
        w_1 = len(orign_data[orign_data>=threshold])
        # 得到阈值下所有像素的平均灰度
        u_0 = orign_data[orign_data<threshold].sum()/w_0 if w_0 > 0 else 0
        # 得到阈值上所有像素的平均灰度
        u_1 = orign_data[orign_data>=threshold].sum()/w_1 if w_1 > 0 else 0
        
#        # 总平均灰度
#        u = w_0 * u_0 + w_1 * u_1
#        # 类间方差
#        g = w_0 * (u_0 - u) * (u_0 - u) + w_1 * (u_1 - u) * (u_1 - u)

#        #类间方差等价公式
        g = w_0 * w_1 * (u_0 * u_1) * (u_0 * u_1)
        # 取最大的
        if g > s_max[1]:
            s_max = (threshold, g)
            
        threshold = threshold+step
    return s_max[0]

#自适应阈值分割   
#orign_data为  pandas.core.series.Series格式 
#数据值可穷尽
def otsu_threshold_enum(orign_data):
    values =  orign_data.value_counts().index.tolist()   
    values = sorted(values)
    s_max=[-1,-1]
    #循环计算不同阈值下的类间方差，得到最大的类间方差对应的阈值
    for i in range(1,len(values)):
        threshold = values[i]
        #小于阈值的个数
        w_0 = len(orign_data[orign_data<threshold])
        #大于阈值的个数
        w_1 = len(orign_data[orign_data>=threshold])
        # 得到阈值下所有像素的平均灰度
        u_0 = orign_data[orign_data<threshold].sum()/w_0 if w_0 > 0 else 0
        # 得到阈值上所有像素的平均灰度
        u_1 = orign_data[orign_data>=threshold].sum()/w_1 if w_1 > 0 else 0
        
#        # 总平均灰度
#        u = w_0 * u_0 + w_1 * u_1
#        # 类间方差
#        g = w_0 * (u_0 - u) * (u_0 - u) + w_1 * (u_1 - u) * (u_1 - u)

#        #类间方差等价公式
        g = w_0 * w_1 * (u_0 * u_1) * (u_0 * u_1)
        # 取最大的
        if g > s_max[1]:
            s_max = (threshold, g)
    return s_max[0]