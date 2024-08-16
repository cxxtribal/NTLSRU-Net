import torch
import os
import numpy as np
from scipy import ndimage
import cv2

from NPP_LIKE.prepareDataset import getRNTL_YearDict

def get_imgIds(sample_txt_path):
    file_list = tuple(open(sample_txt_path, "r"))
    file_list = [id_.rstrip() for id_ in file_list]
    return file_list
def read_imgdata(path,refData=None):
    inData = cv2.imread(path,flags=cv2.IMREAD_UNCHANGED)
    inData = np.where(np.isnan(inData), 0, inData)
    inData[(inData <= 0)] = 0
    if refData is not None:
        inData[refData==0]=0
    return inData
def normlized_otherdata(data, stat_dict):
    if stat_dict['normlizedtype'] == 'minMax':
        return (data - stat_dict['min']) / (stat_dict['max'] - stat_dict['min'])
    elif stat_dict['normlizedtype'] == 'meanStd':
        return (data - stat_dict['mean']) / stat_dict['std']
    else:
        return data
def viirs_maxClip(viirsdata,viirs_maxclip):
    # 最大值裁剪
    maxValue = viirs_maxclip
    num = len(viirsdata[viirsdata > maxValue])
    if num >0:
        label_inData1 = np.where(viirsdata > maxValue, 0, viirsdata)
        label_inData1 = ndimage.maximum_filter(label_inData1, size=3)
        label_inData1 = np.where(viirsdata > maxValue, label_inData1, viirsdata)
        return label_inData1
    else:
        return viirsdata
def viirs_preprocess(viirsdata,viirs_maxclip,label_use_ln,viirs_log_add_smallValue,stat_dict):
    # 最大值裁剪
    label_inData1 = viirs_maxClip(viirsdata, viirs_maxclip)
    # 归一化
    if label_use_ln & (stat_dict['normlizedtype'] == 'minMax'):
        label_inData2 = np.log(label_inData1 + viirs_log_add_smallValue)  # +1  防止出现警告：divide by zero encountered in log
        label_inData2 = (label_inData2 - np.log(viirs_log_add_smallValue)) / \
                        (np.log(stat_dict['max'] + viirs_log_add_smallValue) - np.log(viirs_log_add_smallValue))
    else:
        label_inData2 = normlized_otherdata(label_inData1,stat_dict)
    return label_inData2
def getOtherDatas_path(otherdatapath_dict,ori_dmsp,refData=None,is_arid_one_hot=False,runMode='valid'):
    out = None
    for key in otherdatapath_dict:
        if runMode == 'valid':
            input_inData = read_imgdata(otherdatapath_dict[key]["path"])
        else:
            input_inData = read_imgdata(otherdatapath_dict[key]["path"], refData)
        if key in ['RNTL', 'CfCvg']:
            input_inData[ori_dmsp == 0] = 0
        elif key in ['NDVI','AVHRR']:
            input_inData[refData == 0] = 0
        if key == 'Water_Aridity':
            if is_arid_one_hot:
                encodeData = None
                for i in [1, 2, 3, 4, 5]:
                    tempdata = np.zeros_like(input_inData)
                    tempdata[input_inData == i] = 1
                    tempdata = np.expand_dims(tempdata, 0)
                    if encodeData is None:
                        encodeData = tempdata
                    else:
                        encodeData = np.concatenate((encodeData, tempdata), 0)
                input_inData = encodeData
            else:
                input_inData = np.expand_dims(input_inData, 0)
        else:
            input_inData = normlized_otherdata(input_inData, otherdatapath_dict[key])
            input_inData = np.expand_dims(input_inData, 0)
        if out is None:
            out = input_inData
        else:
            out = np.concatenate((out,input_inData),0)
    return out
def load_data_path(inputdata_path,label_path,water_path,otherdatapath_dict,runMode,config):
    out_dict = {}
    # Load an image
    # Load an image (水体设为0)
    water_inData = read_imgdata(water_path)
    if runMode == 'valid':
        input_inData = read_imgdata(inputdata_path)
        label_inData = read_imgdata(label_path)
    else:
        input_inData = read_imgdata(inputdata_path,water_inData)
        label_inData = read_imgdata(label_path,water_inData)
    # get data
    viirsdata_ori = label_inData
    out_dict['viirsdata_ori'] = np.expand_dims(viirsdata_ori, 0).astype(np.float32)
    dmspdata_ori_inter = input_inData
    out_dict['dmspdata_ori_inter'] = np.expand_dims(dmspdata_ori_inter, 0).astype(np.float32)
    otherdatas = getOtherDatas_path(otherdatapath_dict, dmspdata_ori_inter,water_inData,config.is_arid_one_hot,runMode)
    # get input data
    input_inData = normlized_otherdata(input_inData,config.dmsp_stat_dict)
    input_inData = np.expand_dims(input_inData, 0).astype(np.float32)
    if otherdatas is not None:
        otherdatas = otherdatas.astype(np.float32)
        input_inData = np.concatenate((input_inData, otherdatas), 0)
    out_dict['input_inData'] = input_inData

    # to torch
    out_dict['input_inData'] = torch.from_numpy(out_dict['input_inData'])
    out_dict['dmspdata_ori_inter'] = torch.from_numpy(out_dict['dmspdata_ori_inter'])
    out_dict['viirsdata_ori'] = torch.from_numpy(out_dict['viirsdata_ori'])

    if runMode != 'valid':
        viirs_maxclip = config.viirs_maxClip
        label_use_ln = config.label_use_ln
        viirs_log_add_smallValue = config.viirs_log_add_smallValue
        label_inData = viirs_preprocess(label_inData,viirs_maxclip,label_use_ln,viirs_log_add_smallValue,config.viirs_stat_dict)
        out_dict['label_inData'] = np.expand_dims(label_inData, 0).astype(np.float32)
        out_dict['label_inData'] = torch.from_numpy(out_dict['label_inData'])
    return out_dict
def addBatchDimension(out_dict,runMode='valid'):
    '''
    对于c,h,w的数据，增加batch维度
    :param out_dict:
    :return:
    '''
    dmspdata_ori_inter = out_dict['dmspdata_ori_inter']
    viirsdata_ori = out_dict['viirsdata_ori']
    x = out_dict['input_inData']

    #img 扩展到3维，C,H,W
    x = x.unsqueeze(0)
    dmspdata_ori_inter = dmspdata_ori_inter.unsqueeze(0)
    viirsdata_ori = viirsdata_ori.unsqueeze(0)

    #out
    out_dict = {}
    out_dict['dmspdata_ori_inter'] = dmspdata_ori_inter
    out_dict['viirsdata_ori'] = viirsdata_ori
    out_dict['input_inData'] = x

    if runMode != 'valid':
        label = out_dict['label_inData']
        label = label.unsqueeze(0)
        out_dict['label_inData'] = label

    return out_dict

class Dataset_oriDMSP(torch.utils.data.Dataset):
    def __init__(self, config,sample_txt_path, runMode,label_dir = "",inputdata_dir = "",waterdata_dir="",otherdata_dict=None,data_dir = "",is_multi_years_combined = False):
        '''
        :param sample_txt_path: 记录样本编号的txt,一个样本编号一行，样本编号就是对应图像的图像名称（不带文件后缀）
        :param label_dir: label image 所在目录路径
        :param inputdata_dir: 输入影像（待训练、待预测）所在目录路径
        :param runMode: 标记输入影像的类型，决定是否需要标签数据。true 表示输入影像是待预测影像，目的是用于预测，不需要标签数据。false 表示输入影像是待训练影像，目的是训练模型，需要标签数据。
        :param otherdata_dict:其他辅助数据 格式为 {数据名称:{"path":"","normlizedtype":归一化方式,"max":最大值,"min":最小值,"mean":均值,"std"：标准差}}其中，max,min,mean,std用于数据标准化，用到什么提供什么，不必全部都有
        '''
        super(Dataset_oriDMSP, self).__init__()
        if label_dir.strip()=='':
            self.label_dir = config.label_dir
        else:
            self.label_dir = label_dir
        if inputdata_dir.strip()=='':
            self.inputdata_dir = config.inputdata_dir
        else:
            self.inputdata_dir = inputdata_dir
        if waterdata_dir.strip()=='':
            self.waterdata_dir = config.waterdata_dir
        else:
            self.waterdata_dir = waterdata_dir
        if otherdata_dict is not None:
            self.otherdata_dict = otherdata_dict
        else:
            self.otherdata_dict = config.otherdata_dict
        if data_dir.strip()=='':
            self.data_dir = config.data_dir
        else:
            self.data_dir = data_dir
        self.is_multi_years_combined = is_multi_years_combined
        self.sample_txt_path = sample_txt_path
        self.runMode = runMode
        self.files = []
        self._set_files()
        self.dmsp_stat_dict = config.dmsp_stat_dict
        self.viirs_stat_dict = config.viirs_stat_dict
        self.inputdata_maxDN = config.dmsp_stat_dict['max']
        self.inputdata_minDN = config.dmsp_stat_dict['min']
        self.label_maxDN = config.viirs_stat_dict['max']
        self.label_minDN = config.viirs_stat_dict['min']
        self.label_use_ln = config.label_use_ln
        self.viirs_log_add_smallValue = config.viirs_log_add_smallValue
        self.viirs_maxclip = config.viirs_maxClip
        self.is_arid_one_hot = config.is_arid_one_hot

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        image_id = self.files[index]
        out_dict = self._load_data(image_id)
        out_dict['index'] = torch.IntTensor([index])
        return out_dict

    def _set_files(self):
        file_list = get_imgIds(self.sample_txt_path)
        self.files = file_list

    def _load_data(self, image_id):
        out_dict = {}
        # Set paths
        if self.is_multi_years_combined:
            year,imgId = image_id[0:4],image_id[4:]
            inputdata_path = os.path.join(self.data_dir,year+'_oriDNL' ,imgId + ".tif")
            label_path = os.path.join(self.data_dir,year+'_VNL' ,imgId + ".tif")
            water_path = os.path.join(self.waterdata_dir,imgId+'.tif')
        else:
            inputdata_path = os.path.join(self.inputdata_dir, image_id + ".tif")
            label_path = os.path.join(self.label_dir, image_id + ".tif")
            water_path = os.path.join(self.waterdata_dir,image_id+'.tif')
        # Load an image (水体设为0)
        water_inData = self.read_raster_data(water_path)
        if self.runMode == 'valid':
            input_inData = self.read_raster_data(inputdata_path)
            label_inData = self.read_raster_data(label_path)
        else:
            input_inData = self.read_raster_data(inputdata_path,water_inData)
            label_inData = self.read_raster_data(label_path,water_inData)

        #get data
        viirsdata_ori = label_inData
        out_dict['viirsdata_ori'] = np.expand_dims(viirsdata_ori,0).astype(np.float32)
        dmspdata_ori_inter =  input_inData
        out_dict['dmspdata_ori_inter'] =  np.expand_dims(dmspdata_ori_inter,0).astype(np.float32)
        otherdatas = self.getOtherDatas(image_id, dmspdata_ori_inter,water_inData)
        # get input data
        input_inData = self.dmsp_input_preprocess(input_inData)
        input_inData = np.expand_dims(input_inData, 0).astype(np.float32)
        if otherdatas is not None:
            otherdatas = otherdatas.astype(np.float32)
            input_inData = np.concatenate((input_inData,otherdatas),0)
        out_dict['input_inData'] = input_inData

        #to torch
        out_dict['input_inData'] = torch.from_numpy(out_dict['input_inData'])
        out_dict['dmspdata_ori_inter']  = torch.from_numpy(out_dict['dmspdata_ori_inter'] )
        out_dict['viirsdata_ori'] = torch.from_numpy( out_dict['viirsdata_ori'])

        if self.runMode != 'valid':
            label_inData = self.viirs_input_preprocess(label_inData)
            out_dict['label_inData'] = np.expand_dims(label_inData,0).astype(np.float32)
            out_dict['label_inData'] = torch.from_numpy(out_dict['label_inData'])
        return out_dict

    def read_raster_data(self,path,refData=None):
        return read_imgdata(path,refData)

    def dmsp_input_preprocess(self, dmspdata):
        return normlized_otherdata(dmspdata,self.dmsp_stat_dict)

    def viirs_input_preprocess(self,viirsdata):
        return viirs_preprocess(viirsdata,self.viirs_maxclip,self.label_use_ln,self.viirs_log_add_smallValue,self.viirs_stat_dict)

    def getOtherDatas(self,image_id,ori_dmsp,refData=None):
        out = None
        for key in self.otherdata_dict:
            if self.is_multi_years_combined:
                year, imgId = image_id[0:4], image_id[4:]
                if key == 'Water_Aridity':
                    path = os.path.join(self.waterdata_dir, imgId + ".tif")
                    input_inData = self.read_raster_data(path)
                elif key == 'RNTL':
                    year1 = str(getRNTL_YearDict[int(year)])
                    path = os.path.join(self.data_dir, year1 + '_' + key, imgId + ".tif")
                    if self.runMode == 'valid':
                        input_inData = self.read_raster_data(path)
                    else:
                        input_inData = self.read_raster_data(path, refData)
                else:
                    path = os.path.join(self.data_dir, year + '_' + key, imgId + ".tif")
                    if key == 'NDVI':
                        input_inData = self.read_raster_data(path, refData)
                    else:
                        if self.runMode == 'valid':
                            input_inData = self.read_raster_data(path)
                        else:
                            input_inData = self.read_raster_data(path, refData)
            else:
                path = os.path.join(self.otherdata_dict[key]["path"], image_id + ".tif")
                input_inData = self.read_raster_data(path,refData)
            if key in ['RNTL','CfCvg']:
                input_inData[ori_dmsp==0]=0
            if key == 'Water_Aridity':
                if self.is_arid_one_hot:
                    encodeData = None
                    for i in [1,2,3,4,5]:
                        tempdata = np.zeros_like(input_inData)
                        tempdata[input_inData==i] = 1
                        tempdata = np.expand_dims(tempdata,0)
                        if encodeData is None:
                            encodeData = tempdata
                        else:
                            encodeData = np.concatenate((encodeData, tempdata), 0)
                    input_inData = encodeData
                else:
                    input_inData = np.expand_dims(input_inData, 0)
            else:
                input_inData = normlized_otherdata(input_inData, self.otherdata_dict[key])
                input_inData = np.expand_dims(input_inData, 0)
            if out is None:
                out = input_inData
            else:
                out = np.concatenate((out,input_inData),0)
        return out


