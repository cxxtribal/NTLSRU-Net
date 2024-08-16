import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.nn import functional as F

import numpy as np
import pickle
import argparse
import os
import time
import sys

sys.path.append("/home/zju/cxx/NTL_timeserise/code/ExtractedUrbanFromNTL/commontools")
sys.path.append("/home/zju/cxx/NTL_timeserise/code/ExtractedUrbanFromNTL/NPP_LIKE")
sys.path.append("/home/zju/cxx/NTL_timeserise/code/ExtractedUrbanFromNTL")
sys.path.append("/home/zju/cxx/NTL_timeserise/code/ExtractedUrbanFromNTL/src")
from NPP_LIKE.net.DRLN import DNLSRNet
from NPP_LIKE.net.Unet import U_Net

Training_Quality_Indices_Lst = ['EPOCH','RMSE', 'MRE','TNL']
Valid_Quality_Key = "valid"

"""parsing and configuration"""
def parse_args():
    desc = "PyTorch implementation of DMSP convert to NPP_like SR collections"
    parser = argparse.ArgumentParser(description=desc)

    # Hardware specifications
    parser.add_argument('--num_threads', type=int, default=0, help='number of threads for data loader to use')

    parser.add_argument('--subcommand', type=str, default='train',choices=['train', 'test','valid'], help='The commond for model')
    #model name
    parser.add_argument('--model_name', type=str, default='DNLSRNet', help='The type of model')
    parser.add_argument('--model_name_appendix', type=str, default='01', help='The appendix name for the current training output directory')
    parser.add_argument('--dataset_type', type=str, default='ndvi',choices=['ndvi', 'ndvi_gntl','sndvi'], help='The appendix name for the current training output directory')
    parser.add_argument('--in_channels', type=int, default=2, help='channel number of net input')
    parser.add_argument('--base_filter', type=int, default=16, help='base out channels in process')
    parser.add_argument('--attentiontype',type=str, default='LCA', help='attention type')
    parser.add_argument('--isNDVIcoefficient',type=bool, default=True, help='ndvi feature mul dnl as cofficent')
    parser.add_argument('--upsample',type=str, default='ps',choices=['deconv', 'ps','rnc'], help='upsample type')
    parser.add_argument('--isLR',type=bool, default=False, help='if use LR img to net')
    parser.add_argument('--nums_msc', type=int, default=2,  help='msc block number')
    parser.add_argument('--nums_DRAblk', type=int, default=2, help='dense residual attention block number')

    #Training specifications
    parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
    parser.add_argument('--num_epochs', type=int, default=4000, help='The number of epochs to run')
    parser.add_argument('--batch_size', type=int, default=5, help='training batch size')
    parser.add_argument('--test_batch_size', type=int, default=5, help='testing batch size')
    parser.add_argument('--save_epochs', type=int, default=1, help='interval of trained model save')
    parser.add_argument('--save_plt_epochs', type=int, default=1, help='interval of loss plt save')
    parser.add_argument('--sample_interval', type=int, default=1,help='interval between sampling of images from generators')

    #optimizer
    parser.add_argument('--optim_type', type=str, default='Adam', choices=('Adam','SGD','Momentum','RMSProp'),help='optimizer method')
    parser.add_argument("--learning_rate", type=float, default=1e-4,help="learning rate, default set to 1e-3")
    parser.add_argument("--epochs_upLr", type=str, default="5000,8000", help="epoch list to update learning rate")
    parser.add_argument('--b1', type=float, default=0.9, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
    parser.add_argument('--momentum', type=float, default=0.8, help='Momentum:momentum')
    parser.add_argument('--gpu_mode', type=bool, default=False)

    # Data specifications；数据目录以代码为准，config txt中记录的是上次训练使用的数据路径。
    parser.add_argument('--root_dir', type=str, default=r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN')#'/home/zju/cxx/NTL_timeserise/data/SRCNN'
    parser.add_argument('--data_dir', type=str, default=r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img04')#'/home/zju/cxx/NTL_timeserise/data/SRCNN/data/img'
    parser.add_argument('--save_dir_name', type=str, default='net_02_1210', help='Directory name to save the results')
    parser.add_argument('--config_dir_name', type=str, default='config', help='Directory name to save the config')
    parser.add_argument('--config_name', type=str, default='config', help='File name to save the config')
    #smaple txt
    parser.add_argument('--train_sample_txt_path', type=str, default=r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\04\train_07.txt', help='path of train sample txt')
    parser.add_argument('--test_sample_txt_path', type=str,default=r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\04\test_07.txt',help='path of test sample txt')
    parser.add_argument('--valid_sample_dir', type=str,default=r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\sample\01\04',help='path of test sample txt')
    parser.add_argument('--valid_sample_txt_lst', type=str,default=r'valid_07',help='path of test sample txt')
    parser.add_argument('--waterdata_dir', type=str,default=r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img04\Water_Aridity',help='water directory name for train')
    parser.add_argument('--valid_waterdata_dir', type=str,default=r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img04\Water_Aridity',help='water directory name for valid')
    #img dir
    parser.add_argument('--is_multi_years_combined', type=bool, default=True, help='train dataset is multi years')
    parser.add_argument('--label_dir_name', type=str, default='2012_VNL', help='Directory name of viirs')
    parser.add_argument('--inputdata_dir_name', type=str, default='2012_oriDNL', help='Directory name of dmsp')
    parser.add_argument('--ndvidata_dir_name', type=str, default='2012_NDVI', help='Directory name of ndvi')
    parser.add_argument('--rntl_dir_name', type=str, default='2010_RNTL', help='Directory name of ndvi')
    parser.add_argument('--chendata_dir_name', type=str, default='2012_Chen', help='Directory name of chen NTL')
    parser.add_argument('--valid_label_dir_name', type=str, default='2012_VNL', help='valid Directory name of viirs')
    parser.add_argument('--valid_inputdata_dir_name', type=str, default='2012_oriDNL', help='valid Directory name of dmsp')
    parser.add_argument('--valid_ndvidata_dir_name', type=str, default='2012_NDVI', help='valid Directory name of ndvi')

    # data preprocess params
    parser.add_argument('--is_arid_one_hot', type=bool, default=False, help='is encode arid as one hot')
    parser.add_argument('--viirs_log_add_smallValue', type=float, default=1, help='add the small value to label data for log operatorion to prevent errors')
    parser.add_argument('--viirs_maxClip', type=float, default=1000, help='the value for viirs max clip')
    parser.add_argument('--inputdata_maxDN',type=float,default=63,help='the max value of dmsp data')
    parser.add_argument('--inputdata_minDN', type=float, default=0, help='the max value of dmsp data')
    parser.add_argument('--inputdata_meanDN',type=float,default=14.45,help='the max value of dmsp data')
    parser.add_argument('--inputdata_std', type=float, default=17.65, help='the max value of dmsp data')
    parser.add_argument('--rntl_maxDN', type=float, default=8000, help='the max value of dmsp data')
    parser.add_argument('--label_maxDN', type=float, default=1000, help='the max value of viirs data')
    parser.add_argument('--label_minDN', type=float, default=0, help='the max value of viirs data')
    parser.add_argument('--label_meanDN', type=float, default=1.12, help='the max value of viirs data')
    parser.add_argument('--label_std', type=float, default=4.55, help='the max value of viirs data')
    parser.add_argument('--label_use_ln', type=bool, default=True, help='ln labeldata')
    parser.add_argument('--indata_normlizeType', type=str, default='minMax', help='dsmp normalized type')
    parser.add_argument('--labeldata_normlizeType', type=str, default='minMax', help='viirs normalized type')
    #out data process
    parser.add_argument('--outdata_mask', type=str, default='0', help='mask data for loss: 0 label_inData; 1 dmspdata_ori_inter; 2 viirsdata_ori')
    parser.add_argument('--noMask', type=bool, default=True, help='loss computed with all pixels: masktype 2')
    parser.add_argument('--gt0Mask', type=bool, default=False, help='loss computed with pixels which larger than 0: masktype 0 label_inData >0')
    parser.add_argument('--urbanMask', type=bool, default=False, help='loss computed with pixels which urban pixels: masktype 1 dmspdata_ori_inter in range')
    parser.add_argument('--weight_urbFocus_loss', type=float, default=0.2, help='weight for Urban Focus Loss')
    parser.add_argument('--urbFocus_loss_urban_min_DN', type=float, default=20, help='UrbanFocusLoss urban_min_DN')
    parser.add_argument('--urbFocus_loss_urban_max_DN', type=float, default=63, help='UrbanFocusLoss urban_min_DN')
    parser.add_argument('--urbFocus_loss_loss_Lp_norm', type=str,default='L2', choices=["L2", "L1"], help='UrbanFocusLoss loss_Lp_norm')
    parser.add_argument('--weight_dimArea', type=float, default=10, help='Loss weight for dim Area')
    parser.add_argument('--isClip_to_lossCal', type=bool, default=False, help='True means clip image to calculate loss')
    parser.add_argument('--PaddingClipped_to_lossCal', type=int, default=20, help='set padding for clip image')


    #weight init
    parser.add_argument("--weight_init_name", type=str, default='kaming',help="weight init method")
    parser.add_argument("--weight_init_normal_mean", type=float, default=0.0, help="mean for normal weight init")
    parser.add_argument("--weight_init_normal_std", type=float, default=0.02, help="std for normal weight init")

    # loss
    parser.add_argument('--loss_Lp_norm', type=str, default='L2', choices=["L2", "L1"], help='loss Lp norm')
    parser.add_argument('--use_gan', type=bool, default=False, help='GAN')
    parser.add_argument('--perceptual_loss', type=bool, default=False, help='true means use perceptual_loss')
    parser.add_argument('--gradient_loss', type=bool, default=False, help='true means use gradient_loss')
    parser.add_argument('--isUrbanMask_gradient_loss', type=bool, default=False, help='true means use gradient_loss')
    parser.add_argument('--weight_gradient', type=float, default=1e-4, help='Loss weight for sstv loss')
    parser.add_argument('--weight_content', type=float, default=6e-3, help='Loss weight for content loss')
    parser.add_argument('--weight_gan', type=float, default=1e-3, help='Loss weight for gan loss')
    parser.add_argument('--clip_value', type=float, default=0.01, help='lower and upper clip value for disc. weights')

    #ae
    parser.add_argument('--ae_out_channel_nums', type=str, default='16,32,64,128,256', help='out channel num lst for encoder and decoder')
    parser.add_argument('--ae_encoder_act', type=str, default='relu',help='act for encoder')
    parser.add_argument('--ae_encoder_norm', type=str, default='batch', help='norm for encoder')
    parser.add_argument('--ae_decoder_body_act', type=str, default='relu',help='act for decoder_body')
    parser.add_argument('--ae_decoder_body_norm', type=str, default=None, help='norm for decoder_body')
    parser.add_argument('--ae_decoder_tail_act', type=str, default=None,help='act for decoder_tail')
    parser.add_argument('--ae_decoder_tail_norm', type=str, default=None, help='norm for decoder_tail')
    #drcn
    parser.add_argument('--num_recursions', type=int, default=16, help='num recursions')
    #SRGAN  edsr
    parser.add_argument('--num_residuals', type=int, default=16, help='the number of resnet block')
    #unet
    parser.add_argument('--unet_encoder_act', type=str, default='relu',help='act for encoder')
    parser.add_argument('--unet_encoder_norm', type=str, default='batch', help='norm for encoder')
    parser.add_argument('--unet_decoder_act', type=str, default='relu',help='act for decoder_body')
    parser.add_argument('--unet_decoder_norm', type=str, default=None, help='norm for decoder_body')
    parser.add_argument('--unet_tail_act', type=str, default=None,help='act for decoder_tail')
    parser.add_argument('--unet_tail_norm', type=str, default=None, help='norm for decoder_tail')

    return check_args(parser.parse_args())

"""checking arguments"""
def check_args(config):
    # --save_dir
    config.save_dir = os.path.join(config.root_dir, config.save_dir_name, config.model_name, config.model_name + "_" + config.model_name_appendix)
    if not os.path.exists(config.save_dir):
        os.makedirs(config.save_dir)
    # --config_dir
    config.config_dir = os.path.join(config.save_dir, config.config_dir_name)
    if not os.path.exists(config.config_dir):
        os.makedirs(config.config_dir)
    #--reloaded_config_path
    config.reloaded_config_path = os.path.join(config.config_dir, config.config_name + ".pickle")
    #--reloaded_config_txt_path
    config.reloaded_config_txt_path = os.path.join(config.config_dir, config.config_name + ".txt")
    #--label_dir
    config.label_dir = os.path.join(config.data_dir, config.label_dir_name)
    #--inputdata_dir
    config.inputdata_dir = os.path.join(config.data_dir, config.inputdata_dir_name)
    #--ndvidata_dir
    config.ndvidata_dir = os.path.join(config.data_dir, config.ndvidata_dir_name)
    config.rntl_dir = os.path.join(config.data_dir, config.rntl_dir_name)
    #--chen
    config.chendata_dir = os.path.join(config.data_dir, config.chendata_dir_name)
    #--valid label_dir
    config.valid_label_dir = os.path.join(config.data_dir, config.valid_label_dir_name)
    #--valid inputdata_dir
    config.valid_inputdata_dir = os.path.join(config.data_dir, config.valid_inputdata_dir_name)
    #--valid ndvidata_dir
    config.valid_ndvidata_dir = os.path.join(config.data_dir, config.valid_ndvidata_dir_name)
    # --epoch
    try:
        assert config.num_epochs >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert config.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    if config.epoch == 0:
        config.avg_loss_G = []
        config.test_avg_loss_G = []
        config.avg_loss_D = []
        config.test_avg_loss_D = []
        #验证集指标
        # config.valid_quality_indices_dict = {'whole':{}}
        # config.valid_quality_indices_set0_dict = {'whole':{}}
        config.valid_quality_indices_dict = {}
        config.valid_quality_indices_set0_dict = {}
        config.valid_quality_indices_dict[Valid_Quality_Key] = {}
        config.valid_quality_indices_set0_dict[Valid_Quality_Key] = {}
        for ind in config.valid_quality_indices_dict:
            for q in Training_Quality_Indices_Lst:
                config.valid_quality_indices_dict[ind][q] = []
                config.valid_quality_indices_set0_dict[ind][q] = []
    # config.device_ids = [0,1] #默认值。后面根据机器情况具体定义
    return config

'''load arguments'''
def load_args(epoch=0,config=None,opts={},pathOpts={},device=None):
    if config is None:
        # parse arguments
        config = parse_args()
        # 更新用户指定参数
        for key in opts.keys():
            setattr(config, key, opts[key])

    #重新加载配置
    if epoch != 0:
        # 更新超参配置路径
        for key in pathOpts.keys():
            setattr(config, key, pathOpts[key])
        config = check_args(config)  # 设置路径
        config = reload_args(config, config.reloaded_config_path)

    # 更新用户指定参数
    config.epoch = epoch
    if device is not None:
        #device为None,config初始化时已根据机子情况设置了device;device不为None，则表示用户另外根据自己的需要重新设置device。
        config.device = device
    for key in opts.keys():
        setattr(config, key, opts[key])
    # 设置路径超参配置
    for key in pathOpts.keys():
        setattr(config, key, pathOpts[key])
    config = check_args(config)  # 设置路径

    return config

"""reload arguments: 输入的old config 路径要预先设置好"""
def reload_args(default_config, reloaded_config_path):
    with open(reloaded_config_path, "rb") as config_f:
        config = pickle.load(config_f)
    #复制没有的key
    new_keys = list(config.__dict__.keys())
    old_keys = list(default_config.__dict__.keys())
    for old_key in old_keys:
        if old_key not in new_keys:
            setattr(config, old_key, getattr(default_config, old_key))
    config.device = default_config.device
    #path
    config.root_dir = default_config.root_dir
    config.data_dir = default_config.data_dir
    config.save_dir = default_config.save_dir
    config.config_dir = default_config.config_dir
    config.reloaded_config_path = default_config.reloaded_config_path
    config.reloaded_config_txt_path = default_config.reloaded_config_txt_path
    config.train_sample_txt_path = default_config.train_sample_txt_path
    config.test_sample_txt_path = default_config.test_sample_txt_path
    config.valid_sample_dir = default_config.valid_sample_dir
    config.valid_sample_txt_lst = default_config.valid_sample_txt_lst
    config.valid_label_dir = default_config.valid_label_dir
    config.valid_inputdata_dir = default_config.valid_inputdata_dir
    config.valid_ndvidata_dir = default_config.valid_ndvidata_dir
    config.chendata_dir = default_config.chendata_dir
    config.label_dir = default_config.label_dir
    config.inputdata_dir = default_config.inputdata_dir
    config.ndvidata_dir = default_config.ndvidata_dir
    config.save_dir_name = default_config.save_dir_name
    config.config_dir_name = default_config.config_dir_name
    config.inputdata_dir_name = default_config.inputdata_dir_name
    config.label_dir_name = default_config.label_dir_name
    config.ndvidata_dir_name = default_config.ndvidata_dir_name
    config.chendata_dir_name = default_config.chendata_dir_name
    config.valid_label_dir_name = default_config.valid_label_dir_name
    config.valid_inputdata_dir_name = default_config.valid_inputdata_dir_name
    config.valid_ndvidata_dir_name = default_config.valid_ndvidata_dir_name
    config.rntl_dir_name = default_config.rntl_dir_name
    config.rntl_dir = default_config.rntl_dir

    # #训练参数
    config.epoch = default_config.epoch
    # config.num_epochs = default_config.num_epochs
    # config.batch_size = default_config.batch_size
    # config.test_batch_size = default_config.test_batch_size
    # config.learning_rate = default_config.learning_rate
    # config.gpu_mode = default_config.gpu_mode
    # config.save_epochs = default_config.save_epochs
    # config.sample_interval = default_config.sample_interval
    # config.epoch = default_config.epoch
    # config.epochs_upLr = default_config.epochs_upLr
    # config.save_plt_epochs = default_config.save_plt_epochs
    #loss
    config.avg_loss_G = config.avg_loss_G[0:default_config.epoch]
    config.test_avg_loss_G = config.test_avg_loss_G[0:default_config.epoch]
    if config.use_gan:
        config.avg_loss_D = config.avg_loss_D[0:default_config.epoch]
        config.test_avg_loss_D = config.test_avg_loss_D[0:default_config.epoch]
    # #验证集
    # for ind in config.valid_quality_indices_dict:
    #     for q in Training_Quality_Indices_Lst:
    #         config.valid_quality_indices_dict[ind][q] =  config.valid_quality_indices_dict[ind][q][0:default_config.epoch]
    return config

"""save arguments"""
def save_args(config):
    with open(config.reloaded_config_txt_path, "w") as text_file:
        text_file.write("%s" % config)
    criterion_content = config.criterion_content
    config.criterion_content = 0
    if config.gradient_loss:
        gradientLoss = config.gradientLoss
        config.gradientLoss=0
    generator = config.generator
    config.generator = 0
    if config.use_gan:
        discriminator = config.discriminator
        config.discriminator = 0
        criterion_raGAN = config.criterion_raGAN
        config.criterion_raGAN = 0
        feature_extractor = config.feature_extractor
        config.feature_extractor =0
    with open(config.reloaded_config_path, 'wb') as outfile:
        pickle.dump(config, outfile)
    config.criterion_content = criterion_content
    if config.gradient_loss:
        config.gradientLoss=gradientLoss
    config.generator = generator
    if config.use_gan:
        config.discriminator = discriminator
        config.criterion_raGAN = criterion_raGAN
        config.feature_extractor = feature_extractor

"""load model"""
# helper loading function that can be used by subclasses
def load_epoch_network(load_path, network,device=None,optimizer=None, strict=True):
    state_dict = torch.load(load_path, map_location=lambda storage, loc: storage)
    if isinstance(network, nn.DataParallel):
        network = network.module
    network.load_state_dict(state_dict['net'], strict=strict)
    if device is not None:
        network.to(device)
    if optimizer is not None:
        optimizer.load_state_dict(state_dict["optimizer"])
    print('Trained model is loaded.')
def load_epoch_network_AE(load_path, network,device=None,strict=True):
    # if isinstance(network, nn.DataParallel):
    #     network = network.module
    # network.load_state_dict(torch.load(load_path), strict=strict)
    state_dict = torch.load(load_path, map_location=lambda storage, loc: storage)
    if isinstance(network, nn.DataParallel):
        network = network.module
    network.load_state_dict(state_dict, strict=strict)
    if device is not None:
        network.to(device)
    print('Trained model is loaded.')
"""save model"""
# helper saving function that can be used by subclasses
def save_epoch_network(save_dir, network, network_label, iter_label,optimizer=None):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    save_filename = '{}_param_epoch_{}.pkl'.format(network_label, iter_label)
    save_path = os.path.join(save_dir, save_filename)
    state_dict = {}
    if isinstance(network, nn.DataParallel):
        network = network.module
    state_dict_net = network.state_dict()
    for key, param in state_dict_net.items():
        state_dict_net[key] = param.cpu()
    state_dict['net'] = state_dict_net
    if optimizer is not None:
        state_dict['optimizer'] = optimizer.state_dict()
    else:
        state_dict['optimizer'] = None
    #37 需要设置_use_new_zipfile_serialization=False，否则在笔记本上解析不了得到的network state_dict。
    #若37忘记设置，可以用下面的函数trans_plk_to_unzip（）进行转换
    # torch.save(state_dict, save_path,_use_new_zipfile_serialization=False)
    torch.save(state_dict, save_path)
def trans_plk_to_unzip():
    indir = '/home/zju/cxx/NTL_timeserise/data/SRCNN/net/AE'
    namelst = ['AE_01','AE_gradient_loss_-3',
               'AE_gradient_loss_-4','AE_gradient_loss_-2',
               'AE_02']
    epochlst = [121,421,2981,1001,4221]
    outdir = '/home/zju/cxx/NTL_timeserise/new_plk'
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    for i in range(len(namelst)):
        for j in range(len(epochlst)):
            print(namelst[i],epochlst[j])
            plkpath = os.path.join(indir,namelst[i],'config','generator_param_epoch_'+str(epochlst[j])+'.pkl')
            if os.path.exists(plkpath):
                state_dict = torch.load(plkpath)
                savedir = os.path.join(outdir,namelst[i])
                if not os.path.exists(savedir):
                    os.makedirs(savedir)
                savepath = os.path.join(savedir,'generator_param_epoch_'+str(epochlst[j])+'.pkl')
                torch.save(state_dict,savepath,_use_new_zipfile_serialization=False)
# trans_plk_to_unzip()
def get_optimizer(config):
    if config.optim_type == 'Adam':
        optimizer = torch.optim.Adam(config.generator.parameters(), lr=config.learning_rate,
                                       betas=(config.b1, config.b2))
    elif config.optim_type == 'SGD':
        optimizer = torch.optim.SGD(config.generator.parameters(), lr=config.learning_rate)
    elif config.optim_type == 'Momentum':
        optimizer = torch.optim.SGD(config.generator.parameters(), lr=config.learning_rate, momentum=config.momentum)
    elif config.optim_type == 'RMSProp':
        optimizer = torch.optim.RMSprop(config.generator.parameters(), lr=config.learning_rate, alpha=0.9)
    else:
        optimizer = torch.optim.Adam(config.generator.parameters(), lr=config.learning_rate,
                                     betas=(config.b1, config.b2))
    return optimizer


'''设置属性信息'''
def getDMSP_stat_dict(normlizedtype = 'minMax',min=0,max=63,mean=14.45,std=17.65):
    return {'normlizedtype':normlizedtype,'min':min,'max':max,'mean':mean,'std':std}
def getVIIRS_stat_dict(normlizedtype = 'minMax',min=0,max=1000,mean=1.12,std=8.662):
    return {'normlizedtype':normlizedtype,'min':min,'max':max,'mean':mean,'std':std}
def getNDVI_stat_dict(normlizedtype = 'minMax',min=0,max=1,mean=0.42696,std=0.3155):
    return {'normlizedtype': normlizedtype, 'min': min, 'max': max, 'mean': mean, 'std': std}
def getRNTL_stat_dict(normlizedtype = 'minMax',min=0,max=2000,mean=3.02,std=11.79):
    return {'normlizedtype': normlizedtype, 'min': min, 'max': max, 'mean': mean, 'std': std}
def getCfcvg_stat_dict(normlizedtype = 'minMax',min=0,max=128,mean=50.3,std=23.417):
    return {'normlizedtype': normlizedtype, 'min': min, 'max': max, 'mean': mean, 'std': std}
def getWater_Aridity_stat_dict(normlizedtype = None):
    return {'normlizedtype': normlizedtype}

'''生成SR图像'''
def generateSISR(config,network,out_dict):
    img = out_dict['input_inData'].to(config.device)
    if config.model_name in ['DNLSRNet','UNet']:
        gen_hr = network(img)

    return gen_hr
def generateSISR_partition(out_dict,configlst,networklst,minvalues,maxvalues):
    gen_hr = None
    maskdata = out_dict['dmspdata_ori_inter']
    maskdata = maskdata.to(configlst[-1].device)
    for i in range(len(minvalues)):
        minval = minvalues[i]
        maxval = maxvalues[i]
        config = configlst[i]
        network = networklst[i]
        mask_loc = (maskdata <= minval) | (maskdata > maxval)
        if gen_hr is None:
            gen_hr = generateSISR(config, network, out_dict)
            gen_hr = torch.where(mask_loc,gen_hr,torch.full_like(gen_hr, 0))

        else:
            gen_hr1 = generateSISR(config, network, out_dict)
            gen_hr = torch.where(mask_loc, gen_hr1, gen_hr)
    return gen_hr
'''网络输出图像恢复为原始VNL形式（图像质量评价：预测数据还原为viirs原始形式）'''
def outdata_transform_qualityassess(config,gen_hr):
    if torch.is_tensor(gen_hr):
        if config.label_use_ln & (config.labeldata_normlizeType=='minMax'):
            outdata = gen_hr * (np.log(config.label_maxDN+config.viirs_log_add_smallValue)-np.log(config.viirs_log_add_smallValue))\
                      +np.log(config.viirs_log_add_smallValue)
            outdata = torch.exp(outdata) - config.viirs_log_add_smallValue
        else:
            if config.viirs_stat_dict['normlizedtype'] == 'minMax':
                outdata = gen_hr * (config.label_maxDN - config.label_minDN) + config.label_minDN
            elif config.viirs_stat_dict['normlizedtype']== 'meanStd':
                outdata = gen_hr*config.label_std+config.label_meanDN
            else:
                outdata = gen_hr
        outdata = torch.where(outdata < 0, torch.full_like(outdata, 0), outdata)
    else:
        if config.label_use_ln & (config.labeldata_normlizeType=='minMax'):
            outdata = gen_hr * (np.log(config.label_maxDN+config.viirs_log_add_smallValue)-np.log(config.viirs_log_add_smallValue))\
                      +np.log(config.viirs_log_add_smallValue)
            outdata = np.exp(outdata) -config.viirs_log_add_smallValue
        else:
            if config.viirs_stat_dict['normlizedtype']== 'minMax':
                outdata = gen_hr * (config.label_maxDN - config.label_minDN) + config.label_minDN
            elif config.viirs_stat_dict['normlizedtype'] == 'meanStd':
                outdata = gen_hr * config.label_std + config.label_meanDN
            else:
                outdata = gen_hr
        outdata = np.where(outdata < 0, 0, outdata)
    return outdata
'''图像质量评价：labeldata 处理'''
def labeldata_transform_qualityassess(viirs_ori):
    if torch.is_tensor(viirs_ori):
        viirs_ori1 = torch.where(viirs_ori > 1, viirs_ori, torch.full_like(viirs_ori, 0))
    else:
        viirs_ori1 = np.where(viirs_ori > 1,viirs_ori,0)
    return viirs_ori1




'''设置随机种子'''
def setRandomSeed(seed=3000):
    iscuda = True if torch.cuda.is_available() else False
    device = torch.device('cuda' if iscuda else 'cpu')
    print("Start seed: ", seed)
    # random.seed(args.seed)
    torch.manual_seed(seed)  # 为CPU设置种子用于生成随机数，以使得结果是确定的
    if iscuda:
        torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子；
        torch.cuda.manual_seed_all(seed)  # 如果使用多个GPU，应该使用torch.cuda.manual_seed_all()为所有的GPU设置种子。
        cudnn.deterministic = True
    cudnn.benchmark = True
    torch.backends.cudnn.benchmark = True
    return iscuda,device
def initSetting(seed=3000):
    '''
    默认值初始化config,设置随机种子和device
    :param seed:
    :return:
    '''
    ########### 默认配置
    iscuda, device = setRandomSeed(seed)
    config = parse_args()
    config.cuda = iscuda
    config.seed = seed
    config.device = device
    # config.device_ids = device_ids
    ############################
    return config
def getconfig(modelname,modelappendix,epoch,seed=3000,opts={},pathOpts={}):
    config = initSetting(seed)
    config.model_name = modelname
    config.model_name_appendix = modelappendix
    config.epoch = epoch
    config = load_args(epoch, config, opts, pathOpts)
    return config
def getNet_fromModel(config, epoch):
    #eval predt时使用
    G_model_path = os.path.join(config.config_dir, 'generator_param_epoch_%d.pkl' % epoch)
    if config.model_name == 'DNLSRNet':
        net = DNLSRNet(config,config.in_channels,config.base_filter)
    elif config.model_name == 'UNet':
        net = U_Net(config.in_channels, 1, config.base_filter, config.unet_encoder_act,
                    config.unet_encoder_norm, config.unet_decoder_act, config.unet_decoder_norm,
                    config.unet_tail_act, config.unet_tail_norm)
    load_epoch_network(load_path=G_model_path, network=net)
    net.to(config.device)
    # if len(config.device_ids) > 1:
    #     net = nn.DataParallel(net)
    #     net.to(config.device)
    return net

