import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np
from datetime import datetime
import skimage.measure
import sklearn.metrics
import matplotlib.pyplot as plt
import os
import time
import sys
import pandas as pd
import pickle
import argparse

from NPP_LIKE.options import *
from NPP_LIKE.utils import sum_dict,weights_init_normal,weights_init_kaming,weigth_init_xavier,clip_img
from NPP_LIKE.base_networks import Discriminator,FeatureExtractor
from NPP_LIKE.load_Dataset import Dataset_oriDMSP
from NPP_LIKE.loss import GradientLoss,GANLoss
from NPP_LIKE.net.DRLN import DNLSRNet
from NPP_LIKE.net.Unet import U_Net

'''损失计算：输出数据处理'''
def outdata_transform_train(config, out_dict):
    '''训练模型：用于计算损失的预测结果的转化处理（如是否还原为原来的数据再计算损失。这里采用预测结果不做处理'''
    outdata = out_dict['outdata']
    is_cuda = outdata.is_cuda
    # 用归一化后的数据计算损失，输出
    labeldata = out_dict['label_inData']
    if is_cuda:
        labeldata = labeldata.cuda()
    out_dict['labeldata'] = labeldata
    out_dict['outdata'] = outdata
    return out_dict
def outdata_mask_trainLoss(config, out_dict, masktype):
    '''训练模型：用于计算损失的预测结果的掩膜处理'''
    #1 参与掩膜的数据
    outdata = out_dict['outdata']
    labeldata = out_dict['labeldata']
    mask_loc = None
    #2 掩膜参照数据
    if config.outdata_mask == '0':
        maskdata = out_dict['label_inData']
    elif config.outdata_mask == '1':
        maskdata = out_dict['dmspdata_ori_inter']
    elif config.outdata_mask == '2':
        maskdata = out_dict['viirsdata_ori']
    #3 是否裁剪
    if config.isClip_to_lossCal:
        pd = config.PaddingClipped_to_lossCal
        outdata = clip_img(outdata,pd)
        labeldata = clip_img(labeldata,pd)
        maskdata = clip_img(maskdata,pd)

    #4 掩膜条件设置
    if masktype == '0':
        if outdata.is_cuda:
            maskdata = maskdata.cuda()
        # mask DN = 0
        mask_loc = (maskdata < 1e-6) & (maskdata > -1e-6)
    elif masktype == '1':
        maskdata = out_dict['dmspdata_ori_inter']
        if outdata.is_cuda:
            maskdata = maskdata.cuda()
        # extrct urban area
        if (config.urbFocus_loss_urban_max_DN <= config.urbFocus_loss_urban_min_DN) | (config.urbFocus_loss_urban_max_DN>=63):
            mask_loc = (maskdata < config.urbFocus_loss_urban_min_DN)
        else:
            mask_loc = (maskdata < config.urbFocus_loss_urban_min_DN)|(maskdata > config.urbFocus_loss_urban_max_DN)
    elif masktype == '2':
        # get whole
        mask_loc = None

    #5 进行掩膜计算
    if mask_loc is not None:
        outdata = torch.where(mask_loc,torch.full_like(outdata,0),outdata)
        labeldata = torch.where(mask_loc, torch.full_like(labeldata,0), labeldata)
    return outdata,labeldata
def gradientData_mask_trainLoss(G_genhr,G_label, out_dict):
    '''训练模型：用于计算损失的预测结果的掩膜处理'''
    #1 参与掩膜的数据
    maskdata = out_dict['dmspdata_ori_inter']
    if G_genhr.is_cuda:
        maskdata = maskdata.cuda()
    mask_loc = (maskdata < 40)
    G_genhr_mask = torch.where(mask_loc,torch.full_like(G_genhr,0),G_genhr)
    G_label_mask = torch.where(mask_loc, torch.full_like(G_label,0), G_label)
    return G_genhr_mask,G_label_mask
'''计算损失'''
def loss_Generators(config,out_dict,loss_G):
    '''计算损失'''
    if config.gradient_loss | config.noMask | config.use_gan | config.perceptual_loss:
        gen_hr_mask, label_mask = outdata_mask_trainLoss(config, out_dict, '2')
        if config.noMask:
            mse_loss_G = config.criterion_content(gen_hr_mask, label_mask)
            loss_G = loss_G + mse_loss_G
        if config.gradient_loss:
            G_gen_hr,G_label = config.gradientLoss(gen_hr_mask, label_mask)
            if config.isUrbanMask_gradient_loss:
                G_gen_hr, G_label = gradientData_mask_trainLoss(G_gen_hr,G_label,out_dict)
            gradientloss = config.criterion_content(G_gen_hr,G_label)
            loss_G += config.weight_gradient * gradientloss
        if config.perceptual_loss:
            gen_features = config.feature_extractor(gen_hr_mask)
            real_features = Variable(config.feature_extractor(label_mask).data, requires_grad=False)
            loss_content = config.criterion_content(gen_features, real_features)
            loss_G += config.weight_content * loss_content
        # gan loss
        if config.use_gan:
            gen_validity = config.discriminator(gen_hr_mask)
            loss_gan = config.criterion_raGAN(gen_validity, True)
            loss_G += config.weight_gan * loss_gan

    #过滤不参与计算的像元（set 0）后计算损失
    if config.gt0Mask:
        gen_hr_mask, label_mask = outdata_mask_trainLoss(config, out_dict, '0')
        mse_loss_G = config.criterion_content(gen_hr_mask, label_mask)
        loss_G = loss_G + mse_loss_G
    if config.urbanMask:
        gen_hr_mask, label_mask = outdata_mask_trainLoss(config, out_dict, '1')
        mse_loss_G = config.criterion_content(gen_hr_mask, label_mask)
        loss_G = loss_G + mse_loss_G

    return loss_G
def loss_Discriminator(config,out_dict):
    # Loss of real and fake images
    gen_hr_mask, label_mask = outdata_mask_trainLoss(config, out_dict, '2')
    loss_real = config.criterion_raGAN(config.discriminator(label_mask), True)
    loss_fake = config.criterion_raGAN(config.discriminator(gen_hr_mask), False)
    loss_D = loss_real + loss_fake
    return loss_D
'''训练损失图'''
def plot_train_loss(config,start_epoch,end_epoch,save_dir='',show=False):
    xlst = list(range(1,len(config.avg_loss_G)+1,1))
    # loss
    train_loss = config.avg_loss_G[start_epoch-1:end_epoch]
    test_loss =  config.test_avg_loss_G[start_epoch-1:end_epoch]
    xlst = xlst[start_epoch-1:end_epoch]
    plt.figure()
    if config.use_gan:
        train_loss_D = config.avg_loss_D[start_epoch:end_epoch + 1]
        test_loss_D = config.test_avg_loss_D[start_epoch:end_epoch + 1]
        ax = plt.subplot(1, 2, 1)
        ax.set_xlim(start_epoch, end_epoch + 1)
        plt.plot(xlst, train_loss, label='train_loss_G')
        plt.plot(xlst, test_loss, label='test_loss_G')
        plt.title('Generator')
        plt.legend()
        ax = plt.subplot(1, 2, 2)
        ax.set_xlim(start_epoch, end_epoch + 1)
        plt.plot(xlst, train_loss_D, label='train_loss_D')
        plt.plot(xlst, test_loss_D, label='test_loss_D')
        plt.title('Disciminator')
        plt.legend()
    else:
        plt.plot(xlst, train_loss, label='train_loss')
        plt.plot(xlst, test_loss, label='test_loss')
        plt.xlim(start_epoch,end_epoch+1)
        plt.title('Generator')
        plt.legend()
    # save figure
    if save_dir.strip() == '':
        save_dir = config.config_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_fn = 'train_loss_startEpoch_{:d}'.format(start_epoch) + '.png'
    save_fn = os.path.join(save_dir, save_fn)
    plt.savefig(save_fn)
    plt.close()
def plot_train_valid_loss(config,start_epoch=0,save_dir='',show=False):
    #valid
    epochlst = config.valid_quality_indices_dict[Valid_Quality_Key]['EPOCH']
    rmse_lst = config.valid_quality_indices_dict[Valid_Quality_Key]['RMSE']
    mre_lst = config.valid_quality_indices_dict[Valid_Quality_Key]['MRE']
    tnl_lst = config.valid_quality_indices_dict[Valid_Quality_Key]['TNL']
    if start_epoch in epochlst:
        index = epochlst.index(start_epoch)
        epochlst = epochlst[index:]
        rmse_lst = rmse_lst[index:]
        mre_lst = mre_lst[index:]
        tnl_lst = tnl_lst[index:]
    else:
        start_epoch = epochlst[0]
    plt.figure(figsize=(16,9))
    ax = plt.subplot(1, 3, 1)
    plt.plot(epochlst, rmse_lst, label='RMSE')
    plt.title('RMSE')
    plt.legend()
    ax = plt.subplot(1, 3, 2)
    plt.plot(epochlst, mre_lst, label='MRE')
    plt.title('MRE')
    plt.legend()
    ax = plt.subplot(1, 3, 3)
    plt.plot(epochlst, tnl_lst, label='TNL')
    plt.title('TNL')
    plt.legend()
    if save_dir.strip() == '':
        save_dir = config.config_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_fn = 'valid_'+str(start_epoch)+'.png'
    save_fn = os.path.join(save_dir, save_fn)
    plt.savefig(save_fn)
    plt.close()



'''validate'''
def validate(config,network,valid_dataloader,epoch = 0,mode='test',save_img=False):
    #load model
    if mode == 'test':
        G_model_path = os.path.join(config.config_dir, 'generator_param_epoch_%d.pkl' % epoch)
        load_epoch_network(load_path=G_model_path, network=network)
        if len(config.device_ids)>1:
            network = nn.DataParallel(network)
        network.to(config.device)
    network.eval()
    #valid
    #设置变量
    gen_indices_dict = {}
    l1_loss = torch.nn.L1Loss(reduction='sum').to(config.device)
    l2_loss = torch.nn.MSELoss(reduction='sum').to(config.device)
    qi_mse = 0
    qi_mae = 0
    qi_sum = 0
    qi_tnlerr=0
    img_num = 0
    #生成超分图像并计算图像质量指数
    with torch.no_grad():
        for idx,out_dict in enumerate(valid_dataloader):
            # generate SR
            gen_hr = generateSISR(config,network,out_dict)
            out_dict['outdata'] = gen_hr
            # pixel loss
            batchnum = gen_hr.shape[0]
            #数据处理
            outdata = outdata_transform_qualityassess(config,gen_hr)
            labeldata = out_dict['viirsdata_ori'].to(config.device)
            #裁剪
            if config.isClip_to_lossCal:
                outdata = clip_img(outdata,config.PaddingClipped_to_lossCal)
                labeldata = clip_img(labeldata, config.PaddingClipped_to_lossCal)
            #质量评价
            qi_mse += l2_loss(outdata,labeldata).data.item()
            qi_mae += l1_loss(outdata,labeldata).data.item()
            qi_sum += labeldata.sum().data.item()
            #计算图像块TNL差值
            srTNL = torch.sum(outdata,dim=(1,2,3))
            hrTNL = torch.sum(labeldata,dim=(1,2,3))
            qi_tnlerr += l1_loss(srTNL,hrTNL).data.item()
            img_num += batchnum
    qi_rmse = (qi_mse/img_num)**0.5
    qi_mre = qi_mae/qi_sum
    qi_tnlerr = qi_tnlerr/img_num
    gen_indices = {'EPOCH':epoch,'RMSE':qi_rmse,'MRE':qi_mre,'TNL':qi_tnlerr}
    gen_indices_dict[Valid_Quality_Key] = gen_indices
    return gen_indices_dict
'''test'''
def test(config, net, test_loader):
    torch.cuda.empty_cache()
    net.eval()  # 在测试模型时在前面使用model.eval()。可以禁止forward过程改变权值。如Batch Normalization和drop会使测试集和训练集的样本分布不一样，会有batch normalization 所带来的影响。
    if config.use_gan:
        config.discriminator.eval()
        test_epoch_loss_D = 0

    test_epoch_loss_G = 0
    test_num = 0

    # img,ndvidata, label
    with torch.no_grad():
        for idx, out_dict in enumerate(test_loader):
            # generate SR
            gen_hr = generateSISR(config,net,out_dict)
            batchnum = gen_hr.shape[0]
            out_dict['outdata'] = gen_hr
            out_dict = outdata_transform_train(config, out_dict)

            # pixel loss
            loss_G = 0
            loss_G = loss_Generators(config, out_dict, loss_G)
            test_epoch_loss_G += loss_G.data.item()

            # ------------------
            #  test Disiciminator
            # ------------------
            if config.use_gan:
                loss_D = loss_Discriminator(config, out_dict)
                test_epoch_loss_D += loss_D.data.item()

            test_num += batchnum
    avg_loss_G = test_epoch_loss_G/test_num
    avg_loss_dic = {'avg_loss_G':avg_loss_G}
    if config.use_gan:
        avg_loss_D = test_epoch_loss_D / test_num
        avg_loss_dic['avg_loss_D'] = avg_loss_D
    torch.cuda.empty_cache()
    net.train()  # 在训练模型时在前面使用model.train()
    if config.use_gan:
        config.discriminator.train()
    return avg_loss_dic
'''train'''
def train(config,network):
    #net
    config.generator = network
    # Losses
    if config.loss_Lp_norm== "L1":
        config.criterion_content = torch.nn.L1Loss(reduction = 'sum').to(config.device)
    else:
        config.criterion_content = torch.nn.MSELoss(reduction = 'sum').to(config.device)
    if config.gradient_loss:
        config.gradientLoss = GradientLoss().to(config.device)
    if config.perceptual_loss:
        config.feature_extractor = FeatureExtractor(torchvision.models.vgg19(pretrained=True))
    if config.use_gan:
        config.discriminator = Discriminator(norm_type='batch', use_spectralnorm=False, attention=False)
        config.criterion_raGAN = GANLoss(gan_type='vanilla', real_label_val=1.0, fake_label_val=0.0,reduction='sum')
    # Optimizers
    optimizer_G = get_optimizer(config)
    if config.use_gan:
        optimizer_D = torch.optim.SGD(config.discriminator.parameters(), lr=config.learning_rate / 100, momentum=0.9, nesterov=True)

    if len(config.device_ids) > 1:
        config.generator = nn.DataParallel(config.generator)
        if config.perceptual_loss:
            config.feature_extractor = nn.DataParallel(config.feature_extractor).to(config.device)
        if config.use_gan:
            config.discriminator = nn.DataParallel(config.discriminator).to(config.device)
    if config.epoch != 0:
        # Load pretrained models
        G_model_path = os.path.join(config.config_dir , 'generator_param_epoch_%d.pkl' % config.epoch)
        load_epoch_network(load_path=G_model_path, network=config.generator,optimizer=optimizer_G,device=config.device)
        for param_group in optimizer_G.param_groups:
            param_group["lr"] = config.learning_rate
        if config.use_gan:
            D_model_path = os.path.join(config.config_dir,'discriminator_param_epoch_%d.pkl' % config.epoch)
            load_epoch_network(load_path=D_model_path, network=config.discriminator,optimizer=optimizer_D)

    else:
        # Initialize weights
        if config.weight_init_name == 'normal':
            config.generator.apply(weights_init_normal)
        elif config.weight_init_name == 'kaming':
            config.generator.apply(weights_init_kaming)
        elif config.weight_init_name == 'xavier':
            config.generator.apply(weigth_init_xavier)
        if config.use_gan:
            # Initialize weights
            config.discriminator.apply(weights_init_normal)

    #get dataset,dataloader
    train_dataset = Dataset_oriDMSP(config,config.train_sample_txt_path,'train',is_multi_years_combined=config.is_multi_years_combined)
    test_dataset = Dataset_oriDMSP(config, config.test_sample_txt_path, 'test',is_multi_years_combined=config.is_multi_years_combined)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True,
                                                   num_workers=config.num_threads, drop_last=False)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=config.test_batch_size, shuffle=False,
                                                  num_workers=config.num_threads, drop_last=False)
    #valid dataset
    sample_txt_lst = config.valid_sample_txt_lst.split(';')
    sample_txt = sample_txt_lst[0]
    valid_sample_txt_path = os.path.join(config.valid_sample_dir,sample_txt+'.txt')
    valid_dataset = Dataset_oriDMSP(config,valid_sample_txt_path,'valid',config.valid_label_dir,config.valid_inputdata_dir,config.waterdata_dir,config.valid_otherdata_dict,config.data_dir,is_multi_years_combined=config.is_multi_years_combined)
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset,batch_size=config.test_batch_size, shuffle=False,num_workers=config.num_threads, drop_last=False)

    print('Training is started.',datetime.now())
    config.generator.to(config.device).train()
    if config.use_gan:
        config.discriminator.to(config.device).train()

    step = 0
    start_time = time.time()
    epoch = config.epoch
    config.last_lr = optimizer_G.param_groups[0]["lr"]
    epochs_upLr = list(map(int,config.epochs_upLr.split(',')))
    while epoch < config.num_epochs:
        if epoch in epochs_upLr:
            for param_group in optimizer_G.param_groups:
                param_group["lr"] /= 2.0
                config.last_lr = param_group["lr"]
            if config.use_gan:
                for param_group in optimizer_D.param_groups:
                    param_group["lr"] /= 2.0

        train_epoch_loss_G = 0
        train_num = 0
        if config.use_gan:
            epoch_loss_D = 0
        # img,ndvidata, label
        for idx,out_dict in enumerate(train_dataloader):
            # ------------------
            #  Train Generators
            # ------------------
            # torch.autograd.set_detect_anomaly(True)
            optimizer_G.zero_grad()
            #generate SR
            gen_hr = generateSISR(config,config.generator,out_dict)
            batchnum,h,w = gen_hr.shape[0],gen_hr.shape[2],gen_hr.shape[3]
            out_dict['outdata'] = gen_hr
            out_dict = outdata_transform_train(config, out_dict)

            loss_G = 0
            loss_G = loss_Generators(config,out_dict,loss_G)
            train_epoch_loss_G += loss_G.data.item()
            train_num += batchnum
            loss_G = loss_G / (batchnum*h*w)
            # with torch.autograd.detect_anomaly():
            #     loss_G.backward(retain_graph=True)
            loss_G.backward(retain_graph=True)
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------
            if config.use_gan:
                optimizer_D.zero_grad()
                loss_D = loss_Discriminator(config,out_dict)
                epoch_loss_D = epoch_loss_D + loss_D.data.item()
                loss_D = loss_D / (batchnum*h*w)
                loss_D.backward(retain_graph=True)
                optimizer_D.step()
                # # Clip weights of discriminator
                # for p in config.discriminator.parameters():
                #     p.data.clamp_(-config.clip_value, config.clip_value)

            # --------------
            #  Log Progress
            # --------------
            if idx % 50 == 0:
                print("\nmodel:{} {} [Train Epoch: {}/{}] [Batch: {}/{}] [G loss: {:.4f}] time: {}\n".format(
                    config.model_name,config.model_name_appendix, epoch, config.num_epochs, idx, len(train_dataloader), loss_G.item(),datetime.now()))
                if config.use_gan:
                    print("\nmodel:{} {} [Train Epoch: {}/{}] [Batch: {}/{}] [D loss: {:.4f}] time: {}\n".format(
                        config.model_name, config.model_name_appendix, epoch, config.num_epochs, idx,
                        len(train_dataloader), loss_D.item() , datetime.now()))

            if config.use_gan:
                del loss_D
            step += 1
            del  gen_hr,out_dict,loss_G
            torch.cuda.empty_cache()
        # avg. loss per epoch
        train_loss = train_epoch_loss_G / train_num
        config.avg_loss_G.append(train_loss)
        #test
        test_avg_loss_dic = test(config, config.generator, test_dataloader)
        test_loss = test_avg_loss_dic['avg_loss_G']
        config.test_avg_loss_G.append(test_loss)
        torch.cuda.empty_cache()
        print('\nmodel:{} {} Train Epoch:{} Train loss:{:.4f} Test loss:{:.4f} time: {} \n'.format(
            config.model_name, config.model_name_appendix, epoch, train_loss, test_loss, datetime.now()))
        if config.use_gan:
            train_loss_D = epoch_loss_D / train_num
            config.avg_loss_D.append(train_loss_D)
            config.test_avg_loss_D.append(test_avg_loss_dic['avg_loss_D'])
            print('\nmodel:{} {} Train Epoch: {} Train loss D:{:.4f} Test loss D:{:.4f} time: {}\n'.format(
                config.model_name, config.model_name_appendix, epoch, train_loss_D,test_avg_loss_dic['avg_loss_D'], datetime.now()))

        # global epoch (样本集训练、测试、GAN 判别器训练、测试完成，epoch+1)
        epoch += 1

        #下面代码的epoch都是表示训练了几次。如训练了1次，epoch为1.
        #保存配置
        if epoch % config.save_epochs == 0:
            config.generator.cpu()
            save_epoch_network(save_dir=config.config_dir, network=config.generator, network_label='generator',
                               iter_label=epoch,optimizer=optimizer_G)
            save_args(config)
            config.generator.to(config.device)
            if config.use_gan:
                config.discriminator.cpu()
                save_epoch_network(save_dir=config.config_dir, network=config.discriminator, network_label='discriminator',
                                   iter_label=epoch,optimizer=optimizer_D)
                config.discriminator.to(config.device)
        #保存图片
        if epoch % config.save_plt_epochs == 0:
            if epoch < 600:
                plot_train_loss(config,1,epoch)
            else:
                plot_train_loss(config, 50, epoch)
        #validate
        if epoch%config.sample_interval == 0:
            #valid 在epoch+1之后，使得验证集评价结果与模型文件中的epoch对应
            valid_quality_indices_dict = validate(config,config.generator,valid_dataloader,epoch,mode='train')
            torch.cuda.empty_cache()
            config.generator.train()
            for index in valid_quality_indices_dict:
                for qind in config.valid_quality_indices_dict[index]:
                    config.valid_quality_indices_dict[index][qind].append(valid_quality_indices_dict[index][qind])
            plot_train_valid_loss(config,config.sample_interval)

        torch.cuda.empty_cache()

    # Plot avg. loss
    print("Training is finished.", datetime.now())
def main_train(config):
    # if config.gpu_mode and not torch.cuda.is_available():
    #     raise Exception("No GPU found, please run without --gpu_mode=False")
    # model
    if config.model_name == 'DNLSRNet':
        net = DNLSRNet(config,config.in_channels,config.base_filter)
    elif config.model_name == 'UNet':
        net = U_Net(config.in_channels,1,config.base_filter,config.unet_encoder_act,
                    config.unet_encoder_norm,config.unet_decoder_act,config.unet_decoder_norm,
                    config.unet_tail_act,config.unet_tail_norm)
    else:
        raise Exception("[!] There is no option for " + config.model_name)
    if config.subcommand == "train":
        train(config, net)
def run_train(epoch=0,model_name='DNLSRNet',model_name_appendix='01',seed=3000,opts={},pathOpts={}):
    config = getconfig(model_name,model_name_appendix,epoch,seed,opts,pathOpts)
    main_train(config)

class RunModel():
    def DNLSRNet_2012_01(self):
        ###########  根据机器GPU使用情况设置device_ids
        device_ids = [0,1]
        # device_ids = [0]
        # torch.cuda.set_device(1)
        ###########
        epoch = 165
        model_name = 'DNLSRNet'
        seed = 3000
        model_name_appendix = '2012_01_seed'+str(seed)+'_NoCA'

        pathOpts = {}
        opts={}
        #模型参数设置
        opts['isNDVIcoefficient'] = False
        opts['attentiontype'] = None
        #训练设置
        opts['batch_size']=55
        opts['test_batch_size']=30
        opts['save_plt_epoches'] = 50
        opts['sample_interval'] = 1
        opts['save_epochs']=1
        opts['num_epochs']=4000
        opts['epochs_upLr'] = "5000,8000"
        opts['device_ids'] = device_ids
        run_train(epoch=epoch,model_name=model_name,model_name_appendix=model_name_appendix,seed=seed,opts=opts,pathOpts=pathOpts)

    def Unet_01(self):
        ###########  根据机器GPU使用情况设置device_ids
        # device_ids = [0,1]
        device_ids = [0]
        # torch.cuda.set_device(1)
        ###########
        epoch = 0
        model_name = 'UNet'
        seed = 3000
        model_name_appendix = '01'

        ###########设置参数
        pathOpts = {}
        opts={}

        opts['dmsp_stat_dict'] = getDMSP_stat_dict()
        opts['viirs_stat_dict'] = getVIIRS_stat_dict()
        ndvi_stat_dict = getNDVI_stat_dict()
        ndvi_stat_dict['path'] = r'D:\04study\00Paper\Dissertation\01experiment\04NTL_Timeseries\data\net\SRCNN\data\img04\2012_NDVI'
        opts['otherdata_dict'] = {'NDVI':ndvi_stat_dict}
        opts['valid_otherdata_dict'] = {'NDVI':ndvi_stat_dict}
        opts['is_multi_years_combined'] = False
        opts['is_arid_one_hot'] = False

        opts['batch_size'] = 50
        opts['test_batch_size'] = 20
        opts['save_plt_epoches'] = 1
        opts['sample_interval'] = 50
        opts['save_epochs'] = 1
        opts['num_epochs'] = 4000
        opts['epochs_upLr'] = "5000,8000"
        opts['device_ids'] = device_ids
        run_train(epoch=epoch, model_name=model_name, model_name_appendix=model_name_appendix, seed=seed, opts=opts,
                  pathOpts=pathOpts)

    def Unet_01_labelNoProcess(self):
        ###########  根据机器GPU使用情况设置device_ids
        # device_ids = [0,1]
        device_ids = [0]
        torch.cuda.set_device(0)
        ###########
        epoch = 0
        model_name = 'UNet'
        seed = 3000
        model_name_appendix = '01_labelNoProcess'

        ###########设置参数
        pathOpts = {}
        opts = {}
        opts['dmsp_stat_dict'] = getDMSP_stat_dict()
        opts['viirs_stat_dict'] = getVIIRS_stat_dict(normlizedtype=None)
        opts['label_use_ln'] = False
        ndvi_stat_dict = getNDVI_stat_dict()
        ndvi_stat_dict['path'] = r'/home/zju/cxx/NTL_timeserise/data/SRCNN/data/img04/2012_NDVI'
        opts['otherdata_dict'] = {'NDVI': ndvi_stat_dict}
        opts['valid_otherdata_dict'] = {'NDVI': ndvi_stat_dict}
        opts['is_multi_years_combined'] = False
        opts['is_arid_one_hot'] = False

        opts['batch_size'] = 50
        opts['test_batch_size'] = 50
        opts['save_plt_epoches'] = 1
        opts['sample_interval'] = 1
        opts['save_epochs'] = 1
        opts['num_epochs'] = 4000
        opts['epochs_upLr'] = "5000,8000"
        opts['device_ids'] = device_ids
        run_train(epoch=epoch, model_name=model_name, model_name_appendix=model_name_appendix, seed=seed, opts=opts,
                  pathOpts=pathOpts)
    def Unet_01_MeanStd(self):
        ###########  根据机器GPU使用情况设置device_ids
        # device_ids = [0,1]
        device_ids = [1]
        torch.cuda.set_device(1)
        ###########
        epoch = 0
        model_name = 'UNet'
        seed = 3000
        model_name_appendix = '01_MeanStd'

        ###########设置参数
        pathOpts = {}
        opts = {}
        opts['dmsp_stat_dict'] = getDMSP_stat_dict(normlizedtype="meanStd")
        opts['viirs_stat_dict'] = getVIIRS_stat_dict(normlizedtype="meanStd")
        opts['label_use_ln'] = False
        ndvi_stat_dict = getNDVI_stat_dict(normlizedtype="meanStd")
        ndvi_stat_dict['path'] = r'/home/zju/cxx/NTL_timeserise/data/SRCNN/data/img04/2012_NDVI'
        opts['otherdata_dict'] = {'NDVI': ndvi_stat_dict}
        opts['valid_otherdata_dict'] = {'NDVI': ndvi_stat_dict}
        opts['is_multi_years_combined'] = False
        opts['is_arid_one_hot'] = False

        opts['batch_size'] = 50
        opts['test_batch_size'] = 50
        opts['save_plt_epoches'] = 1
        opts['sample_interval'] = 1
        opts['save_epochs'] = 1
        opts['num_epochs'] = 4000
        opts['epochs_upLr'] = "5000,8000"
        opts['device_ids'] = device_ids
        run_train(epoch=epoch, model_name=model_name, model_name_appendix=model_name_appendix, seed=seed, opts=opts,
                  pathOpts=pathOpts)
    def Unet_01_train12_13(self):
        ###########  根据机器GPU使用情况设置device_ids
        # device_ids = [0,1]
        device_ids = [0]
        torch.cuda.set_device(0)
        ###########
        epoch = 0
        model_name = 'UNet'
        seed = 3000
        model_name_appendix = '01_train12_13'

        ###########设置参数
        pathOpts = {}
        pathOpts['train_sample_txt_path'] = '/home/zju/cxx/NTL_timeserise/data/SRCNN/data/sample/01/04/train_08.txt'
        pathOpts['test_sample_txt_path'] = '/home/zju/cxx/NTL_timeserise/data/SRCNN/data/sample/01/04/test_08.txt'
        pathOpts['valid_sample_txt_lst'] = 'valid_08'
        pathOpts['label_dir_name'] = '2012_2013VNL'
        pathOpts['inputdata_dir_name'] = '2012_2013oriDNL'
        pathOpts['ndvidata_dir_name'] = '2012_2013NDVI'
        pathOpts['valid_label_dir_name'] = '2012_2013VNL'
        pathOpts['valid_inputdata_dir_name'] = '2012_2013oriDNL'
        pathOpts['valid_ndvidata_dir_name'] = '2012_2013NDVI'

        opts={}
        opts['dmsp_stat_dict'] = getDMSP_stat_dict()
        opts['viirs_stat_dict'] = getVIIRS_stat_dict()
        ndvi_stat_dict = getNDVI_stat_dict()
        ndvi_stat_dict['path'] = r'/home/zju/cxx/NTL_timeserise/data/SRCNN/data/img04/2012_2013NDVI'
        opts['otherdata_dict'] = {'NDVI':ndvi_stat_dict}
        opts['valid_otherdata_dict'] = {'NDVI':ndvi_stat_dict}
        opts['is_multi_years_combined'] = True
        opts['is_arid_one_hot'] = False


        opts['batch_size'] = 50
        opts['test_batch_size'] = 50
        opts['save_plt_epoches'] = 1
        opts['sample_interval'] = 50
        opts['save_epochs'] = 1
        opts['num_epochs'] = 4000
        opts['epochs_upLr'] = "5000,8000"
        opts['device_ids'] = device_ids
        run_train(epoch=epoch, model_name=model_name, model_name_appendix=model_name_appendix, seed=seed, opts=opts,
                  pathOpts=pathOpts)
    def Unet_01_RNTL(self):
        ###########  根据机器GPU使用情况设置device_ids
        # device_ids = [0,1]
        device_ids = [1]
        torch.cuda.set_device(1)
        ###########
        epoch = 0
        model_name = 'UNet'
        seed = 3000
        model_name_appendix = '01_RNTL'

        ###########设置参数
        pathOpts = {}
        opts={}
        opts['dmsp_stat_dict'] = getDMSP_stat_dict()
        opts['viirs_stat_dict'] = getVIIRS_stat_dict()
        rntl_stat_dict = getRNTL_stat_dict()
        rntl_stat_dict['path'] = r'/home/zju/cxx/NTL_timeserise/data/SRCNN/data/img04/2010_RNTL'
        opts['otherdata_dict'] = {'RNTL':rntl_stat_dict}
        opts['valid_otherdata_dict'] = {'RNTL':rntl_stat_dict}
        opts['is_multi_years_combined'] = False
        opts['is_arid_one_hot'] = False


        opts['batch_size'] = 50
        opts['test_batch_size'] = 20
        opts['save_plt_epoches'] = 1
        opts['sample_interval'] = 50
        opts['save_epochs'] = 1
        opts['num_epochs'] = 4000
        opts['epochs_upLr'] = "5000,8000"
        opts['device_ids'] = device_ids
        run_train(epoch=epoch, model_name=model_name, model_name_appendix=model_name_appendix, seed=seed, opts=opts,
                  pathOpts=pathOpts)
    def Unet_01_Cfcvg(self):
        ###########  根据机器GPU使用情况设置device_ids
        # device_ids = [0,1]
        device_ids = [0]
        torch.cuda.set_device(0)
        ###########
        epoch = 0
        model_name = 'UNet'
        seed = 3000
        model_name_appendix = '01_Cfcvg'

        ###########设置参数
        pathOpts = {}
        opts={}
        opts['dmsp_stat_dict'] = getDMSP_stat_dict()
        opts['viirs_stat_dict'] = getVIIRS_stat_dict()
        cfcvg_stat_dict = getCfcvg_stat_dict()
        cfcvg_stat_dict['path'] = r'/home/zju/cxx/NTL_timeserise/data/SRCNN/data/img04/2012_CfCvg'
        opts['otherdata_dict'] = {'Cfcvg':cfcvg_stat_dict}
        opts['valid_otherdata_dict'] = {'Cfcvg':cfcvg_stat_dict}
        opts['is_multi_years_combined'] = False
        opts['is_arid_one_hot'] = False

        opts['batch_size'] = 50
        opts['test_batch_size'] = 20
        opts['save_plt_epoches'] = 1
        opts['sample_interval'] = 50
        opts['save_epochs'] = 1
        opts['num_epochs'] = 4000
        opts['epochs_upLr'] = "5000,8000"
        opts['device_ids'] = device_ids
        run_train(epoch=epoch, model_name=model_name, model_name_appendix=model_name_appendix, seed=seed, opts=opts,
                  pathOpts=pathOpts)
    def Unet_01_RNTL_Cfcvg(self):
        ###########  根据机器GPU使用情况设置device_ids
        # device_ids = [0,1]
        device_ids = [1]
        torch.cuda.set_device(1)
        ###########
        epoch = 0
        model_name = 'UNet'
        seed = 3000
        model_name_appendix = '01_RNTL_Cfcvg'

        ###########设置参数
        pathOpts = {}
        opts={}
        opts['dmsp_stat_dict'] = getDMSP_stat_dict()
        opts['viirs_stat_dict'] = getVIIRS_stat_dict()
        rntl_stat_dict = getRNTL_stat_dict()
        rntl_stat_dict['path'] = r'/home/zju/cxx/NTL_timeserise/data/SRCNN/data/img04/2010_RNTL'
        cfcvg_stat_dict = getCfcvg_stat_dict()
        cfcvg_stat_dict['path'] = r'/home/zju/cxx/NTL_timeserise/data/SRCNN/data/img04/2012_CfCvg'
        opts['otherdata_dict'] = {'RNTL':rntl_stat_dict,'Cfcvg':cfcvg_stat_dict}
        opts['valid_otherdata_dict'] = {'RNTL':rntl_stat_dict,'Cfcvg':cfcvg_stat_dict}
        opts['is_multi_years_combined'] = False
        opts['is_arid_one_hot'] = False


        opts['in_channels'] = 3
        opts['batch_size'] = 50
        opts['test_batch_size'] = 20
        opts['save_plt_epoches'] = 1
        opts['sample_interval'] = 50
        opts['save_epochs'] = 1
        opts['num_epochs'] = 4000
        opts['epochs_upLr'] = "5000,8000"
        opts['device_ids'] = device_ids
        run_train(epoch=epoch, model_name=model_name, model_name_appendix=model_name_appendix, seed=seed, opts=opts,
                  pathOpts=pathOpts)
    def Unet_01_noOtherData(self):
        ###########  根据机器GPU使用情况设置device_ids
        # device_ids = [0,1]
        device_ids = [0]
        torch.cuda.set_device(0)
        ###########
        epoch = 0
        model_name = 'UNet'
        seed = 3000
        model_name_appendix = '01_noOtherData'

        ###########设置参数
        pathOpts = {}
        opts={}
        opts['dmsp_stat_dict'] = getDMSP_stat_dict()
        opts['viirs_stat_dict'] = getVIIRS_stat_dict()
        opts['otherdata_dict'] = {}
        opts['valid_otherdata_dict'] = {}
        opts['is_multi_years_combined'] = False
        opts['is_arid_one_hot'] = False


        opts['in_channels'] = 1
        opts['batch_size'] = 50
        opts['test_batch_size'] = 20
        opts['save_plt_epoches'] = 1
        opts['sample_interval'] = 50
        opts['save_epochs'] = 1
        opts['num_epochs'] = 4000
        opts['epochs_upLr'] = "5000,8000"
        opts['device_ids'] = device_ids
        run_train(epoch=epoch, model_name=model_name, model_name_appendix=model_name_appendix, seed=seed, opts=opts,
                  pathOpts=pathOpts)
    def Unet_01_NDVI_Cfcvg_12_13(self):
        ###########  根据机器GPU使用情况设置device_ids
        # device_ids = [0,1]
        device_ids = [1]
        torch.cuda.set_device(1)
        ###########
        epoch = 0
        model_name = 'UNet'
        seed = 3000
        model_name_appendix = '01_NDVI_Cfcvg_12_13'

        ###########设置参数
        pathOpts = {}
        pathOpts['train_sample_txt_path'] = '/home/zju/cxx/NTL_timeserise/data/SRCNN/data/sample/01/04/train_08.txt'
        pathOpts['test_sample_txt_path'] = '/home/zju/cxx/NTL_timeserise/data/SRCNN/data/sample/01/04/test_08.txt'
        pathOpts['valid_sample_txt_lst'] = 'valid_08'
        pathOpts['label_dir_name'] = '2012_2013VNL'
        pathOpts['inputdata_dir_name'] = '2012_2013oriDNL'
        pathOpts['ndvidata_dir_name'] = '2012_2013NDVI'
        pathOpts['valid_label_dir_name'] = '2012_2013VNL'
        pathOpts['valid_inputdata_dir_name'] = '2012_2013oriDNL'
        pathOpts['valid_ndvidata_dir_name'] = '2012_2013NDVI'

        opts={}
        opts['dmsp_stat_dict'] = getDMSP_stat_dict()
        opts['viirs_stat_dict'] = getVIIRS_stat_dict()
        ndvi_stat_dict = getNDVI_stat_dict()
        ndvi_stat_dict['path'] = r'/home/zju/cxx/NTL_timeserise/data/SRCNN/data/img04/2012_2013NDVI'
        cfcvg_stat_dict = getCfcvg_stat_dict()
        cfcvg_stat_dict['path'] = r'/home/zju/cxx/NTL_timeserise/data/SRCNN/data/img04/2012_2013CfCvg'
        opts['otherdata_dict'] = {'NDVI':ndvi_stat_dict,'Cfcvg':cfcvg_stat_dict}
        opts['valid_otherdata_dict'] = {'NDVI':ndvi_stat_dict,'Cfcvg':cfcvg_stat_dict}
        opts['is_multi_years_combined'] = True
        opts['is_arid_one_hot'] = False

        opts['in_channels'] = 3
        opts['batch_size'] = 50
        opts['test_batch_size'] = 20
        opts['save_plt_epoches'] = 1
        opts['sample_interval'] = 50
        opts['save_epochs'] = 1
        opts['num_epochs'] = 4000
        opts['epochs_upLr'] = "5000,8000"
        opts['device_ids'] = device_ids
        run_train(epoch=epoch, model_name=model_name, model_name_appendix=model_name_appendix, seed=seed, opts=opts,
                  pathOpts=pathOpts)
    def Unet_01_labelNoProcess_12_13(self):
        ###########  根据机器GPU使用情况设置device_ids
        # device_ids = [0,1]
        device_ids = [1]
        torch.cuda.set_device(1)
        ###########
        epoch = 0
        model_name = 'UNet'
        seed = 3000
        model_name_appendix = 'labelNoProcess_12_13'

        ###########设置参数
        pathOpts = {}
        pathOpts['train_sample_txt_path'] = '/home/zju/cxx/NTL_timeserise/data/SRCNN/data/sample/01/04/train_08.txt'
        pathOpts['test_sample_txt_path'] = '/home/zju/cxx/NTL_timeserise/data/SRCNN/data/sample/01/04/test_08.txt'
        pathOpts['valid_sample_txt_lst'] = 'valid_08'
        pathOpts['label_dir_name'] = '2012_2013VNL'
        pathOpts['inputdata_dir_name'] = '2012_2013oriDNL'
        pathOpts['ndvidata_dir_name'] = '2012_2013NDVI'
        pathOpts['valid_label_dir_name'] = '2012_2013VNL'
        pathOpts['valid_inputdata_dir_name'] = '2012_2013oriDNL'
        pathOpts['valid_ndvidata_dir_name'] = '2012_2013NDVI'

        opts = {}
        opts['dmsp_stat_dict'] = getDMSP_stat_dict()
        opts['viirs_stat_dict'] = getVIIRS_stat_dict(normlizedtype=None)
        opts['label_use_ln'] = False
        ndvi_stat_dict = getNDVI_stat_dict()
        ndvi_stat_dict['path'] = r'/home/zju/cxx/NTL_timeserise/data/SRCNN/data/img04/2012_2013NDVI'
        opts['otherdata_dict'] = {'NDVI': ndvi_stat_dict}
        opts['valid_otherdata_dict'] = {'NDVI': ndvi_stat_dict}
        opts['is_multi_years_combined'] = True
        opts['is_arid_one_hot'] = False


        opts['batch_size'] = 50
        opts['test_batch_size'] = 50
        opts['save_plt_epoches'] = 1
        opts['sample_interval'] = 1
        opts['save_epochs'] = 1
        opts['num_epochs'] = 4000
        opts['epochs_upLr'] = "5000,8000"
        opts['device_ids'] = device_ids
        run_train(epoch=epoch, model_name=model_name, model_name_appendix=model_name_appendix, seed=seed, opts=opts,
                  pathOpts=pathOpts)

    def Unet_01_labelNoProcess_12_13_hyperarid(self):
        ###########  根据机器GPU使用情况设置device_ids
        # device_ids = [0,1]
        device_ids = [0]
        torch.cuda.set_device(0)
        ###########
        epoch = 4394
        model_name = 'UNet'
        seed = 3000
        model_name_appendix = 'labelNoProcess_12_13'

        ###########设置参数
        pathOpts = {}
        pathOpts['train_sample_txt_path'] = '/home/zju/cxx/NTL_timeserise/data/SRCNN/data/sample/01/04/train_09_Hyperarid.txt'
        pathOpts['test_sample_txt_path'] = '/home/zju/cxx/NTL_timeserise/data/SRCNN/data/sample/01/04/test_09_Hyperarid.txt'
        pathOpts['valid_sample_txt_lst'] = 'valid_09_Hyperarid'
        pathOpts['label_dir_name'] = '2012_2013VNL'
        pathOpts['inputdata_dir_name'] = '2012_2013oriDNL'
        pathOpts['ndvidata_dir_name'] = '2012_2013NDVI'
        pathOpts['valid_label_dir_name'] = '2012_2013VNL'
        pathOpts['valid_inputdata_dir_name'] = '2012_2013oriDNL'
        pathOpts['valid_ndvidata_dir_name'] = '2012_2013NDVI'

        opts = {}
        opts['dmsp_stat_dict'] = getDMSP_stat_dict()
        opts['viirs_stat_dict'] = getVIIRS_stat_dict(normlizedtype=None)
        opts['label_use_ln'] = False
        ndvi_stat_dict = getNDVI_stat_dict()
        ndvi_stat_dict['path'] = r'/home/zju/cxx/NTL_timeserise/data/SRCNN/data/img04/2012_2013NDVI'
        opts['otherdata_dict'] = {'NDVI': ndvi_stat_dict}
        opts['valid_otherdata_dict'] = {'NDVI': ndvi_stat_dict}
        opts['is_multi_years_combined'] = True
        opts['is_arid_one_hot'] = False

        opts['learning_rate'] = 1e-7

        opts['batch_size'] = 50
        opts['test_batch_size'] = 10
        opts['save_plt_epoches'] = 1
        opts['sample_interval'] = 50
        opts['save_epochs'] = 1
        opts['num_epochs'] = 6000
        opts['epochs_upLr'] = "8000"
        opts['device_ids'] = device_ids
        run_train(epoch=epoch, model_name=model_name, model_name_appendix=model_name_appendix, seed=seed, opts=opts,
                  pathOpts=pathOpts)
    def Unet_01_labelNoProcess_12_13_rarid_without_drySubhumid(self):
        ###########  根据机器GPU使用情况设置device_ids
        # device_ids = [0,1]
        device_ids = [0]
        torch.cuda.set_device(0)
        ###########
        epoch = 1571
        model_name = 'UNet'
        seed = 3000
        model_name_appendix = 'labelNoProcess_12_13'

        ###########设置参数
        pathOpts = {}
        pathOpts['train_sample_txt_path'] = '/home/zju/cxx/NTL_timeserise/data/SRCNN/data/sample/01/04/train_09_Aridity_without_drySubhumid.txt'
        pathOpts['test_sample_txt_path'] = '/home/zju/cxx/NTL_timeserise/data/SRCNN/data/sample/01/04/test_09_Aridity_without_drySubhumid.txt'
        pathOpts['valid_sample_txt_lst'] = 'valid_09_Aridity_without_drySubhumid'
        pathOpts['label_dir_name'] = '2012_2013VNL'
        pathOpts['inputdata_dir_name'] = '2012_2013oriDNL'
        pathOpts['ndvidata_dir_name'] = '2012_2013NDVI'
        pathOpts['valid_label_dir_name'] = '2012_2013VNL'
        pathOpts['valid_inputdata_dir_name'] = '2012_2013oriDNL'
        pathOpts['valid_ndvidata_dir_name'] = '2012_2013NDVI'

        opts = {}
        opts['dmsp_stat_dict'] = getDMSP_stat_dict()
        opts['viirs_stat_dict'] = getVIIRS_stat_dict(normlizedtype=None)
        opts['label_use_ln'] = False
        ndvi_stat_dict = getNDVI_stat_dict()
        ndvi_stat_dict['path'] = r'/home/zju/cxx/NTL_timeserise/data/SRCNN/data/img04/2012_2013NDVI'
        opts['otherdata_dict'] = {'NDVI': ndvi_stat_dict}
        opts['valid_otherdata_dict'] = {'NDVI': ndvi_stat_dict}
        opts['is_multi_years_combined'] = True
        opts['is_arid_one_hot'] = False

        opts['learning_rate'] = 1e-7

        opts['batch_size'] = 50
        opts['test_batch_size'] = 50
        opts['save_plt_epoches'] = 1
        opts['sample_interval'] = 50
        opts['save_epochs'] = 1
        opts['num_epochs'] = 5000
        opts['epochs_upLr'] = "8000"
        opts['device_ids'] = device_ids
        run_train(epoch=epoch, model_name=model_name, model_name_appendix=model_name_appendix, seed=seed, opts=opts,
                  pathOpts=pathOpts)
    def Unet_01_rarid_without_drySubhumid(self):
        ###########  根据机器GPU使用情况设置device_ids
        # device_ids = [0,1]
        device_ids = [1]
        torch.cuda.set_device(1)
        ###########
        epoch = 0
        model_name = 'UNet'
        seed = 3000
        model_name_appendix = '01_rarid_without_drySubhumid'

        ###########设置参数
        pathOpts = {}
        pathOpts[
            'train_sample_txt_path'] = '/home/zju/cxx/NTL_timeserise/data/SRCNN/data/sample/01/04/train_09_Aridity_without_drySubhumid.txt'
        pathOpts[
            'test_sample_txt_path'] = '/home/zju/cxx/NTL_timeserise/data/SRCNN/data/sample/01/04/test_09_Aridity_without_drySubhumid.txt'
        pathOpts['valid_sample_txt_lst'] = 'valid_09_Aridity_without_drySubhumid'
        pathOpts['label_dir_name'] = '2012_2013VNL'
        pathOpts['inputdata_dir_name'] = '2012_2013oriDNL'
        pathOpts['ndvidata_dir_name'] = '2012_2013NDVI'
        pathOpts['valid_label_dir_name'] = '2012_2013VNL'
        pathOpts['valid_inputdata_dir_name'] = '2012_2013oriDNL'
        pathOpts['valid_ndvidata_dir_name'] = '2012_2013NDVI'

        opts = {}
        opts['is_multi_years_combined'] = True
        opts['is_arid_one_hot'] = False
        opts['dmsp_stat_dict'] = getDMSP_stat_dict()
        opts['viirs_stat_dict'] = getVIIRS_stat_dict(normlizedtype=None)
        opts['label_use_ln'] = False
        ndvi_stat_dict = getNDVI_stat_dict()
        ndvi_stat_dict['path'] = r'/home/zju/cxx/NTL_timeserise/data/SRCNN/data/img04/2012_2013NDVI'
        opts['otherdata_dict'] = {'NDVI': ndvi_stat_dict}
        opts['valid_otherdata_dict'] = {'NDVI': ndvi_stat_dict}

        opts['batch_size'] = 50
        opts['test_batch_size'] = 50
        opts['save_plt_epoches'] = 1
        opts['sample_interval'] = 50
        opts['save_epochs'] = 1
        opts['num_epochs'] = 4000
        opts['epochs_upLr'] = "5000,8000"
        opts['device_ids'] = device_ids
        run_train(epoch=epoch, model_name=model_name, model_name_appendix=model_name_appendix, seed=seed, opts=opts,
                  pathOpts=pathOpts)
    def Unet_01_rarid(self):
        ###########  根据机器GPU使用情况设置device_ids
        # device_ids = [0,1]
        device_ids = [0]
        torch.cuda.set_device(0)
        ###########
        epoch = 0
        model_name = 'UNet'
        seed = 3000
        model_name_appendix = '01_rarid'

        ###########设置参数
        pathOpts = {}
        pathOpts[
            'train_sample_txt_path'] = '/home/zju/cxx/NTL_timeserise/data/SRCNN/data/sample/01/04/train_09_Aridity.txt'
        pathOpts[
            'test_sample_txt_path'] = '/home/zju/cxx/NTL_timeserise/data/SRCNN/data/sample/01/04/test_09_Aridity.txt'
        pathOpts['valid_sample_txt_lst'] = 'valid_09_Aridity'
        pathOpts['label_dir_name'] = '2012_2013VNL'
        pathOpts['inputdata_dir_name'] = '2012_2013oriDNL'
        pathOpts['ndvidata_dir_name'] = '2012_2013NDVI'
        pathOpts['valid_label_dir_name'] = '2012_2013VNL'
        pathOpts['valid_inputdata_dir_name'] = '2012_2013oriDNL'
        pathOpts['valid_ndvidata_dir_name'] = '2012_2013NDVI'

        opts = {}
        opts['dmsp_stat_dict'] = getDMSP_stat_dict()
        opts['viirs_stat_dict'] = getVIIRS_stat_dict(normlizedtype=None)
        opts['label_use_ln'] = False
        ndvi_stat_dict = getNDVI_stat_dict()
        ndvi_stat_dict['path'] = r'/home/zju/cxx/NTL_timeserise/data/SRCNN/data/img04/2012_2013NDVI'
        opts['otherdata_dict'] = {'NDVI': ndvi_stat_dict}
        opts['valid_otherdata_dict'] = {'NDVI': ndvi_stat_dict}
        opts['is_multi_years_combined'] = True
        opts['is_arid_one_hot'] = False

        opts['batch_size'] = 40
        opts['test_batch_size'] = 40
        opts['save_plt_epoches'] = 1
        opts['sample_interval'] = 50
        opts['save_epochs'] = 1
        opts['num_epochs'] = 4000
        opts['epochs_upLr'] = "5000,8000"
        opts['device_ids'] = device_ids
        run_train(epoch=epoch, model_name=model_name, model_name_appendix=model_name_appendix, seed=seed, opts=opts,
                  pathOpts=pathOpts)
    def Unet_01_rarid_class(self):
        ###########  根据机器GPU使用情况设置device_ids
        # device_ids = [0,1]
        device_ids = [1]
        torch.cuda.set_device(1)
        ###########
        epoch = 0
        model_name = 'UNet'
        seed = 3000
        model_name_appendix = '01_rarid_class'

        ###########设置参数
        pathOpts = {}
        pathOpts['train_sample_txt_path'] = '/home/zju/cxx/NTL_timeserise/data/SRCNN/data/sample/01/04/train_08.txt'
        pathOpts['test_sample_txt_path'] = '/home/zju/cxx/NTL_timeserise/data/SRCNN/data/sample/01/04/test_08.txt'
        pathOpts['valid_sample_txt_lst'] = 'valid_08'
        pathOpts['label_dir_name'] = '2012_2013VNL'
        pathOpts['inputdata_dir_name'] = '2012_2013oriDNL'
        pathOpts['ndvidata_dir_name'] = '2012_2013NDVI'
        pathOpts['valid_label_dir_name'] = '2012_2013VNL'
        pathOpts['valid_inputdata_dir_name'] = '2012_2013oriDNL'
        pathOpts['valid_ndvidata_dir_name'] = '2012_2013NDVI'

        opts = {}
        opts['dmsp_stat_dict'] = getDMSP_stat_dict()
        opts['viirs_stat_dict'] = getVIIRS_stat_dict(normlizedtype=None)
        opts['label_use_ln'] = False
        ndvi_stat_dict = getNDVI_stat_dict()
        ndvi_stat_dict['path'] = r'/home/zju/cxx/NTL_timeserise/data/SRCNN/data/img04/2012_2013NDVI'
        arid_stat_dict = getWater_Aridity_stat_dict()
        arid_stat_dict['path'] = '/home/zju/cxx/NTL_timeserise/data/SRCNN/data/img04/Water_Aridity'
        opts['otherdata_dict'] = {'NDVI': ndvi_stat_dict,'Water_Aridity':arid_stat_dict}
        opts['valid_otherdata_dict'] = {'NDVI': ndvi_stat_dict,'Water_Aridity':arid_stat_dict}
        opts['is_multi_years_combined'] = True
        opts['is_arid_one_hot'] = False
        opts['in_channels'] = 3


        opts['batch_size'] = 40
        opts['test_batch_size'] = 40
        opts['save_plt_epoches'] = 1
        opts['sample_interval'] = 50
        opts['save_epochs'] = 1
        opts['num_epochs'] = 4000
        opts['epochs_upLr'] = "5000,8000"
        opts['device_ids'] = device_ids
        run_train(epoch=epoch, model_name=model_name, model_name_appendix=model_name_appendix, seed=seed, opts=opts,
                  pathOpts=pathOpts)
    def Unet_01_rarid_onehot(self):
        ###########  根据机器GPU使用情况设置device_ids
        # device_ids = [0,1]
        device_ids = [0]
        torch.cuda.set_device(0)
        ###########
        epoch = 0
        model_name = 'UNet'
        seed = 3000
        model_name_appendix = '01_rarid_onehot'

        ###########设置参数
        pathOpts = {}
        pathOpts['train_sample_txt_path'] = '/home/zju/cxx/NTL_timeserise/data/SRCNN/data/sample/01/04/train_08.txt'
        pathOpts['test_sample_txt_path'] = '/home/zju/cxx/NTL_timeserise/data/SRCNN/data/sample/01/04/test_08.txt'
        pathOpts['valid_sample_txt_lst'] = 'valid_08'
        pathOpts['label_dir_name'] = '2012_2013VNL'
        pathOpts['inputdata_dir_name'] = '2012_2013oriDNL'
        pathOpts['ndvidata_dir_name'] = '2012_2013NDVI'
        pathOpts['valid_label_dir_name'] = '2012_2013VNL'
        pathOpts['valid_inputdata_dir_name'] = '2012_2013oriDNL'
        pathOpts['valid_ndvidata_dir_name'] = '2012_2013NDVI'

        opts = {}
        opts['dmsp_stat_dict'] = getDMSP_stat_dict()
        opts['viirs_stat_dict'] = getVIIRS_stat_dict(normlizedtype=None)
        opts['label_use_ln'] = False
        ndvi_stat_dict = getNDVI_stat_dict()
        ndvi_stat_dict['path'] = r'/home/zju/cxx/NTL_timeserise/data/SRCNN/data/img04/2012_2013NDVI'
        arid_stat_dict = getWater_Aridity_stat_dict()
        arid_stat_dict['path'] = '/home/zju/cxx/NTL_timeserise/data/SRCNN/data/img04/Water_Aridity'
        opts['otherdata_dict'] = {'NDVI': ndvi_stat_dict,'Water_Aridity':arid_stat_dict}
        opts['valid_otherdata_dict'] = {'NDVI': ndvi_stat_dict,'Water_Aridity':arid_stat_dict}
        opts['is_multi_years_combined'] = True
        opts['is_arid_one_hot'] = True
        opts['in_channels'] = 7

        opts['batch_size'] = 20
        opts['test_batch_size'] = 20
        opts['save_plt_epoches'] = 1
        opts['sample_interval'] = 50
        opts['save_epochs'] = 1
        opts['num_epochs'] = 4000
        opts['epochs_upLr'] = "5000,8000"
        opts['device_ids'] = device_ids
        run_train(epoch=epoch, model_name=model_name, model_name_appendix=model_name_appendix, seed=seed, opts=opts,
                  pathOpts=pathOpts)


    def Unet_02(self):
        ###########  根据机器GPU使用情况设置device_ids
        # device_ids = [0,1]
        device_ids = [0]
        # torch.cuda.set_device(1)
        ###########
        epoch = 0
        model_name = 'UNet'
        seed = 3000
        model_name_appendix = '02'

        ###########设置参数
        pathOpts = {}
        pathOpts['train_sample_txt_path'] = '/home/zju/cxx/NTL_timeserise/data/SRCNN/data/sample/01/04/train_08.txt'
        pathOpts['test_sample_txt_path'] = '/home/zju/cxx/NTL_timeserise/data/SRCNN/data/sample/01/04/test_08.txt'
        pathOpts['valid_sample_txt_lst'] = 'valid_08'

        opts = {}
        opts['base_filter'] = 64
        opts['dmsp_stat_dict'] = getDMSP_stat_dict()
        opts['viirs_stat_dict'] = getVIIRS_stat_dict(normlizedtype=None)
        opts['label_use_ln'] = False
        ndvi_stat_dict = getNDVI_stat_dict()
        ndvi_stat_dict['path'] = r'/home/zju/cxx/NTL_timeserise/data/SRCNN/data/img04/2012NDVI'
        opts['otherdata_dict'] = {'NDVI': ndvi_stat_dict}
        opts['valid_otherdata_dict'] = {'NDVI': ndvi_stat_dict}
        opts['is_multi_years_combined'] = True
        opts['is_arid_one_hot'] = False

        opts['batch_size'] = 10
        opts['test_batch_size'] = 10
        opts['save_plt_epoches'] = 1
        opts['sample_interval'] = 1
        opts['save_epochs'] = 1
        opts['num_epochs'] = 4000
        opts['epochs_upLr'] = "5000,8000"
        opts['device_ids'] = device_ids
        run_train(epoch=epoch, model_name=model_name, model_name_appendix=model_name_appendix, seed=seed, opts=opts,
                  pathOpts=pathOpts)

runModel = RunModel()
# runModel.DNLSRNet_2012_01()
# runModel.Unet_01()




