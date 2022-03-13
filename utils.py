import os
import sys
import json
import pickle
import random
import numpy as np
import torch
from tqdm import tqdm
from loss import Contrast_depth_loss
from sklearn.metrics import roc_curve
import pandas as pd

def enable_dropout(model):
    """ Function to enable the dropout layers during test-time """
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout') or m.__class__.__name__.startswith('Block'):
            m.train()


def get_err_threhold(fpr, tpr, threshold):
    RightIndex=(tpr+(1-fpr)-1); 
    right_index = np.argmax(RightIndex)
    best_th = threshold[right_index]
    err = fpr[right_index]

    differ_tpr_fpr_1=tpr+fpr-1.0
  
    right_index = np.argmin(np.abs(differ_tpr_fpr_1))
    best_th = threshold[right_index]
    err = fpr[right_index]    

    #print(err, best_th)
    return err, best_th


def train_one_epoch(model, optimizer_Adam,optimizer_SGD, data_loader, epoch, train_stage):
    model.train()

    criterion_absolute_loss = torch.nn.MSELoss().cuda()
    criterion_contrastive_loss = Contrast_depth_loss().cuda()

    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).cuda()  
    accu_num = torch.zeros(1).cuda()   

    optimizer_Adam.zero_grad()
    optimizer_SGD.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader)
    for step, sample_batched in enumerate(data_loader):
        inputs = sample_batched['image_x'].cuda()
        map_label= sample_batched['map_x'].cuda()
        spoof_label =sample_batched['spoofing_label'].cuda() 
        sample_num += inputs.shape[0]

        pred,depth = model(inputs)
        absolute_loss    = criterion_absolute_loss(depth, map_label)
        contrastive_loss = criterion_contrastive_loss(depth, map_label)

        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, spoof_label).sum()

        if train_stage==1:
            loss = absolute_loss + contrastive_loss 
            loss.backward()
        elif train_stage == 2:
            loss = loss_function(pred, spoof_label)
            loss.backward()
        else:
            loss_dccdn = absolute_loss + contrastive_loss 
            loss= 0.2*loss_dccdn + 0.8*(loss_function(pred, spoof_label))
            # loss= loss_function(pred, spoof_label)
            loss.backward()

        accu_loss += loss.detach()
        data_loader.desc = "[train epoch {} absolute_loss:{:.10f}, contrastive_loss:{:.10f}, Vit loss: {:.3f}, acc: {:.3f}]".format(epoch,absolute_loss.item(),contrastive_loss.item(),accu_loss.item() / (step + 1),accu_num.item() / sample_num)

        optimizer_Adam.step()
        optimizer_Adam.zero_grad()
        optimizer_SGD.step()
        optimizer_SGD.zero_grad()

    return absolute_loss.item(),contrastive_loss.item(),accu_loss,accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader_val, data_loader_test,epoch,train_stage,sampling_size):

    model.eval()

    sample_num = 0
    T = 30
    data_loader_val = tqdm(data_loader_val)


    val_scores_cdcn = []
    val_labels_cdcn = []
    val_scores_vit = []
    val_labels_vit = []
    for step, sample_batched in enumerate(data_loader_val):

        inputs = sample_batched['image_x'].cuda() 
        spoof_label =sample_batched['spoofing_label'].cuda() 
        sample_num += inputs.shape[0]
        map_score_cdcn = 0
        map_score_vit = 0
        for frame_t in np.arange(sampling_size) :
            pred,map_x= model(inputs[:, frame_t, ...])
            map_score_cdcn += torch.mean(map_x, axis=[1,2])

            out = torch.nn.functional.softmax(pred,dim=1)
            map_score_vit += out

        map_score_cdcn = map_score_cdcn / sampling_size
        map_score_vit = map_score_vit / sampling_size

        for batch_id in np.arange(map_score_cdcn.shape[0]):
            val_scores_cdcn.append(map_score_cdcn[batch_id].item())
            val_labels_cdcn.append(spoof_label[batch_id].item())
            val_scores_vit.append(map_score_vit[batch_id][1].item())
            val_labels_vit.append(spoof_label[batch_id].item())

    ## star
    fpr,tpr,threshold = roc_curve(val_labels_cdcn, val_scores_cdcn, pos_label=1)
    val_err_cdcn, val_threshold_cdcn = get_err_threhold(fpr, tpr, threshold)
    fpr,tpr,threshold = roc_curve(val_labels_vit, val_scores_vit, pos_label=1)
    val_err_vit, val_threshold_vit = get_err_threhold(fpr, tpr, threshold)
    # print("val_err_cdcn = {} val_threshold_cdcn = {}".format(val_err_cdcn,val_threshold_cdcn))
    # print("val_err_vit = {} val_threshold_vit = {}".format(val_err_vit,val_threshold_vit))


    data_loader_test = tqdm(data_loader_test)
    test_scores_cdcn = []
    test_labels_cdcn = []
    test_scores_vit = []
    test_labels_vit = []
    data_cdcn = []
    data_vit = []
    for step, sample_batched in enumerate(data_loader_test):

        inputs = sample_batched['image_x'].cuda() 
        spoof_label =sample_batched['spoofing_label'].cuda() 
        sample_num += inputs.shape[0]
        map_score_cdcn = 0
        map_score_vit = 0
        for frame_t in np.arange(sampling_size) :
            pred,map_x = model(inputs[:, frame_t, ...])
            map_score_cdcn += torch.mean(map_x, axis=[1,2])

            out = torch.nn.functional.softmax(pred,dim=1)
            map_score_vit += out

        map_score_cdcn = map_score_cdcn / sampling_size
        map_score_vit = map_score_vit / sampling_size

        for batch_id in np.arange(map_score_cdcn.shape[0]):
            test_scores_cdcn.append(map_score_cdcn[batch_id].item())
            test_labels_cdcn.append(spoof_label[batch_id].item())
            test_scores_vit.append(map_score_vit[batch_id][1].item())
            test_labels_vit.append(spoof_label[batch_id].item())
            data_cdcn.append({'map_score': map_score_cdcn[batch_id].item(),'label': spoof_label[batch_id].item()})
            data_vit.append({'map_score': map_score_vit[batch_id][1].item(),'label': spoof_label[batch_id].item()})
    ## star
    # fpr,tpr,threshold = roc_curve(test_labels_cdcn, test_scores_cdcn, pos_label=1)
    # test_err_cdcn, test_threshold_cdcn = get_err_threhold(fpr, tpr, threshold)
    # fpr,tpr,threshold = roc_curve(test_labels_vit, test_scores_vit, pos_label=1)
    # test_err_vit, test_threshold_vit = get_err_threhold(fpr, tpr, threshold)
    # print("test_err_cdcn = {} test_threshold_cdcn = {}".format(test_err_cdcn,test_threshold_cdcn))
    # print("test_err_vit = {} test_threshold_vit = {}".format(test_err_vit,test_threshold_vit))
    count = 0
    num_real = 0
    num_fake = 0
    for item in data_cdcn:
        count+=1
        if item['label']==1:
            num_real += 1
        else:
            num_fake += 1
    type1_cdcn = len([s for s in data_cdcn if s['map_score'] <= val_threshold_cdcn and s['label'] == 1])
    type2_cdcn = len([s for s in data_cdcn if s['map_score'] > val_threshold_cdcn and s['label'] == 0])

    type1_vit= len([s for s in data_vit if s['map_score'] <= val_threshold_vit and s['label'] == 1])
    type2_vit= len([s for s in data_vit if s['map_score'] > val_threshold_vit and s['label'] == 0])

    val_ACC_cdcn = 1-(type1_cdcn + type2_cdcn) / count
    val_APCER_cdcn = type2_cdcn / num_fake
    val_BPCER_cdcn = type1_cdcn / num_real
    val_ACER_cdcn = (val_APCER_cdcn + val_BPCER_cdcn) / 2.0
    print('CDCNpp Result: \nval_err = {} \nval_threshold = {} \nACC = {} \nAPCER = {} \nBPCER = {} \nACER = {}'.format(val_err_cdcn,val_threshold_cdcn,val_ACC_cdcn,val_APCER_cdcn,val_BPCER_cdcn,val_ACER_cdcn))

    val_ACC_vit= 1-(type1_vit+ type2_vit) / count
    val_APCER_vit= type2_vit/ num_fake
    val_BPCER_vit= type1_vit/ num_real
    val_ACER_vit= (val_APCER_vit+ val_BPCER_vit) / 2.0
    print('ViT Result: \nval_err = {} \nval_threshold = {} \nACC = {} \nAPCER = {} \nBPCER = {} \nACER = {}'.format(val_err_vit,val_threshold_vit,val_ACC_vit,val_APCER_vit,val_BPCER_vit,val_ACER_vit))




        # loss = loss_function(pred, spoof_label)
        # accu_loss += loss

        # data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
        #                                                                        accu_loss.item() / (step + 1),
        #                                                                        accu_num.item() / sample_num)

    return val_ACC_cdcn,val_ACC_vit,val_APCER_cdcn,val_APCER_vit,val_BPCER_cdcn,val_BPCER_vit,val_ACER_cdcn,val_ACER_vit

