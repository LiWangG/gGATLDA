import time
import os
import math
import multiprocessing as mp
import numpy as np
import networkx as nx
import torch
import torch.nn.functional as F
from torch import tensor
from torch.optim import Adam
from sklearn.model_selection import StratifiedKFold
from torch_geometric.data import DataLoader, DenseDataLoader as DenseLoader
from tqdm import tqdm
import pdb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from util_functions import PyGGraph_to_nx
from sklearn.metrics import roc_auc_score,accuracy_score
from sklearn import metrics
import random
from ranger import Ranger
from ranger import RangerVA
from ranger import RangerQH
from torch.optim import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_k_fold_data(k,i,data):
    assert k>1
    data_pos=data[0:621]
    data_neg=data[621:1242]

    start=int(i*621//k)
    end=int((i+1)*621//k)

    data_train, data_valid=None, None
    data_valid_pos, data_valid_neg=None, None
    data_train_pos, data_train_neg=None, None

    data_valid_pos=data_pos[start:end]
    data_train_pos=data_pos[0:start]+data_pos[end:621]
    data_valid_neg=data_neg[start:end]
    data_train_neg=data_neg[0:start]+data_neg[end:621]
    data_train=data_train_pos+data_train_neg
    data_valid=data_valid_pos+data_valid_neg
    return data_train,data_valid



def train_multiple_epochs(train_graphs, test_graphs, model,  adj):
    train_loss,test_loss=[],[] 

    print("starting train...")
    LR=0.01
    batch_size=64
    epochs=50
    train_loader = DataLoader(train_graphs, batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_graphs, batch_size, shuffle=True, num_workers=0)
    optimizer = Ranger(model.parameters(), lr=0.001, weight_decay=0)
    start_epoch = 1
    pbar = tqdm(range(start_epoch, epochs + start_epoch))
    count=0

    for epoch in pbar:
        total_loss=0
        
        
        model.train()
        for data in train_loader:
            optimizer.zero_grad()
            #data=data.cuda()
            out = model(data)
            loss=F.cross_entropy(out, data.y.view(-1).long())
            loss.backward()
            total_loss+=loss.item()*num_graphs(data)
            optimizer.step()
        train_loss=total_loss/len(train_loader.dataset)
        train_auc=evaluate(model,train_loader,1)
        print('\n Epoch: {:03d}, Loss: {:.5f}, Train Auc: {:.5f}'.format(epoch, train_loss, train_auc))

    
    test_auc,one_pred_result=evaluate(model,test_loader,2)
    truth=one_pred_result[1]
    predict=one_pred_result[0]
    vmax=max(predict)
    vmin=min(predict)

    alpha=0.8
    predict_f1=[0 for x in range(len(predict))]
    for p in range(len(predict)):
        predict_f1[p]=(predict[p]-vmin)/(vmax-vmin)
    predict_f1=[int(item>alpha) for item in predict_f1]
    
    f1=metrics.f1_score(truth,predict_f1)
    accuracy=metrics.accuracy_score(truth,predict_f1)
    recall=metrics.recall_score(truth,predict_f1)
    precision=metrics.precision_score(truth,predict_f1)    
    fpr,tpr, thresholds1=metrics.roc_curve(truth,predict,pos_label=1)
    auc_score=metrics.auc(fpr,tpr)
    p,r,thresholds2=metrics.precision_recall_curve(truth,predict,pos_label=1)
    aupr_score=metrics.auc(r,p)
    print('f1:',f1)
    print('accuracy:',accuracy)
    print('recall:',recall)
    print('precision:',precision)
    print('auc:',auc_score)
    print('aupr:',aupr_score)
    print('test_auc:',test_auc)
    return test_auc, f1,accuracy,recall,precision,auc_score,aupr_score,truth,predict
 
def evaluate(model,loader,flag):
    one_pred_result=[]
    model.eval()
    predictions=torch.Tensor()
    labels=torch.Tensor()
    with torch.no_grad():
        for data in loader:
            data=data.to(device)
            pred=model(data)
            #predictions.append(pred[:,1].cpu().detach())
            
            predictions=torch.cat((predictions,pred[:,1].detach()),0)
            labels=torch.cat((labels,data.y),0)
    
    #labels=labels.cuda().data.cpu().numpy()
    #predictions=predictions.cuda().data.cpu().numpy()
    if flag==2:
        one_pred_result=np.vstack((predictions,labels))

    fpr,tpr,_=metrics.roc_curve(labels,predictions,pos_label=1)
    auc=metrics.auc(fpr,tpr)
    if flag==1:
        return auc
    else:
        return auc,one_pred_result;


def num_graphs(data):
    if data.batch is not None:
        return data.num_graphs
    else:
        return data.x.size(0)




