from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.sparse as sp
import scipy.io as scio
import pickle as pkl
import os
import h5py
import pandas as pd
import random
import pdb
import math
from random import randint,sample
from sklearn.model_selection import KFold

def load_data(dataset):
    print("Loading lncRNAdisease dataset")
    path_dataset = 'raw_data/' + dataset + '/training_test_dataset.mat'
    data=scio.loadmat(path_dataset)
    net=data['interMatrix']
   
    #lncRNA features and disease features
    u_features=data['lncSim']
    disSim_path='raw_data/' + dataset + '/disSim.xlsx'   
    disSim_data=pd.read_excel(disSim_path,header=0)
    v_features=np.array(disSim_data)

    num_list=[len(u_features)]
    num_list.append(len(v_features))
    temp=np.zeros((net.shape[0],net.shape[1]),int)     
    u_features=np.hstack((u_features,net))
    v_features=np.hstack((net.T,v_features))
    
    a=np.zeros((1,u_features.shape[0]+v_features.shape[0]),int)
    b=np.zeros((1,v_features.shape[0]+u_features.shape[0]),int)
    u_features=np.vstack((a,u_features))
    v_features=np.vstack((b,v_features))

    num_lncRNAs=net.shape[0]
    num_diseases=net.shape[1]
    
    row,col,_=sp.find(net)
    perm=random.sample(range(len(row)),len(row))
    row,col=row[perm],col[perm]
    sample_pos=(row,col)
    print("the number of all positive sample:",len(sample_pos[0]))

    print("sampling negative links for train and test")
    sample_neg=([],[])
    net_flag=np.zeros((net.shape[0],net.shape[1]))
    X=np.ones((num_lncRNAs,num_diseases))
    net_neg=X-net
    row_neg,col_neg,_=sp.find(net_neg)
    perm_neg=random.sample(range(len(row_neg)),len(row))
    row_neg,col_neg=row_neg[perm_neg],col_neg[perm_neg]
    sample_neg=(row_neg,col_neg)
    sample_neg=list(sample_neg)
    print("the number of all negative sample:", len(sample_neg[0]))
    	
    u_idx = np.hstack([sample_pos[0], sample_neg[0]])
    v_idx = np.hstack([sample_pos[1], sample_neg[1]])
    labels= np.hstack([[1]*len(sample_pos[0]), [0]*len(sample_neg[0])])
     
    l1=np.zeros((1,net.shape[1]),int)
    print(l1.shape)
    net=np.vstack([l1,net])
    print("old net:",net.shape)
    l2=np.zeros((net.shape[0],1),int)
    net=np.hstack([l2,net])
    print("new net:",net.shape)

    u_idx=u_idx+1
    v_idx=v_idx+1	

    return u_features, v_features, net, labels, u_idx, v_idx,num_list

def load_predict_data(dataset):
    print("Loading lncRNAdisease dataset")
    path_dataset = 'raw_data/' + dataset + '/training_test_dataset.mat'
    data=scio.loadmat(path_dataset)
    net=data['interMatrix']
    num_lncRNAs=net.shape[0]
    num_diseases=net.shape[1]

    net_new=np.zeros((num_lncRNAs+1,num_diseases+1),dtype=np.int32)
    for i in range(1,num_lncRNAs+1):
        for j in range(1,num_diseases+1):
            net_new[i,j]=net[i-1,j-1]
    u_features=data['lncSim']
    disSim_path='raw_data/' + dataset + '/disSim.xlsx'        
    disSim_data=pd.read_excel(disSim_path,header=0)
    v_features=np.array(disSim_data)
    
    num_list=[len(u_features)]
    num_list.append(len(v_features))
    temp=np.zeros((net.shape[0],net.shape[1]),int)
    u_features=np.hstack((u_features,net))
    v_features=np.hstack((net.T,v_features))  
    a=np.zeros((1,u_features.shape[0]+v_features.shape[0]),int)
    b=np.zeros((1,v_features.shape[0]+u_features.shape[0]),int)
    u_features=np.vstack((a,u_features))
    v_features=np.vstack((b,v_features))
  
    #loading miRNA_name and disease_name
    lncRNA_name=[]
    disease_name=[]
    disease_name.append([])
    lncRNA_name.append([])
    f=open('raw_data/' + dataset+'/lncRNA_Name.txt','r') 
    while True:
        line=f.readline()
        if not line:
            break
        lncRNA_name.append(line)
    f.close()
    f=open('raw_data/' + dataset+'/disease_Name.txt','r') 
    while True:
        line=f.readline()
        if not line:
            break
        disease_name.append(line)
    f.close()
    print("lncRNA_name:",len(lncRNA_name))
    case_disease='renal carcinoma\n'
    if case_disease in disease_name:
        idx=disease_name.index(case_disease)

    u_idx,v_idx, labels=[],[],[]
    list=[]
    for i in range(1,net_new.shape[0]):
        if net_new[i][idx]==0:
            list.append([i,idx,net_new[i][idx]])
            
    for i in range(len(list)):
        u_idx.append(list[i][0])
        v_idx.append(list[i][1])
        labels.append(list[i][2])
    class_values=np.array([0,1],dtype=float)

    return u_features, v_features, net_new, labels, u_idx, v_idx, class_values,lncRNA_name,disease_name


