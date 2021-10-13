# -*- coding: utf-8 -*-
import torch
import numpy as np
import sys, copy, math, time, pdb, warnings, traceback
import pickle
import scipy.io as sio
import scipy.sparse as ssp
import os.path
import random
import argparse
from shutil import copy, rmtree, copytree
from torch.optim.lr_scheduler import ReduceLROnPlateau
from util_functions import *
from preprocessing_Dataset1 import *
from train_eval import *
from models import *
from torch_geometric.data import Data, Dataset
import traceback
import warnings
import sys
import xlwt
from torchsummary import summary
import gc
from sklearn.metrics import precision_recall_curve,roc_curve,roc_auc_score,f1_score,precision_score,recall_score,auc



if __name__ == '__main__':

    # Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-train', action='store_true', default=False)
    parser.add_argument('--dataset', help='dataset name')
    
    parser.add_argument('--use-features', action='store_true', default=False)
    
    args = parser.parse_args()
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(2341)
    seed=2341
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    hop = 1
  
    if not args.no_train: 
        #Construct model
        print('training.....')
        data_combo = (args.dataset, '', '')
        u_features, v_features, net, labels, u_indices, v_indices, num_list = load_data(args.dataset)
        print('preprocessing end.')
        adj=torch.tensor(net)
        if args.use_features:
            n_features = u_features.shape[1] + v_features.shape[1]
        else:
            u_features, v_features = None, None
            n_features = 0
        all_indices = (u_indices, v_indices)
        print('begin constructing all_graphs')     
        all_graphs = extracting_subgraphs(net, all_indices, labels,hop, u_features,v_features,hop*2+1)
        mydataset = MyDataset(all_graphs, root='data/{}{}/{}/train'.format(*data_combo))
        print('constructing all_graphs end.') 

        sum=0
        all_results=[]
        max_f1=0
        
        for count in range(1):
            model = gGATLDA(515, side_features=args.use_features, n_side_features=515)
            #model=model.cuda()
            print('########',count,' training.'+'#########')
            
	        #K-fold cross-validation
            K=5
            all_f1_mean,all_f1_std=0,0
            all_accuracy_mean,all_accuracy_std=0,0
            all_recall_mean,all_recall_std=0,0
            all_precision_mean,all_precision_std=0,0
            all_auc_mean,all_auc_std=0,0
            all_aupr_mean,all_aupr_std=0,0
            truth=[]
            predict=[]
            f1_s=[]
            accuracy_s=[]
            recall_s=[]
            precision_s=[]
            auc_s=[]
            aupr_s=[]
            max=0
            for i in range(K):
                print('*'*25,i+1,'*'*25)
                                          
                train_graphs,test_graphs=get_k_fold_data(K,i,mydataset)
                test_auc,f1,accuracy,recall,precision,auc,aupr,one_truth,one_predict=train_multiple_epochs(train_graphs,test_graphs, model, adj)
                truth.extend(one_truth)
                predict.extend(one_predict)
                f1_s.append(f1)
                accuracy_s.append(accuracy)
                recall_s.append(recall)
                precision_s.append(precision)
                auc_s.append(auc)
                aupr_s.append(aupr)
                
                          
            print('#'*10,'Final k-fold cross validation results','#'*10) 
            print('The %d-fold CV auc: %f +/- %f' %(i,np.mean(auc_s),np.std(auc_s))) 
            print('The %d-fold CV aupr: %f +/- %f' %(i,np.mean(aupr_s),np.std(aupr_s))) 
            print('The %d-fold CV f1-score: %f +/- %f' %(i,np.mean(f1_s),np.std(f1_s)))
            print('The %d-fold CV recall: %f +/- %f' %(i,np.mean(recall_s),np.std(recall_s)))
            print('The %d-fold CV accuracy: %f +/- %f' %(i,np.mean(accuracy_s),np.std(accuracy_s)))
            print('The %d-fold CV precision: %f +/- %f' %(i,np.mean(precision_s),np.std(precision_s)))
            all_f1_mean=all_f1_mean+np.mean(f1_s)
            all_f1_std=all_f1_std+np.std(f1_s)
           
            all_recall_mean=all_recall_mean+np.mean(recall_s)
            all_recall_std=all_recall_std+np.std(recall_s)

            all_accuracy_mean=all_accuracy_mean+np.mean(accuracy_s)
            all_accuracy_std=all_accuracy_std+np.std(accuracy_s)

            all_precision_mean=all_precision_mean+np.mean(precision_s)
            all_precision_std=all_precision_std+np.std(precision_s)

            all_auc_mean=all_auc_mean+np.mean(auc_s)
            all_auc_std=all_auc_std+np.std(auc_s)

            all_aupr_mean=all_aupr_mean+np.mean(aupr_s)
            all_aupr_std=all_aupr_std+np.std(aupr_s)
          
            truth_predict=[truth,predict]
            all_results.append(truth_predict)
              
            
        np.save('results/log_truth_Dataset1_CV1.npy',np.array(truth))
        np.save('results/log_predict_Dataset1_CV1.npy',np.array(predict))
        torch.save(model,'model.pth')
    else:
        print("begin to predict")
        u_features, v_features, net, labels, u_indices, v_indices, class_values,lncRNA_name,disease_name =load_predict_data(args.dataset)
        all_indices = (u_indices, v_indices)
        print(len(labels))
        
        all_graphs = extracting_subgraphs(net,all_indices,labels, hop, u_features, v_features, hop*2+1)
        pred_loader = DataLoader(all_graphs, 1, shuffle=False, num_workers=0)
        model=torch.load('model.pth')
        
        pred=torch.Tensor()
        print(len(all_graphs),len(pred_loader),'begin predicting 1') 
        model.eval()
        torch.no_grad()
        #model=model.to(device)
        count=0
        for data in pred_loader:
            data=data.to(device)
            out=model(data)
            pred=torch.cat((pred,out),0)
        pred=pred[:,1]
        disease_id=192
        labels = np.array(net[1:286,disease_id])
        idx=np.where(labels == 0)[0]
        result=[]
        for i in range(len(pred)):
            result.append([lncRNA_name[u_indices[i]],pred[i]])
        topK_result=sorted(result, reverse=True,key=lambda a: a[1])
        for i in range(15):
             print(topK_result[i])
        
        workbook=xlwt.Workbook()
        sheet=workbook.add_sheet('renal carcinoma')
        for i in range(15):
            sheet.write(i,0,topK_result[i][0])

        workbook.save(r'top_K_result/renal carcinoma.xls')

        
    print("All end...")



