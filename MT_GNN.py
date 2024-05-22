#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 15:49:39 2024

In this script, six reaction components are considered,
including arene, electrophile, catalyst, solvent, ligand and addtive.
DG referred to the arene reaction substrate, and RX referred to electrophile.

@author: chenxinran
"""
import numpy as np
import dgl
import torch
import torch.nn as nn
import dgl.function as fn
import dgl.nn.pytorch as dglnn
from dgl.nn import GATConv
import torch.nn.functional as F
from torch.utils.data import Dataset
from sklearn.metrics import mean_absolute_error
from scipy.stats import pearsonr

class MultiTaskDataset(Dataset):
    def __init__(self, graph1_list, graph2_list, graph3_list, label1_list, label2_list, label3_list, num_list):
        self.graph1_list = graph1_list
        self.graph2_list = graph2_list
        self.graph3_list = graph3_list
        self.label1_list = label1_list
        self.label2_list = label2_list
        self.label3_list = label3_list
        self.num_list = num_list

    def __len__(self):
        return len(self.graph1_list)

    def __getitem__(self, index):
        graph1 = self.graph1_list[index]
        graph2 = self.graph2_list[index]
        graph3 = self.graph3_list[index]
        label1 = self.label1_list[index]
        label2 = self.label2_list[index]
        label3 = self.label3_list[index]
        num = self.num_list[index]
        return graph1, graph2, graph3, label1, label2, label3, num

class MultiTaskGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, regression_output_dim1, regression_output_dim2, num_classes):
        super(MultiTaskGNN, self).__init__()
        
        #The original dimension for DG and RX graphs is one smaller than reaction graphs
        self.gatconv1 = GATConv(in_dim-1, in_dim-1, num_heads=3)
        self.gatconv2 = GATConv(in_dim, in_dim, num_heads=3)
        
        self.conv1 = dglnn.GraphConv(in_dim, hidden_dim, allow_zero_in_degree=True)
        self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim, allow_zero_in_degree=True)
        
        self.conv3 = dglnn.GraphConv(in_dim-1, hidden_dim, allow_zero_in_degree=True)
        
        self.regression_output1 = nn.Linear(hidden_dim, regression_output_dim1)
        self.regression_output2 = nn.Linear(hidden_dim, regression_output_dim2)
        self.classification_output = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, g1, g2, g3, h1, h2, h3):
        
        h1, attention_values = self.gatconv1(g1, h1, get_attention = True)
        attention_values, _ = torch.max(attention_values, dim=1)
        g1.update_all(fn.copy_u('nfeat', 'm'), fn.sum('m', 'nfeat'))
        h1, _ = torch.max(h1, dim=1)
        h1 = F.relu(h1)
        h1 = F.relu(self.conv3(g1, h1))
        h1 = F.relu(self.conv2(g1, h1))
        
        h2 = self.gatconv1(g2, h2, get_attention = False)
        h2, _ = torch.max(h2, dim=1)
        g2.update_all(fn.copy_u('nfeat', 'm'), fn.sum('m', 'nfeat'))
        h2 = F.relu(self.conv3(g2, h2))
        h2 = F.relu(self.conv2(g2, h2))
        
        h3, attention_reactions = self.gatconv2(g3, h3, get_attention = True)
        attention_reactions, _ = torch.max(attention_reactions, dim=1)
        h3, _ = torch.max(h3, dim=1)
        g3.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h'))
        h3 = F.relu(self.conv1(g3, h3))
        h3 = F.relu(self.conv2(g3, h3))
        
        with g1.local_scope(), g2.local_scope(), g3.local_scope():
            g1.ndata['h'] = h1
            hg1 = dgl.mean_nodes(g1, 'h')         
            
            g2.ndata['h'] = h2
            hg2 = dgl.mean_nodes(g2, 'h')
            
            g3.ndata['h'] = h3
            hg3 = dgl.mean_nodes(g3, 'h')
            weight = h3[:,78]
            
            regression_output1 = self.regression_output1(hg1)  # Regression task for DG
            regression_output2 = self.regression_output2(hg2)  # Regression task for RX
            classification_output = self.classification_output(hg3)  # Classification task
        
            return regression_output1, regression_output2, classification_output, attention_values, attention_reactions, weight, hg3

def evaluate(model, dataloader, o_sub_list, m_sub_list):
    model.eval()
    correct_task3 = 0
    total = 0
    num_list = []
    pred_task1 = []
    label_task1 = []
    pred_task2 = []
    label_task2 = []
    attention_zero = torch.zeros(1,1)
    attention_DG = torch.empty(0)
    attention_r_all = torch.empty(0)

    for g1, g2, g3, labels1, labels2, labels3, num in dataloader:
        feat1 = g1.ndata['nfeat']
        feat2 = g2.ndata['nfeat']
        feat3 = g3.ndata['h']
        
        task1_output, task2_output, task3_output, attention, attention_r, weight, h3 = model(g1, g2, g3, feat1, feat2, feat3)
        
        total += len(labels3)
        _, predicted = torch.max(task3_output, dim=1)
        
        for i in range(len(labels3)):
            if num[i] in o_sub_list:
                if labels3[i] == predicted[i]:
                    correct_task3 += 1               
            elif num[i] in m_sub_list:
                if labels3[i] == predicted[i]:
                    correct_task3 += 1
            else:
                if labels3[i] == 0 and predicted[i] == 0:
                    correct_task3 += 1
                elif labels3[i] == 0 and predicted[i] == 4:
                    correct_task3 += 1
                elif labels3[i] == 1 and predicted[i] == 1:
                    correct_task3 += 1
                elif labels3[i] == 1 and predicted[i] == 3:
                    correct_task3 += 1
                elif labels3[i] == 2 and predicted[i] == 2:
                    correct_task3 += 1

        attention_DG = torch.cat((attention_DG, attention, attention_zero), dim=0) 
        attention_r_all = torch.cat((attention_r_all, attention_r, attention_zero), dim=0)
        num_list.append(num)
        task1_output = task1_output.tolist()
        task2_output = task2_output.tolist()
        labels1 = labels1.tolist()
        labels2 = labels2.tolist()
        pred_task1.extend(task1_output)
        label_task1.extend(labels1)
        pred_task2.extend(task2_output)
        label_task2.extend(labels2)               
        
    accuracy_task3 = 1.0 * correct_task3 / total

    MAE1 = mean_absolute_error(label_task1, pred_task1)
    MAE2 = mean_absolute_error(label_task2, pred_task2)    
    
    label_task1 = np.array(label_task1)
    pred_task1 = np.array(pred_task1)
    label_task1_flat = label_task1.flatten()
    pred_task1_flat = pred_task1.flatten()
    r2_1, _ = pearsonr(label_task1_flat, pred_task1_flat)
    
    label_task2 = np.array(label_task2)
    pred_task2 = np.array(pred_task2)
    label_task2_flat = label_task2.flatten()
    pred_task2_flat = pred_task2.flatten()
    r2_2, _ = pearsonr(label_task2_flat, pred_task2_flat)

    return accuracy_task3, predicted, labels3, num_list, MAE1, MAE2, r2_1, r2_2, attention_DG, attention_r_all, weight, h3
