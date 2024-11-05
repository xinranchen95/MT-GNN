#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 14 15:49:50 2024

In this script, six reaction components are considered,
including arene, electrophile, catalyst, solvent, ligand and addtive.
DG referred to the arene reaction substrate, and RX referred to electrophile.

@author: chenxinran
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import dgl
import torch
from dgllife.utils import CanonicalAtomFeaturizer, CanonicalBondFeaturizer, mol_to_bigraph
import dgl.function as fn
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def gen_mol_form_smi(smi_list):
    
    mol_list = []
    for smi in smi_list:
        if isinstance(smi, str) == True :
            mol = Chem.MolFromSmiles(smi)
            if mol == None:
                print(smi)
            mol_list.append(mol)
        else:
            mol = 0
            mol_list.append(mol)
    return mol_list

def gen_ortho_sub_list(mol_list):
    
    def ortho_substituted(mol):
        mol_no_H = AllChem.RemoveHs(mol)
        for idx, atom in enumerate(mol_no_H.GetAtoms()):
            if idx == 1 :
                if atom.GetDegree() > 2:
                    return True
            elif idx == 5:
                if atom.GetDegree() > 2:
                    return True
    
    ortho_sub_list = []
    
    for idx, mol in enumerate(mol_list):
        if ortho_substituted(mol) == True:
            ortho_sub_list.append(idx)
    
    return  ortho_sub_list

def gen_meta_sub_list(mol_list):
    
    def meta_substituted(mol):
        mol_no_H = AllChem.RemoveHs(mol)
        for idx, atom in enumerate(mol_no_H.GetAtoms()):
            if idx == 2 :
                if atom.GetDegree() > 2:
                    return True
            elif idx == 4:
                if atom.GetDegree() > 2:
                    return True
    
    meta_sub_list = []
    
    for idx, mol in enumerate(mol_list):
        if meta_substituted(mol) == True:
            meta_sub_list.append(idx)
    
    return  meta_sub_list

def make_reaction_graphs(DG_mols_list, RX_mols_list, DG_decs1, DG_decs2, DG_decs3, DG_decs4 
                         , RX_decs1, RX_decs2, RX_decs3, RX_decs4, l_mols, c_mols, s_mols, a_mols):
    n = 0
    DG_graphs = []
    RX_graphs = []
    ligand_graphs = []
    catalyst_graphs = []
    solvent_graphs = []
    addictive_graphs = []
    reaction_graphs = []
    v1_graphs = []
    v2_graphs = []
    decs = []
    decs_mp = []
    
    node_featurizer = CanonicalAtomFeaturizer(atom_data_field='nfeat')
    edge_featurizer = CanonicalBondFeaturizer(bond_data_field='efeat')

    for mol1, mol2 in zip(DG_mols_list, RX_mols_list):
        
        g1 = mol_to_bigraph(mol1,node_featurizer=node_featurizer,edge_featurizer=edge_featurizer,canonical_atom_order=False)
        virtual_node_id1 = g1.number_of_nodes()
        fukui_decs1 = torch.zeros((virtual_node_id1, 1))
        for i in range(1,6):
            fukui_decs1[i][0] = DG_decs1[n,i-1]
        fukui_decs2 = torch.zeros((virtual_node_id1, 1))
        for i in range(1,6):
            fukui_decs2[i][0] = DG_decs2[n,i-1]
        fukui_decs3 = torch.zeros((virtual_node_id1, 1))
        for i in range(1,6):
            fukui_decs3[i][0] = DG_decs3[n,i-1]
        QC_DG_decs = torch.zeros((virtual_node_id1, 1))
        for i in range(1,6):
            QC_DG_decs[i][0] = DG_decs4[n,i-1]
        g1.ndata['nfeat'] = torch.cat([g1.ndata['nfeat'], fukui_decs1, fukui_decs2, fukui_decs3, QC_DG_decs], dim=1)
        g1_ndata_num = g1.ndata['nfeat'].shape[1]# g1_ndata_num is 78
        decs.append(g1.ndata['nfeat'])
        DG_graphs.append(g1)
        
        global_feature1 = torch.mean(g1.ndata['nfeat'], dim=0)
        g1 = dgl.add_nodes(g1, 1) #Create a new node
        g1.ndata['nfeat'][virtual_node_id1] = global_feature1
        g1 = dgl.add_edges(g1,[virtual_node_id1] * (g1.number_of_nodes()-1), list(range(g1.number_of_nodes()-1)))# 将新节点连接到所有节点
        g1.update_all(fn.copy_u('nfeat', 'm'), fn.mean('m', 'h'))#Message passing
        global_feature1 = torch.mean(g1.ndata['h'], dim=0)
        g1.ndata['h'][virtual_node_id1] = global_feature1
        decs_mp.append(g1.ndata['h'])
             
        g_v1 = dgl.graph((torch.tensor([0]), torch.tensor([0])))#Create graph for virtual node 1
        g_v1.ndata['h'] = torch.ones(1, g1_ndata_num)
        g_v1_f = g1.ndata['h'][virtual_node_id1]
        g_v1.ndata['h'][0] = g_v1_f
        v1_graphs.append(g_v1)
        
        g2 = mol_to_bigraph(mol2,node_featurizer=node_featurizer,edge_featurizer=edge_featurizer,canonical_atom_order=False)
        virtual_node_id2 = g2.number_of_nodes()# The number of current node is the ID of the new node
        Qc_decs = torch.zeros((virtual_node_id2, 1))
        Qc_decs[0][0] = RX_decs1[n]
        f0_decs = torch.zeros((virtual_node_id2, 1))
        f0_decs[0][0] = RX_decs2[n]
        f_m_decs = torch.zeros((virtual_node_id2, 1))
        f_m_decs[0][0] = RX_decs4[n]
        f_p_decs = torch.zeros((virtual_node_id2, 1))
        f_p_decs[0][0] = RX_decs4[n]
        g2.ndata['nfeat'] = torch.cat([g2.ndata['nfeat'], Qc_decs, f0_decs, f_m_decs, f_p_decs], dim=1)
        RX_graphs.append(g2)
        
        global_feature2 = torch.mean(g2.ndata['nfeat'], dim=0)
        g2 = dgl.add_nodes(g2, 1)
        g2.ndata['nfeat'][virtual_node_id2] = global_feature2
        g2 = dgl.add_edges(g2,[virtual_node_id2] * (g2.number_of_nodes()-1), list(range(g2.number_of_nodes()-1)))# 将新节点连接到所有节点
        g2.update_all(fn.copy_u('nfeat', 'm'), fn.mean('m', 'h'))
        global_feature2 = torch.mean(g2.ndata['h'], dim=0)
        g2.ndata['h'][virtual_node_id2] = global_feature2
               
        g2_ndata_num = g2.ndata['h'].shape[1]
        g_v2 = dgl.graph((torch.tensor([0]), torch.tensor([0])))#Create graph for virtual node 2
        g_v2.ndata['h'] = torch.ones(1, g2_ndata_num)
        g_v2_f = g2.ndata['h'][virtual_node_id2]
        g_v2.ndata['h'][0] = g_v2_f
        v2_graphs.append(g_v2)
        
        if l_mols[n] != 0:
            g_l = mol_to_bigraph(l_mols[n],node_featurizer=node_featurizer,edge_featurizer=edge_featurizer,canonical_atom_order=False, num_virtual_nodes=1)###创建ligand的图
            g_l_ndata_num = g_l.ndata['nfeat'].shape[1]
            global_feature_l = torch.mean(g_l.ndata['nfeat'], dim=0)
            g_v_l = dgl.graph((torch.tensor([0]), torch.tensor([0])))
            g_v_l.ndata['h'] = torch.ones(1, g_l_ndata_num)
            g_v_l.ndata['h'][0] = global_feature_l
            g_v_l.ndata['h'] = torch.cat([g_v_l.ndata['h'], torch.zeros((1, g1_ndata_num - g_l_ndata_num))], dim=1)
            ligand_graphs.append(g_v_l)
        else:
            g_v_l = dgl.graph((torch.tensor([0]), torch.tensor([0])))
            g_v_l.ndata['h'] = torch.zeros(1, g1_ndata_num)
            ligand_graphs.append(g_v_l)
        
        if c_mols[n] != 0:
            g_c = mol_to_bigraph(c_mols[n],node_featurizer=node_featurizer,edge_featurizer=edge_featurizer,canonical_atom_order=False)###创建ligand的图
            g_c_ndata_num = g_c.ndata['nfeat'].shape[1]
            global_feature_c = torch.mean(g_c.ndata['nfeat'], dim=0)
            g_v_c = dgl.graph((torch.tensor([0]), torch.tensor([0])))
            g_v_c.ndata['h'] = torch.ones(1, g_c_ndata_num)
            g_v_c.ndata['h'][0] = global_feature_c
            g_v_c.ndata['h'] = torch.cat([g_v_c.ndata['h'], torch.zeros((1, g1_ndata_num - g_c_ndata_num))], dim=1)
            catalyst_graphs.append(g_v_c)
        else:
            g_v_c = dgl.graph((torch.tensor([0]), torch.tensor([0])))
            g_v_c.ndata['h'] = torch.zeros(1, g1_ndata_num)
            catalyst_graphs.append(g_v_c)
        
        if s_mols[n] != 0:
            g_s = mol_to_bigraph(s_mols[n],node_featurizer=node_featurizer,edge_featurizer=edge_featurizer,canonical_atom_order=False)###创建ligand的图
            g_s_ndata_num = g_s.ndata['nfeat'].shape[1]
            global_feature_s = torch.mean(g_s.ndata['nfeat'], dim=0)
            g_v_s = dgl.graph((torch.tensor([0]), torch.tensor([0])))
            g_v_s.ndata['h'] = torch.ones(1, g_s_ndata_num)
            g_v_s.ndata['h'][0] = global_feature_s
            g_v_s.ndata['h'] = torch.cat([g_v_s.ndata['h'], torch.zeros((1, g1_ndata_num - g_s_ndata_num))], dim=1)
            solvent_graphs.append(g_v_s)
        else:
            g_v_s = dgl.graph((torch.tensor([0]), torch.tensor([0])))
            g_v_s.ndata['h'] = torch.zeros(1, g1_ndata_num)
            solvent_graphs.append(g_v_s)
        
        if a_mols[n] != 0:
            g_ad = mol_to_bigraph(a_mols[n],node_featurizer=node_featurizer,edge_featurizer=edge_featurizer,canonical_atom_order=False)###创建ligand的图
            g_ad_ndata_num = g_ad.ndata['nfeat'].shape[1]
            global_feature_ad = torch.mean(g_ad.ndata['nfeat'], dim=0)
            g_v_ad = dgl.graph((torch.tensor([0]), torch.tensor([0])))
            g_v_ad.ndata['h'] = torch.ones(1, g_ad_ndata_num)
            g_v_ad.ndata['h'][0] = global_feature_ad
            g_v_ad.ndata['h'] = torch.cat([g_v_ad.ndata['h'], torch.zeros((1, g1_ndata_num - g_ad_ndata_num))], dim=1)
            addictive_graphs.append(g_v_ad)
        else:
            g_v_ad = dgl.graph((torch.tensor([0]), torch.tensor([0])))
            g_v_ad.ndata['h'] = torch.zeros(1, g1_ndata_num)
            addictive_graphs.append(g_v_ad)
        
        g_r = dgl.batch([g_v1, g_v2, g_v_l, g_v_c, g_v_s, g_v_ad])#Merge the six subgraphs
        
        #Add edges between all the nodes
        g_r.add_edges([0]*5,range(1,6))
        g_r.add_edges([1]*5,[0,2,3,4,5])
        g_r.add_edges([2]*5,[0,1,3,4,5])
        g_r.add_edges([3]*5,[0,1,2,4,5])
        g_r.add_edges([4]*5,[0,1,2,3,5])
        g_r.add_edges([5]*5,[0,1,2,3,4])
        g_r.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h'))
        weight = torch.tensor([[0.3], [0.3], [0.1], [0.1], [0.1], [0.1]])
        g_r.ndata['h'] = torch.cat([g_r.ndata['h'], weight], dim=1)
        reaction_graphs.append(g_r)
        
        n = n+1
    return v1_graphs, v2_graphs, reaction_graphs, decs, decs_mp, DG_graphs, RX_graphs
