# MT-GNN
This is a repository for the paper entitled "Integrating Multi-Task Graph Neural Network and DFT Calculations for Site-Selectivity Prediction of Arenes and Mechanistic Knowledge Generation"
# Introduction
A multi-task learning graph neural network was developed to predict the challenging site-selectivity of ruthenium-catalyzed C-H functionalization of arenes. The  ruthenium-catalyzed C-H functionalization dataset was collected manually including 241 individual reactions. Two key design elements facilitate site-selectivity prediction: the mechanistically informed reaction graph and the multi-task architecture. Two additional test set (experimental and out-of-sample test) were prepared to validate the extrapolative ability of the MT-GNN model.
## Mechanistic-informed reaction graph
For a better reaction representation, the previous mechanistic information is embedded to the reactive atoms of the two substrates. Through message passing, the two substrate graphs are condensed to two virtual nodes. Together with other four reaction component nodes (catalyst, solvent, ligand and additive), a reaction graph is completed.

<img src="pictures/1.png" alt="image1" style="width:500px;"/>
## Multi-task architecture
The site-selectivity classification task is optimized alongside with two molecular property regression tasks of reaction substrates (arene and electrophile). These two regression tasks which is related to the site-selectivity aim to assisted the classification task by knowledge acquisition during the simultaneous learning.

![image2](pictures/2.png)
# Dependence
python 3.11.5  
pandas 2.1.4  
numpy 1.24.3  
rdkit 2022.09.5  
torch 1.13.1  
sklearn 1.3.0  
dgl 1.1.3  
scipy 1.11.4  
seaborn 0.13.2  
matplotlib 3.8.0  
# Installation of dependence
We recommend using Anaconda for preparing the dependence as some packages are built-in Anaconda base environment, which are not mentioned in the dependence section. 
# Demo
The Jupyter notebook demo.ipynb shows an example of running the MT-GNN model with ruthenium-catalyzed C-H functionalization dataset including the information of training time.
