# MT-GNN
This is a repository for the paper entitled "Integrating Multi-Task Graph Neural Network and DFT Calculations for Site-Selectivity Prediction of Arenes and Mechanistic Knowledge Generation"
# Introduction
A multi-task learning graph neural network was developed to predict the challenging site-selectivity of ruthenium-catalyzed C-H functionalization of arenes. The dataset was collected manually including 241 individual reactions. The site-selectivity classification task is optimized alongside with two molecular property regression tasks of reaction components (arene and electrophile). For a better reaction representation, the reaction graphs are informed with previous mechanistic insights. Two additional test set (experimental and out-of-sample test) were prepared to validate the extrapolative ability of the MT-GNN model.
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
