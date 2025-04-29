# ASCE-PPIS:A Protein-Protein Interaction Sites Predictior Based on Equivariant Graph Neural Network with Fusion of Structure-Aware Pooling and Graph Collapse


## Abstract
Identifying protein-protein interaction sites constitutes a crucial step in understanding disease mechanisms and drug development. As experimental methods for PPIS identification are expensive and time-consuming, numerous computational screening approaches have been developed, among which graph neural network based methods have achieved remarkable progress in recent years. However, existing methods lack the utilization of interactions between amino acid molecules and fail to address the dense characteristics of protein graphs. In light of this, we propose ASCE-PPIS, an equivariant graph neural network-based method for protein-protein interaction prediction. This novel approach integrates graph pooling and graph collapse to address the aforementioned challenges. Our model learns molecular features and interactions through an equivariant neural network, and constructs subgraphs to acquire multi-scale features based on a structure-adaptive sampling strategy, and fuses the information of the original and subgraphs through graph collapse. Finally, we fusing protein large language model features through the integration strategy based on bagging and meta-modeling to improve the generalization performance on different proteins. Experimental results demonstrate that ASCE-PPIS achieves over 10% performance improvement compared to existing methods on the Test60 dataset, highlighting its potential in PPI site prediction tasks. 

## Preparation
### Environment Setup
The repo mainly requires the following packages.
+ dgl==2.3.0+cu121
+ numpy
+ packaging
+ pandas
+ pyg-lib==0.4.0+pt23cu121
+ scikit-learn
+ scipy
+ tokenizers==0.19.1
+ torch==2.3.1
+ torch-geometric==2.6.0
+ torch_cluster==1.6.3+pt23cu121
+ torch_scatter==2.1.2+pt23cu121
+ torch_sparse==0.6.18+pt23cu121
+ torch_spline_conv==1.2.2+pt23cu121
+ torchaudio==2.3.1
+ torchvision==0.18.1
+ tqdm

## Experimental Procedure
### Dataset
Handcrafted features and LLM features are saved at "./Feature/".We choose the esm2_t33_650M_UR50D() pre-trained model of ESM-2.

### Model Training
Run the following script to run 5-fold cross-validation.
```python
python train.py 
```
**We also provide the model in the paper at "./Model".Run the following script to test the performance of ASCE-PPIS. </br>
```python
python Ensemble.py 
```
Also you can make the corresponding changes in "dataload.py" to test a single model by running the following script.
```python
python test.py 
```

