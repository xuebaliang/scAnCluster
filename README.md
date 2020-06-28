# scAnCluster
Usage
-----
The scAnCluster tool is an implementation of deep supervised clustering and annotation algorithm for single-cell RNA-seq data. With scAnCluster, you can transfer the cell type label of reference data to the target data. We support label transfer in multiple situations, including the same label space for reference data and target data, larger label space for reference data and larger label space for target data. For the latter case, we can also discover novel cell types on the target data. The input of the model is the mixed reference data and target data, the rows represent the cells and the columns represent the genes. At the same time, we also need the batch label of each cell and the estimation value of the total number of cell types on the mixed data. If you have true labels of cells in advance, you can also test the effectiveness of our algorithm.
 
Requirement
-----
Python 3.6

Tensorflow 1.14

Keras 2.2

igraph 0.1.11

scanpy 1.4.3

scikit-learn 0.22.2

tqdm 4.32.2

Example and Quickstart
-----
We have provided some explanatory descriptions for the codes, please see the specific code files. Now we use a simulation dataset "splatter_cluster_num_6_size_equal_mode_balance_dropout_rate_0.5_data" to give an example. You can download this data from folder "scAnCluster/data/simulation". The dataset has two batches and each batch has 6 groups (clusters). You just need to download all code files and focus on the scAnCluster_run.py file. We have set three modes, that is, using the complete reference data (batch 0) and target data (batch 1) for experiment, removing group 0 from reference data and target data, respectively. You can run the following code in your command lines:

python scAnCluster_run.py

When you take the complete mixed data as input, you can get that the annotation accuracy and clustering ARI on the target data is    and    , respectively. Besides, the target prediction information is in the "target_prediction_matrix" variable. It is a data frame, include four columns, they are true label, true cell type, cluster label, annotation cell type. You can save it in .csv file. When you remove the group 0 from the reference data, you can get that the annotation accuracy and clustering ARI on the target data is    and    , respectively. Similarly, when remove the group 0 from the target data, you can get that the annotation accuracy and clustering ARI on the target data is    and    , respectively. We would continue to improve our tool to help our users as much as possible in the future.

Contributing
-----
Author email: clandzyy@pku.edu.cn
