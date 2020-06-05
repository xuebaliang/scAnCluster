from scAnCluster_preprocess import *
from scAnCluster_network import *


if __name__ == "__main__":
    random_seed = 8888
    gpu_option = "0"
    dataname = "splatter_cluster_num_6_size_equal_mode_balance_dropout_rate_0.5_data.h5"
    X, Y, batch_label = read_simu(dataname)
    Y = Y.astype(np.int)
    cellname = np.array(["group" + str(i) for i in Y])
    dims = [1000, 256, 64, 32]
    highly_genes = 1000
    pretrain_epochs = 500
    batch_num = len(np.unique(batch_label))
    count_X = X
    if X.shape[1] == highly_genes:
        highly_genes = None
    print("begin the data proprocess")
    adata = sc.AnnData(X)
    adata.obs["celltype"] = cellname
    adata.obs["batch"] = batch_label
    adata = normalize(adata, highly_genes=highly_genes, size_factors=True, normalize_input=True, logtrans_input=True)
    X = adata.X.astype(np.float32)
    cellname = np.array(adata.obs["celltype"])
    batch_label = np.array(adata.obs["batch"])

    if highly_genes != None:
        high_variable = np.array(adata.var.highly_variable.index, dtype=np.int)
        count_X = count_X[:, high_variable]
    else:
        select_genes = np.array(adata.var.index, dtype=np.int)
        select_cells = np.array(adata.obs.index, dtype=np.int)
        count_X = count_X[:, select_genes]
        count_X = count_X[select_cells]
    assert X.shape == count_X.shape
    size_factor = np.array(adata.obs.size_factors).reshape(-1, 1).astype(np.float32)

    X_source = X[batch_label == 0]
    cellname_source = cellname[batch_label == 0]
    batch_label_source = batch_label[batch_label == 0]
    count_X_source = count_X[batch_label == 0]
    size_factor_source = size_factor[batch_label == 0]

    X_target = X[batch_label != 0]
    cellname_target = cellname[batch_label != 0]
    batch_label_target = batch_label[batch_label != 0]
    count_X_target = count_X[batch_label != 0]
    size_factor_target = size_factor[batch_label != 0]

    setting_list = [["whole", ["nothing"]],
                    ["source", ["group0"]],
                    ["target", ["group0"]]]

    result = []
    for setting in setting_list:
        masklist = setting[1]
        if setting[0] == "source":
            mask_index = []
            for i in range(len(cellname_source)):
                if cellname_source[i] not in masklist:
                    mask_index.append(i)
            mask_X_source = X_source[mask_index]
            mask_cellname_source = cellname_source[mask_index]
            mask_batch_label_source = batch_label_source[mask_index]
            mask_count_X_source = count_X_source[mask_index]
            mask_size_factor_source = size_factor_source[mask_index]
            mask_X_target = X_target
            mask_cellname_target = cellname_target
            mask_batch_label_target = batch_label_target
            mask_count_X_target = count_X_target
            mask_size_factor_target = size_factor_target
        elif setting[0] == "target":
            mask_index = []
            for i in range(len(cellname_target)):
                if cellname_target[i] not in masklist:
                    mask_index.append(i)
            mask_X_source = X_source
            mask_cellname_source = cellname_source
            mask_batch_label_source = batch_label_source
            mask_count_X_source = count_X_source
            mask_size_factor_source = size_factor_source
            mask_X_target = X_target[mask_index]
            mask_cellname_target = cellname_target[mask_index]
            mask_batch_label_target = batch_label_target[mask_index]
            mask_count_X_target = count_X_target[mask_index]
            mask_size_factor_target = size_factor_target[mask_index]
        else:
            mask_X_source = X_source
            mask_cellname_source = cellname_source
            mask_batch_label_source = batch_label_source
            mask_count_X_source = count_X_source
            mask_size_factor_source = size_factor_source
            mask_X_target = X_target
            mask_cellname_target = cellname_target
            mask_batch_label_target = batch_label_target
            mask_count_X_target = count_X_target
            mask_size_factor_target = size_factor_target
            print("we use the complete data to implement experiments")
        mask_X = np.concatenate((mask_X_source, mask_X_target), axis=0)
        mask_count_X = np.concatenate((mask_count_X_source, mask_count_X_target), axis=0)
        mask_cellname = np.concatenate((mask_cellname_source, mask_cellname_target))
        mask_batch_label = np.concatenate((mask_batch_label_source, mask_batch_label_target))
        mask_size_factor = np.concatenate((mask_size_factor_source, mask_size_factor_target), axis=0)
        classes = len(np.unique(mask_cellname_source))
        cluster_num = len(np.unique(mask_cellname))
        tf.reset_default_graph()
        scAn = scAnCluster(dataname, dims, batch_num, classes, cluster_num, 0.01, 0.01, 1.0, 1e-4, distance="fuzzy", distrib="ZINB")
        target_accuracy, target_ARI, annotated_target_accuracy, target_prediction_matrix = scAn.train(mask_X, mask_count_X, mask_cellname, mask_batch_label,
                                                                                                      mask_size_factor, pretrain_epochs, random_seed, gpu_option)
        print("Under this setting, for target data, the clustering accuracy is {}".format(target_accuracy))
        print("Under this setting, for target data, the clustering ARI is {}".format(target_ARI))
        print("Under this setting, for target data, the annotation accuracy is {}".format(target_accuracy))
        print("The target prediction information is in the target_prediction_matrix. It is a data frame, include four columns, "
              "they are true label, true cell type, cluster label, annotation cell type. You can save it in .csv file.")
        print(target_prediction_matrix)









