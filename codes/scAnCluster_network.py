import numpy as np
import pandas as pd
import os
import time
import math
import random
import tensorflow as tf
import keras.backend as K
from keras.layers import GaussianNoise, Dense, Activation
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics.cluster import contingency_matrix


MeanAct = lambda x: tf.clip_by_value(K.exp(x), 1e-5, 1e6)
DispAct = lambda x: tf.clip_by_value(tf.nn.softplus(x), 1e-4, 1e4)

### define cluster accuracy
def cluster_acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

### transform label vector to label matrix
def label2matrix(label):
    unique_label, label = np.unique(label, return_inverse=True)
    one_hot_label = np.zeros((len(label), len(unique_label)))
    one_hot_label[np.arange(len(label)), label] = 1
    return one_hot_label

### calculation the annotation results
def annotation(cellname_train, cellname_test, Y_pred_train, Y_pred_test):
    train_confusion_matrix = contingency_matrix(cellname_train, Y_pred_train)
    annotated_cluster = np.unique(Y_pred_train)[train_confusion_matrix.argmax(axis=1)]
    annotated_celltype = np.unique(cellname_train)
    annotated_score = np.max(train_confusion_matrix, axis=1) / np.sum(train_confusion_matrix, axis=1)
    annotated_celltype[(np.max(train_confusion_matrix, axis=1) / np.sum(train_confusion_matrix, axis=1)) < 0.5] = "unassigned"
    final_annotated_cluster = []
    final_annotated_celltype = []
    for i in np.unique(annotated_cluster):
        candidate_celltype = annotated_celltype[annotated_cluster == i]
        candidate_score = annotated_score[annotated_cluster == i]
        final_annotated_cluster.append(i)
        final_annotated_celltype.append(candidate_celltype[np.argmax(candidate_score)])
    annotated_cluster = np.array(final_annotated_cluster)
    annotated_celltype = np.array(final_annotated_celltype)

    succeed_annotated_train = 0
    succeed_annotated_test = 0
    test_annotation_label = np.array(["original versions for unassigned cell ontology types"] * len(cellname_test))
    for i in range(len(annotated_cluster)):
        succeed_annotated_train += (cellname_train[Y_pred_train == annotated_cluster[i]] == annotated_celltype[i]).sum()
        succeed_annotated_test += (cellname_test[Y_pred_test == annotated_cluster[i]] == annotated_celltype[i]).sum()
        test_annotation_label[Y_pred_test == annotated_cluster[i]] = annotated_celltype[i]
    annotated_train_accuracy = np.around(succeed_annotated_train / len(cellname_train), 4)
    total_overlop_test = 0
    for celltype in np.unique(cellname_train):
        total_overlop_test += (cellname_test == celltype).sum()
    annotated_test_accuracy = np.around(succeed_annotated_test / total_overlop_test, 4)
    test_annotation_label[test_annotation_label == "original versions for unassigned cell ontology types"] = "unassigned"
    return annotated_train_accuracy, annotated_test_accuracy, test_annotation_label


def _nan2zero(x):
    return tf.where(tf.is_nan(x), tf.zeros_like(x), x)


def _nan2inf(x):
    return tf.where(tf.is_nan(x), tf.zeros_like(x)+np.inf, x)


def _nelem(x):
    nelem = tf.reduce_sum(tf.cast(~tf.is_nan(x), tf.float32))
    return tf.cast(tf.where(tf.equal(nelem, 0.), 1., nelem), x.dtype)


def _reduce_mean(x):
    nelem = _nelem(x)
    x = _nan2zero(x)
    return tf.divide(tf.reduce_sum(x), nelem)

### adaptive distance
def adapative_dist(hidden, clusters, sigma):
    dist1 = K.sum(K.square(K.expand_dims(hidden, axis=1) - clusters), axis=2)
    dist2 = K.sqrt(dist1)
    dist = (1 + sigma) * dist1 / (dist2 + sigma)
    return dist

### fuzzy kmeans
def fuzzy_kmeans(hidden, clusters, sigma, theta, adapative = True):
    if adapative:
        dist = adapative_dist(hidden, clusters, sigma)
    else:
        dist = K.sum(K.square(K.expand_dims(hidden, axis=1) - clusters), axis=2)
    temp_dist = dist - tf.reshape(tf.reduce_min(dist, axis=1), [-1, 1])
    q = K.exp(-temp_dist / theta)
    q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
    fuzzy_dist = q * dist
    return dist, fuzzy_dist

### sphere kmeans
def sphere_kmeans(hidden, clusters, theta):
    dist = 2 * (1 - tf.matmul(hidden, tf.transpose(clusters)))
    q = K.exp(-dist / theta)
    q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
    fuzzy_dist = q * dist
    return dist, fuzzy_dist

### negative binomial likelihood
def NB(theta, y_true, y_pred, mask = False, debug = False, mean = False):
    eps = 1e-10
    scale_factor = 1.0
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32) * scale_factor
    if mask:
        nelem = _nelem(y_true)
        y_true = _nan2zero(y_true)
    theta = tf.minimum(theta, 1e6)
    t1 = tf.lgamma(theta + eps) + tf.lgamma(y_true + 1.0) - tf.lgamma(y_true + theta + eps)
    t2 = (theta + y_true) * tf.log(1.0 + (y_pred / (theta + eps))) + (y_true * (tf.log(theta + eps) - tf.log(y_pred + eps)))
    if debug:
        assert_ops = [tf.verify_tensor_all_finite(y_pred, 'y_pred has inf/nans'),
                      tf.verify_tensor_all_finite(t1, 't1 has inf/nans'),
                      tf.verify_tensor_all_finite(t2, 't2 has inf/nans')]
        with tf.control_dependencies(assert_ops):
            final = t1 + t2
    else:
        final = t1 + t2
    final = _nan2inf(final)
    if mean:
        if mask:
            final = tf.divide(tf.reduce_sum(final), nelem)
        else:
            final = tf.reduce_mean(final)
    return final

### zero-inflated negative binomial likelihood
def ZINB(pi, theta, y_true, y_pred, ridge_lambda, mean = True, mask = False, debug = False):
    eps = 1e-10
    scale_factor = 1.0
    nb_case = NB(theta, y_true, y_pred, mean=False, debug=debug) - tf.log(1.0 - pi + eps)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32) * scale_factor
    theta = tf.minimum(theta, 1e6)

    zero_nb = tf.pow(theta / (theta + y_pred + eps), theta)
    zero_case = -tf.log(pi + ((1.0 - pi) * zero_nb) + eps)
    result = tf.where(tf.less(y_true, 1e-8), zero_case, nb_case)
    ridge = ridge_lambda * tf.square(pi)
    result += ridge
    if mean:
        if mask:
            result = _reduce_mean(result)
        else:
            result = tf.reduce_mean(result)

    result = _nan2inf(result)
    return result

### define scAnCluster network
class scAnCluster(object):
    def __init__(self, dataname, dims, batch_num, classes, cluster_num, alpha, gamma, theta, learning_rate, sigma = 0., noise_sd = 1.5,
                 init = "glorot_uniform", act = "relu", distance = "sphere", distrib = "ZINB", supervised = True, selfsupervised = True, unsupervised = True):
        self.dataname = dataname
        self.dims = dims
        self.batch_num = batch_num
        self.classes = classes
        self.cluster_num = cluster_num
        self.alpha = alpha
        self.gamma = gamma
        self.theta = theta
        self.sigma = sigma
        self.learning_rate = learning_rate
        self.noise_sd = noise_sd
        self.init = init
        self.act = act
        self.distance = distance
        self.distrib = distrib
        self.supervised = supervised
        self.selfsupervised = selfsupervised
        self.unsupervised = unsupervised

        self.n_stacks = len(self.dims) - 1
        self.x = tf.placeholder(dtype=tf.float32, shape=(None, self.dims[0]))
        self.y = tf.placeholder(dtype=tf.float32, shape=(None, self.classes))
        self.batch = tf.placeholder(dtype=tf.float32, shape=(None, batch_num))
        self.label_vec = tf.placeholder(dtype=tf.float32, shape=(None, ))
        self.mask_vec = tf.placeholder(dtype=tf.float32, shape=(None, ))
        self.x_count = tf.placeholder(dtype=tf.float32, shape=(None, self.dims[0]))
        self.sf_layer = tf.placeholder(dtype=tf.float32, shape=(None, 1))
        self.upper_threshold = tf.placeholder(dtype=tf.float32, shape=(1, ))
        self.lower_threshold = tf.placeholder(dtype=tf.float32, shape=(1, ))
        self.clusters = tf.get_variable(name=self.dataname + "/clusters_rep", shape=[self.cluster_num, self.dims[-1]],
                                        dtype=tf.float32, initializer=tf.glorot_uniform_initializer())

        self.label_mat = tf.reshape(self.label_vec, [-1, 1]) - tf.reshape(self.label_vec, [1, -1])
        self.label_mat = tf.cast(tf.equal(self.label_mat, 0.), tf.float32)
        self.mask_mat = tf.matmul(tf.reshape(self.mask_vec, [-1, 1]), tf.reshape(self.mask_vec, [1, -1]))

        self.h = tf.concat([self.x, self.batch], axis=1)
        self.h = GaussianNoise(self.noise_sd, name='input_noise')(self.h)
        for i in range(self.n_stacks - 1):
            self.h = Dense(units=self.dims[i + 1], kernel_initializer=self.init, name='encoder_%d' % i)(self.h)
            self.h = GaussianNoise(self.noise_sd, name='noise_%d' % i)(self.h)  # add Gaussian noise
            self.h = Activation(self.act)(self.h)
        self.latent = Dense(units=self.dims[-1], kernel_initializer=self.init, name='encoder_hidden')(self.h)

        if self.supervised:
            self.discriminate = Dense(units=self.classes, activation=tf.nn.softmax, kernel_initializer=self.init, name='classification_layer')(self.latent)
            self.softmax_mat = -self.y * tf.log(tf.clip_by_value(self.discriminate, 1e-10, 1.0))
            self.softmax_loss = tf.reduce_sum(self.softmax_mat)

        self.normalize_latent = tf.nn.l2_normalize(self.latent, axis=1)
        if self.selfsupervised:
            self.similarity_ = tf.matmul(self.normalize_latent, tf.transpose(self.normalize_latent))
            self.similarity = tf.nn.relu(self.similarity_)

            self.positive_mask = tf.cast(tf.greater(self.similarity, self.upper_threshold), tf.float32)
            self.positive_mask = self.positive_mask + self.mask_mat * self.label_mat
            self.positive_mask = tf.cast(tf.greater(self.positive_mask, 0.5), tf.float32)

            self.negative_mask = tf.cast(tf.less(self.similarity, self.lower_threshold), tf.float32)
            self.negative_mask = self.negative_mask + (1.0 - self.label_mat) * self.mask_mat
            self.negative_mask = tf.cast(tf.greater(self.negative_mask, 0.5), tf.float32)

        if self.unsupervised:
            if self.distance == "sphere":
                #self.normalize_clusters = tf.nn.l2_normalize(self.clusters, axis=1)
                self.latent_dist1, self.latent_dist2 = sphere_kmeans(self.normalize_latent, self.normalize_clusters, self.theta)
            elif self.distance == "fuzzy":
                self.latent_dist1, self.latent_dist2 = fuzzy_kmeans(self.latent, self.clusters, self.sigma, self.theta, adapative=True)

        self.h = tf.concat([self.latent, self.batch], axis=1)
        for i in range(self.n_stacks - 1, 0, -1):
            self.h = Dense(units=self.dims[i], activation=self.act, kernel_initializer=self.init,
                           name='decoder_%d' % i)(self.h)

        if self.distrib == "ZINB":
            self.pi = Dense(units=self.dims[0], activation='sigmoid', kernel_initializer=self.init, name='pi')(self.h)
            self.disp = Dense(units=self.dims[0], activation=DispAct, kernel_initializer=self.init, name='dispersion')(self.h)
            self.mean = Dense(units=self.dims[0], activation=MeanAct, kernel_initializer=self.init, name='mean')(self.h)
            self.output = self.mean * tf.matmul(self.sf_layer, tf.ones((1, self.mean.get_shape()[1]), dtype=tf.float32))
            self.likelihood_loss = ZINB(self.pi, self.disp, self.x_count, self.output, ridge_lambda=1.0)
        elif self.distrib == "NB":
            self.disp = Dense(units=self.dims[0], activation=DispAct, kernel_initializer=self.init, name='dispersion')(self.h)
            self.mean = Dense(units=self.dims[0], activation=MeanAct, kernel_initializer=self.init, name='mean')(self.h)
            self.output = self.mean * tf.matmul(self.sf_layer, tf.ones((1, self.mean.get_shape()[1]), dtype=tf.float32))
            self.likelihood_loss = NB(self.disp, self.x_count, self.output, mask=False, debug=False, mean=True)

        if self.supervised:
            self.pre_loss = self.likelihood_loss + self.softmax_loss
        else:
            self.pre_loss = self.likelihood_loss

        if self.selfsupervised:
            self.cross_entropy1 = self.mask_mat * (-self.label_mat * tf.log(tf.clip_by_value(self.similarity, 1e-10, 1.0)) -
                                                   (1 - self.label_mat) * tf.log(tf.clip_by_value(1 - self.similarity, 1e-10, 1.0)))
            self.cross_entropy2 = -self.positive_mask * tf.log(tf.clip_by_value(self.similarity, 1e-10, 1.0)) - \
                                  self.negative_mask * tf.log(tf.clip_by_value(1 - self.similarity, 1e-10, 1.0))
            self.mid_loss1 = self.likelihood_loss + self.alpha * self.cross_entropy1
            self.mid_loss2 = self.likelihood_loss + self.alpha * self.cross_entropy2

        if self.unsupervised:
            self.kmeans_loss = tf.reduce_mean(tf.reduce_sum(self.latent_dist2, axis=1))
            self.total_loss = self.likelihood_loss + self.gamma * self.kmeans_loss

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.pretrain_op = self.optimizer.minimize(self.pre_loss)

        if self.selfsupervised:
            self.midtrain_op1 = self.optimizer.minimize(self.mid_loss1)
            self.midtrain_op2 = self.optimizer.minimize(self.mid_loss2)

        if self.unsupervised:
            self.train_op = self.optimizer.minimize(self.total_loss)

### pretrain and funetrain
    def train(self, X, count_X, cellname, batch_label, size_factor, pretrain_epochs, random_seed, gpu_option):
        t1 = time.time()
        np.random.seed(random_seed)
        tf.set_random_seed(random_seed)
        batch_size = 256
        midtrain_epochs = 101
        funetrain_epochs = 2000
        tol = 0.001
        X_source = X[batch_label == 0]
        count_X_source = count_X[batch_label == 0]
        cellname_source = cellname[batch_label == 0]
        cellname_target = cellname[batch_label != 0]
        cell_type, Y = np.unique(cellname, return_inverse=True)
        Y_source = Y[batch_label == 0]
        Y_target = Y[batch_label != 0]
        batch_mat = label2matrix(batch_label)
        batch_mat_train = batch_mat[batch_label == 0]
        label_vec = Y.astype(np.float32)
        label_vec_train = label_vec[batch_label == 0]
        mask_vec = batch_label - 1 + 1
        mask_vec[mask_vec != 0] = 1
        mask_vec = (1 - mask_vec).astype(np.float32)
        mask_vec_train = mask_vec[batch_label == 0]
        size_factor_source = size_factor[batch_label == 0]

        cluster_num_source = len(np.unique(Y_source))
        print("The source data has {} clusters".format(cluster_num_source))
        cluster_num_target = len(np.unique(Y_target))
        print("The target data has {} clusters".format(cluster_num_target))

        onehot_Y = np.zeros((X.shape[0], len(np.unique(Y_source))))
        onehot_Y_train = label2matrix(Y_source)
        onehot_Y[batch_label == 0] = onehot_Y_train
        onehot_Y = onehot_Y.astype(np.float32)

        if X.shape[0] < batch_size:
            batch_size = X.shape[0]

        n_clusters = len(np.unique(Y))
        print("Mixed data has {} total clusters".format(n_clusters))

        upper_threshold = 0.95
        lower_threshold = 0.455
        eta = 0.
        print("end the data proprocess")
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_option

        config_ = tf.ConfigProto()
        config_.gpu_options.allow_growth = True
        config_.allow_soft_placement = True
        sess = tf.Session(config=config_)
        sess.run(init)

        print("begin model pretrain")
        latent_repre = np.zeros((X.shape[0], self.dims[-1]))
        iteration_per_epoch = math.ceil(float(len(X)) / float(batch_size))
        iteration_per_epoch_train = math.ceil(float(len(X_source)) / float(batch_size))
        for i in range(pretrain_epochs):
            for j in range(iteration_per_epoch):
                batch_idx = random.sample(range(X.shape[0]), batch_size)
                _, likelihood_loss, latent = sess.run(
                    [self.pretrain_op, self.likelihood_loss, self.latent],
                    feed_dict={
                        self.upper_threshold: np.array([upper_threshold]),
                        self.lower_threshold: np.array([lower_threshold]),
                        self.sf_layer: size_factor[batch_idx],
                        self.x: X[batch_idx],
                        self.y: onehot_Y[batch_idx],
                        self.batch: batch_mat[batch_idx],
                        self.x_count: count_X[batch_idx],
                        self.label_vec: label_vec[batch_idx],
                        self.mask_vec: mask_vec[batch_idx]})
                latent_repre[batch_idx] = latent

        if self.supervised:
            print("begin similarity fuse training")
            for i in range(midtrain_epochs):
                for k in range(iteration_per_epoch_train):
                    batch_idx = random.sample(range(X_source.shape[0]), batch_size)
                    _, cross_entropy1, latent = sess.run(
                        [self.midtrain_op1, self.cross_entropy1, self.latent],
                        feed_dict={
                            self.upper_threshold: np.array([upper_threshold]),
                            self.lower_threshold: np.array([lower_threshold]),
                            self.sf_layer: size_factor_source[batch_idx],
                            self.x: X_source[batch_idx],
                            self.y: onehot_Y_train[batch_idx],
                            self.batch: batch_mat_train[batch_idx],
                            self.x_count: count_X_source[batch_idx],
                            self.label_vec: label_vec_train[batch_idx],
                            self.mask_vec: mask_vec_train[batch_idx]})
                for j in range(iteration_per_epoch):
                    batch_idx = random.sample(range(X.shape[0]), batch_size)
                    _, cross_entropy2, latent = sess.run(
                        [self.midtrain_op2, self.cross_entropy2, self.latent],
                        feed_dict={
                            self.upper_threshold: np.array([upper_threshold]),
                            self.lower_threshold: np.array([lower_threshold]),
                            self.sf_layer: size_factor[batch_idx],
                            self.x: X[batch_idx],
                            self.y: onehot_Y[batch_idx],
                            self.batch: batch_mat[batch_idx],
                            self.x_count: count_X[batch_idx],
                            self.label_vec: label_vec[batch_idx],
                            self.mask_vec: mask_vec[batch_idx]})
                    latent_repre[batch_idx] = latent
                eta += 0.0045
                upper_threshold = 0.95 - eta
                lower_threshold = 0.455 + 0.1 * eta
                if upper_threshold < lower_threshold:
                    break

        print("Running k-means on the learned embeddings...")
        kmeans = KMeans(n_clusters=n_clusters, init="k-means++")
        latent_repre = np.nan_to_num(latent_repre)
        Y_pred = kmeans.fit_predict(latent_repre)
        Y_pred_target = Y_pred[batch_label != 0]
        last_pred_target = np.copy(Y_pred_target)

        if self.unsupervised:
            from sklearn import preprocessing
            init_cluster_centers = preprocessing.normalize(kmeans.cluster_centers_, axis=1, norm="l2")
            sess.run(tf.assign(self.clusters, init_cluster_centers))
            #sess.run(tf.assign(self.clusters, kmeans.cluster_centers_))
            print("end the pretraining")
            print("begin the cluster funetraining")
            for i in range(funetrain_epochs):
                if (i + 1) % 10 != 0:
                    for j in range(iteration_per_epoch):
                        batch_idx = random.sample(range(X.shape[0]), batch_size)
                        _, Kmeans_loss = sess.run(
                            [self.train_op, self.kmeans_loss],
                            feed_dict={
                                self.upper_threshold: np.array([upper_threshold]),
                                self.lower_threshold: np.array([lower_threshold]),
                                self.sf_layer: size_factor[batch_idx],
                                self.x: X[batch_idx],
                                self.y: onehot_Y[batch_idx],
                                self.batch: batch_mat[batch_idx],
                                self.x_count: count_X[batch_idx],
                                self.label_vec: label_vec[batch_idx],
                                self.mask_vec: mask_vec[batch_idx]})
                else:
                    dist, kmeans_loss, latent_repre = sess.run(
                        [self.latent_dist1, self.kmeans_loss, self.latent],
                        feed_dict={
                            self.upper_threshold: np.array([upper_threshold]),
                            self.lower_threshold: np.array([lower_threshold]),
                            self.sf_layer: size_factor,
                            self.x: X,
                            self.y: onehot_Y,
                            self.batch: batch_mat,
                            self.x_count: count_X,
                            self.label_vec: label_vec,
                            self.mask_vec: mask_vec})
                    Y_pred = np.argmin(dist, axis=1)
                    Y_pred_source = Y_pred[batch_label == 0]
                    Y_pred_target = Y_pred[batch_label != 0]
                    target_accuracy = np.around(cluster_acc(Y_target, Y_pred_target), 4)
                    target_ARI = np.around(adjusted_rand_score(Y_target, Y_pred_target), 4)
                    print("in the {}-th epoch, the target cluster accuracy is {}".format(i + 1, target_accuracy))
                    print("in the {}-th epoch, the target cluster ARI is {}".format(i + 1, target_ARI))
                    if np.sum(Y_pred_target != last_pred_target) / len(last_pred_target) < tol:
                        break
                    else:
                        last_pred_target = Y_pred_target

        sess.close()

        target_accuracy = np.around(cluster_acc(Y_target, Y_pred_target), 4)
        target_ARI = np.around(adjusted_rand_score(Y_target, Y_pred_target), 4)

        pred_cluster_num = len(np.unique(np.array(Y_pred_target)))
        print("The prediction cluster number on target data is {}".format(pred_cluster_num))
        annotated_source_accuracy, annotated_target_accuracy, target_annotation_label = annotation(cellname_source, cellname_target,
                                                                                                   Y_pred_source, Y_pred_target)

        target_prediction_matrix = pd.DataFrame({"true label": Y_target, "true cell type": cellname_target,
                                               "cluster label": Y_pred_target, "annotation cell type": target_annotation_label})

        t2 = time.time()
        print("The total consuming time for the whole model training is {}".format(t2 - t1))
        return target_accuracy, target_ARI, annotated_target_accuracy, target_prediction_matrix









