import numpy as np
import scipy.spatial.distance as ds

import collections


def norm_matrix(matrix):
    """Nomralize matrix by column
            input: numpy array, dtype = float32
            output: normalized numpy array, dtype = float32
    """

    # check dtype of the input matrix
    np.testing.assert_equal(type(matrix).__name__, 'ndarray')
    np.testing.assert_equal(matrix.dtype, np.float32)
    # axis = 0  across rows (return size is  column length)
    row_sums = matrix.sum(axis = 1) # across columns (return size = row length)

    # Replace zero denominator
    row_sums[row_sums == 0] = 1
    #start:stop:step (:: === :)
    #[:,np.newaxis]: expand dimensions of resulting selection by one unit-length dimension
    # Added dimension is position of the newaxis object in the selection tuple
    norm_matrix = matrix / row_sums[:, np.newaxis]

    return norm_matrix


def replace_nan(X):
    """
    replace nan and inf o 0
    """
    X[np.isnan(X)] = 0
    X[np.isnan(X)] = 0

    return X


def compute_label_sim(sig_y1, sig_y2, sim_scale):
    """
    compute class label similarity
    """
    dist = ds.cdist(sig_y1, sig_y2, 'euclidean')
    dist = dist.astype(np.float32)
    Sim = np.exp(-np.square(dist) * sim_scale)
 #   s = np.sum(Sim, axis=1)
#  Sim = replace_nan(Sim/ s[:, None])

    return Sim


def flatten(lst):
    for item in lst:
        if isinstance(item, collections.Iterable) and not isinstance(item, (str, bytes)):
            yield from flatten(item)
        else:
            yield item


def get_score(cm):
    fs = []
    n_class = cm.shape[0]
    for idx in range(n_class):
        TP = cm[idx][idx]
        r = TP / cm[idx].sum() if cm[idx].sum() != 0 else 0
        p = TP / cm[:, idx].sum() if cm[:, idx].sum() != 0 else 0
        f = 2 * r * p / (r + p) if (r + p) != 0 else 0
        fs.append(f)

    f = np.mean(fs).round(4)
    f_seen = np.mean(fs[:-1]).round(4)
    f_unseen = round(fs[-1], 4)
    print("F1: || Overall:%f || Seen:%f || Uneen:%f" %
          (f, f_seen, f_unseen))

    return f, f_seen, f_unseen


def parse_sklearn_log(log, split_point=-1, digits=4, mode='micro'):
    """
    :param log: skleran.precision_recall_fscore_support output (average=None)
    :param split_point: int, split to seen and unseen group
    :param digits: int, decimal limits
    :param mode: (micro | macro)
    :return: {metric:value}
    """

    support = log[3]
    precision = log[0]
    recall = log[1]
    f1 = log[2]

    if mode == 'macro':
        count_seen = split_point
        count_unseen = support.shape[0]-split_point
        count_all = support.shape[0]
    else:
        precision = precision * support
        recall = recall * support
        f1 = f1 * support
        count_seen = support[:split_point].sum()
        count_unseen = support[split_point:].sum()
        count_all = support.sum()

    perform = dict(
        pre_seen=precision[:split_point].sum()/count_seen,
        rec_seen=recall[:split_point].sum()/count_seen,
        f1_seen=f1[:split_point].sum()/count_seen,
        pre_unseen=precision[split_point:].sum()/count_unseen,
        rec_unseen=recall[split_point:].sum()/count_unseen,
        f1_unseen=f1[split_point:].sum()/count_unseen,
        pre_all=precision.sum()/count_all,
        rec_all=recall.sum()/count_all,
        f1_all=f1.sum()/count_all,
    )

    perform = {(mode + '_' + k): round(v, digits) for k, v in perform.items()}

    return perform
