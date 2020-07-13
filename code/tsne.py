from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np


def tsne_class(tr_emb, te_emb, y_tr_np, y_te_np,
               split_point=None, prefix='./rst/', filename='tsne'):
    """
    tsne and plot scatter, use different colormap and marker for class group larger
    or smaller than the split point
    :param inp: np.array[n, feature_dim]
    :param truth: np.array[n], label index
    :param split_point: int
    :param prefix: str
    :param filename: str, filename without any extension
    :return: None, write the plot to prefix/filename.png
    """

    print("tsne", filename)
    inp = np.concatenate([tr_emb, te_emb], axis=0)
    truth = np.concatenate([y_tr_np, y_te_np], axis=0)
    n_tr = tr_emb.shape[0]
    if split_point is None:
        split_point = np.max(truth)

    # tsne fitting
    tsne = TSNE(n_components=2)
    tsne.fit_transform(inp)
    emb = tsne.embedding_

    # split data
    emb_tr = emb[:n_tr, :]
    emb_te = emb[n_tr:, :]
    emb_te_seen = emb_te[y_te_np < split_point, :]
    emb_te_unseen = emb_te[y_te_np >= split_point, :]

    label_tr = y_tr_np
    label_te_seen = y_te_np[y_te_np < split_point]
    label_te_unseen = y_te_np[y_te_np >= split_point]

    # plot
    cm_seen = plt.cm.get_cmap('YlOrBr')
    cm_unseen = plt.cm.get_cmap('viridis')
    fig, ax = plt.subplots()
    pt1, = ax.scatter(emb_tr[:, 0], emb_tr[:, 1], c='gray', label=label_tr)
    pt2, = ax.scatter(emb_te_seen[:, 0], emb_te_seen[:, 1],
                      c=label_te_seen, cmap=cm_seen, marker=',', label=label_te_seen)
    pt3, = ax.scatter(emb_te_unseen[:, 0], emb_te_unseen[:, 1],
                      c=label_te_unseen, cmap=cm_unseen, marker='x', label=label_te_unseen)

    # plt.legend((pt1, pt2, pt3), ('train', 'test_seen', 'test_unseen'))

    # outp
    plt.title(filename)
    plt.savefig(prefix + filename + ".png")
    plt.close()


def tsne_outlier(inp, pred_outlier, truth_outlier,
                 prefix='./rst/', filename='tsneOne'):
    """
    tsne and plot scatter, different color and marker for tp, tn, fp, fn groups
    :param inp: np.array[n, feature_dim]
    :param pred_outlier: np.array[n] in {1, -1} # 0430, 输出是0, -1, 改成了0,-1
    :param truth_outlier: np.array[n] in {1, -1}
    :param prefix: str
    :param filename: str, filename without any extension
    :return: None, write the plot to prefix/filename.png
    """

    print("tsne", filename)

    # tsne fitting
    tsne = TSNE(n_components=2)
    tsne.fit_transform(inp)
    emb = tsne.embedding_

    # temp solution for pred_outlier
    pred_outlier[pred_outlier == 0] = 1

    # get confusion matrix
    flag_t = pred_outlier == truth_outlier
    flag_p = pred_outlier == -1

    # split dataset
    emb_tp = emb[flag_t & flag_p]
    emb_tn = emb[flag_t & ~flag_p]
    emb_fp = emb[~flag_t & flag_p]
    emb_fn = emb[~flag_t & ~flag_p]

    # plot
    fig, ax = plt.subplots()
    ax.scatter(emb_tp[:, 0], emb_tp[:, 1], c='tomato')
    ax.scatter(emb_tn[:, 0], emb_tn[:, 1], c='r', marker='x')
    ax.scatter(emb_fp[:, 0], emb_fp[:, 1], c='gray')
    ax.scatter(emb_fn[:, 0], emb_fn[:, 1], c='k', marker='x')

    # outp
    plt.title(filename)
    plt.savefig(prefix + filename + ".png")
    plt.close()
