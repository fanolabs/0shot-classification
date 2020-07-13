import numpy as np
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from torch.nn.utils import clip_grad_norm_

from config import *
import model_new_intent as model_out
from tsne import tsne_class, tsne_outlier
from tool import parse_sklearn_log


def train_outlier_detection(data, config, algorithm_key='lof', if_test=True, is_block=False,
                            caption=''):

    print(">> training for embedding <<")

    (tr_emb, acc_outlier, perform_kplus) = (None, None, None)
    sep_outlier = dict(SEG=True, LGMLoss=True, LSoftmax=True, LMCL=True, MSP=False, DOC=False)

    config_outlier = config['outlier']
    device = config['device']
    batch_size = config['batch_size'] * (1 if data['dataset'] == 'SMP18' else 4)
    embedding = data['embedding'] if config['text_represent'] == 'w2v' else None
    class_token_ids = data['class_padded'].to(device)
    n_tr = data['y_tr'].shape[0]
    p_y = data['y_tr'].unique(return_counts=True)[1].float()/n_tr

    encoder = model_out.LSTMEncoder(config['d_emb'], config_outlier['d_lstm'], config_outlier['n_layer'],
                                    config_outlier['d_a'], config_outlier['d_r'], config['n_seen_class'],
                                    config['n_vocab'], config_outlier['dropout_lstm'],
                                    config_outlier['dropout_fc'], embedding=embedding,
                                    sep_outlier=sep_outlier[config_outlier['classifier']]).to(device)
    classifier = getattr(model_out, config_outlier['classifier'])(config['n_seen_class'],
                                                                  config_outlier['d_r']).to(device)
                                                                  # alpha=config_outlier['margin'],
                                                                  # lambda_=config_outlier['lambda_']).to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(classifier.parameters()),
                                 lr=config_outlier['learning_rate'])

    algorithm = get_anomaly_algorithms(algorithm_key, config_outlier['k'], config_outlier['outlier_fraction'])

    # training begin
    n_batch = (n_tr - 1) // batch_size + 1
    for i_epoch in range(config_outlier['n_epoch']):
        encoder.train()
        acc_avg = 0.0
        loss_avg = 0.0
        tr_emb = []
        for i_batch in range(n_batch):

            # make batch data
            index = range(i_batch * batch_size, min((i_batch + 1) * batch_size, n_tr))
            batch_x = data['x_tr'][index, :].to(device)
            batch_y = data['y_tr'][index].to(device)
            # model update
            batch_tr_emb = encoder(batch_x)
            tr_emb.append(batch_tr_emb.detach().cpu().numpy())
            class_emb = encoder(class_token_ids) if config['outlier']['use_labels'] else None

            batch_logits = classifier(batch_tr_emb, labels=batch_y.long(), device=device,
                                      class_emb=class_emb)
            loss = classifier.loss

            optimizer.zero_grad()
            loss.backward()
            # clip_grad_norm_(list(encoder.parameters()) + list(classifier.parameters()), 1)
            optimizer.step()

            # make prediction
            batch_pred = torch.argmax(batch_logits, 1)
            correct = (batch_pred == batch_y.long()).sum().item()

            # log
            acc_avg += correct / batch_pred.shape[0]
            loss_avg += loss.item()

        tr_emb = np.concatenate(tr_emb, axis=0)
        acc_avg = acc_avg/n_batch
        loss_avg = loss_avg/n_batch
        print('%s || epoch:%d || loss:%f || acc_tr:%f' % (caption, i_epoch, loss_avg, acc_avg))
        print('centroids %f' % (classifier.means-class_emb).abs().sum())
        if if_test & (i_epoch % config['test_step'] == 0):
            if sep_outlier[config_outlier['classifier']]:
                algorithm.fit(tr_emb)
            else:
                algorithm = classifier

            (te_emb, acc_outlier,
             perform_kplus) = test_outlier_detection(data['x_te'], data['y_te'],
                                                     encoder, classifier, algorithm,
                                                     config_outlier['is_soft'],
                                                     config_outlier['outlier_outp'],
                                                     config['n_seen_class'], batch_size, device,
                                                     config_outlier['percentile_min'],
                                                     config_outlier['percentile_max'],
                                                     i_epoch, algorithm_key, config['description'],
                                                     config['ckpt_dir'], config_outlier['visualize_outlier'],
                                                     caption=caption)

            if config_outlier['visualize_emb']:
                y_tr_np = data['y_tr'].cpu().clone().numpy()
                y_te_np = data['y_te'].cpu().clone().numpy()

                # visualize train
                # filename = "smp_pre_train_" + config['description'] + str(i_epoch)
                # tsne_class(tr_emb, y_tr_np, prefix=config['ckpt_dir'], filename=filename)
                # # visualize test
                # filename = "smp_test" + config['description'] + str(i_epoch)
                # tsne_class(te_emb, y_te_np, split_point=config['n_seen_class'],
                #            prefix=config['ckpt_dir'], filename=filename)
                # visualize all0
                # all_emb = np.concatenate([tr_emb, te_emb], axis=0)
                # all_y = np.concatenate((y_tr_np, y_te_np), axis=0)
                filename = "all_%s_%s%d" % (caption, config['description'], i_epoch)
                tsne_class(tr_emb, te_emb, y_tr_np, y_te_np,
                           split_point=config['n_seen_class'],
                           prefix=config['ckpt_dir'], filename=filename)

    if is_block:
        return encoder, classifier, tr_emb
    else:
        perform_collected = perform_kplus
        perform_collected['acc_outlier'] = acc_outlier
        return perform_collected


def test_outlier_detection(x_te, y_te, encoder, classifier, algorithm, is_soft, outlier_outp,
                           n_seen_class, batch_size, device, percentile_min=25, percentile_max=75,
                           i_epoch=0, alg_key='lof',
                           description='', prefix='./rst/', visualize=False, caption=''):

    print(">> test for outlier detection <<")
    encoder.eval()
    classifier.eval()
    n_te = y_te.shape[0]

    # test begin
    n_batch = (n_te - 1) // batch_size + 1
    te_emb = torch.tensor([]).to(device)
    te_logits = torch.tensor([]).to(device)
    with torch.no_grad():
        for i_batch in range(n_batch):
            # make batch data
            index = range(i_batch * batch_size, min((i_batch + 1) * batch_size, n_te))
            batch_x = x_te[index, :].to(device)
            # make prediction
            batch_tr_emb = encoder(batch_x)
            batch_logits = classifier(batch_tr_emb, device=device)

            te_emb = torch.cat((te_emb, batch_tr_emb), dim=0)
            te_logits = torch.cat((te_logits, batch_logits), dim=0)

    te_emb_np = te_emb.detach().cpu().clone().numpy()
    te_pred_outlier = getattr(algorithm, outlier_outp)(te_emb_np)
    te_pred_outlier = score_to_outlier(te_pred_outlier, percentile_min, percentile_max, is_soft)
    te_pred_class = torch.argmax(te_logits.cpu(), dim=1)

    te_truth_outlier, acc_outlier = evaluate_outlier(y_te, te_pred_outlier,
                                                     n_seen_class, description)
    perform_kplus = evaluate_kplus(y_te, te_pred_class, te_pred_outlier,
                                   n_seen_class, description)

    if visualize:
        filename = "outlier_%s_%s%d" % (caption, description, i_epoch)
        tsne_outlier(te_emb_np, te_pred_outlier, te_truth_outlier, prefix, filename)

    return te_emb_np, acc_outlier, perform_kplus


def predict_with_outlier_scores(pred_outlier_score, logits, split_point,
                                percentile_min=25, percentile_max=75, is_soft=True):
    """
    predict whether test data is outlier, predict in unseen classes if outlier,
    otherwise predict in seen classes.
    :param pred_outlier_score: np.array[bsz], output of score_samples
    :param logits: tensor[bsz, n_seen_class+n_unseen_class], n_seen_class should be
                at first <split_point> columns.
    :param split_point: int
    :param percentile_min: int, in range [0,100]
    :param percentile_max: int, in range [0,100]
    :param is_soft: int, if False, th_min=th_max=0,
    :return: np.array[bsz], final predicted labels
    """

    pred_outlier_soft = score_to_outlier(pred_outlier_score, percentile_min, percentile_max, is_soft)

    logits_np = logits.clone().numpy()
    pred_final = np.argmax(logits_np, 1)
    pred_final[pred_outlier_soft == 1] = np.argmax(logits_np[pred_outlier_soft == 1, :split_point], 1)
    pred_final[pred_outlier_soft == -1] = (np.argmax(logits_np[pred_outlier_soft == -1, split_point:], 1)
                                           + split_point)

    return pred_final


def score_to_outlier(outlier_score, percentile_min=25, percentile_max=75, is_soft=True):
    """
    :param outlier_score: np.array[n], output of sklearn.neighbors.LocalOutlierFactor.score_samples
    :param percentile_min: int, in range (0,100)
    :param percentile_max: int, in range (0,100)
    while the left turns to 0
    :param is_soft: int, if False, th_min=th_max=0,
    :return: np.array[n] in {-1(outliers), 0, 1(inliers)}
    """

    if is_soft:
        threshold_min = np.percentile(outlier_score, percentile_min)
        threshold_max = np.percentile(outlier_score, percentile_max)
    else:
        threshold_min = 0
        threshold_max = 0

    outlier_score_ = np.zeros(outlier_score.shape)
    outlier_score_[outlier_score < threshold_min] = -1
    outlier_score_[outlier_score >= threshold_max] = 1
    return outlier_score_


def evaluate_outlier(target_class, pred_outlier, k, caption='', digits=4):
    """
    :param target_class: tensor[n] in {1, -1}, target outlier (-1 as outlier)
    :param pred_outlier: np.array[n] in {1, -1}, predict outlier (-1 as outlier)
    :param k: int, seen class number
    :param caption: str, caption to print
    :param digits: int
    :return: acc_outlier and changed target_class_outlier(np.array[n] in {1, -1}) split by k
    """

    idx_eval = np.any([pred_outlier == 1, pred_outlier == -1], axis=0)
    n = pred_outlier.shape[0]
    n_confident = np.sum(idx_eval).item()

    target_class_outlier = target_class.clone().numpy()
    target_class_outlier = target_class_outlier[idx_eval]
    target_class_outlier[target_class_outlier < k] = 1
    target_class_outlier[target_class_outlier >= k] = -1

    pred_class_outlier = pred_outlier.copy()
    pred_class_outlier = pred_class_outlier[idx_eval]

    correct = np.sum((target_class_outlier == pred_class_outlier))
    acc_outlier = correct / n_confident
    acc_outlier = round(acc_outlier, digits)

    ccm_outlier = confusion_matrix(target_class_outlier, pred_class_outlier)
    print('>> %s outlier acc %.4f(%d/%d)<<' % (caption, acc_outlier, n_confident, n))
    print(ccm_outlier)

    return target_class_outlier, acc_outlier


def evaluate_kplus(target_class, pred_class, pred_outlier, k, caption='', digits=4):
    """
    :param target_class: tensor[n], target class index
    :param pred_class: tensor[n], predict class index
    :param pred_outlier: np.array[n] in {1, -1}, outlier prediction, -1 as outliers
    :param k: int, seen class number
    :param caption: str
    :param digits: int
    :return: dict({metric=value}), split to seen and unseen group by k,
        parsed sklearn.precision_recall_fscore_support returned log (average=None)
    """

    print('>> %s k+1 evaluation <<' % caption)

    target_class_kplus = target_class.clone().numpy()
    target_class_kplus[target_class_kplus > k] = k

    pred_class_kplus = pred_class.clone().numpy()
    pred_class_kplus[pred_outlier == -1] = k

    log = precision_recall_fscore_support(target_class_kplus, pred_class_kplus)
    perform_macro = parse_sklearn_log(log, split_point=k, digits=digits, mode='macro')
    perform_micro = parse_sklearn_log(log, split_point=k, digits=digits, mode='micro')
    perform = {**perform_macro, **perform_micro}
    print(perform)

    return perform
