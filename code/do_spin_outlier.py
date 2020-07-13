import pandas as pd

import torch.cuda
import torch.optim as optim
from sklearn.preprocessing import normalize

from config import *
from do_outlier import *
from tool import compute_label_sim, parse_sklearn_log
from model_torch import CapsAll


def train_spin_outlier(data, config, seg, caption=''):

    print(">> training for zero-shot spin <<")
    device = config['device']
    config_outlier = config['outlier']
    config_spin = config['spin']
    batch_size = config['batch_size']
    encoder, classifier, tr_emb_outlier = seg[0], seg[1], seg[2]
    encoder.eval()
    classifier.eval()

    # initialization
    embedding = data['embedding'].to(device)
    algorithm = get_anomaly_algorithms(config_outlier['algorithm_key'][0],
                                       config_outlier['outlier_fraction'])
    spin = CapsAll(config, config_spin, embedding).to(device)
    optimizer = optim.Adam(spin.parameters(), lr=config_spin['learning_rate'])
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config_spin['lr_step_size'],
                                          gamma=config_spin['lr_gamma'])

    # train
    print('>> training begin <<')
    n_batch = data['n_tr'] // batch_size + 1
    for i_epoch in range(config_spin['n_epoch']):

        scheduler.step()
        spin.train()
        acc_avg = 0.0
        epoch_start_time = time.time()
        tr_emb_spin = []
        for i_batch in range(n_batch):
            # make batch data
            index = range(i_batch * batch_size, min((i_batch + 1) * batch_size, data['n_tr']))
            batch_x = data['x_tr'][index].to(device)
            batch_y = data['y_tr'][index].to(device)
            batch_ind = data['y_ind'][index].to(device)

            # model update
            spin(batch_x)
            tr_emb_spin.append(spin.sentence_reduction.detach().cpu().numpy())
            loss_val = spin.loss(batch_ind)
            optimizer.zero_grad()
            loss_val.backward()
            optimizer.step()

            # train predication
            batch_pred = torch.argmax(spin.logits, dim=1)
            correct = (batch_pred == batch_y.long()).sum().item()
            acc_avg += correct / batch_pred.shape[0]

        acc_avg = acc_avg / n_batch
        train_time = time.time() - epoch_start_time
        print("%s || epoch:%d || loss:%.4f || acc:%.4f || tr_time:%4f" %
              (caption, i_epoch, loss_val.item(), acc_avg, train_time))

        if i_epoch % config['test_step'] == 0:
            print('>> fit outlier detection <<')
            # fit outlier detection
            if config['combine_mode'] in ['merge', 'combine']:
                tr_emb = np.concatenate(tr_emb_spin, axis=0)
                if config['combine_mode'] == 'combine':
                    tr_emb = np.concatenate([tr_emb, tr_emb_outlier], axis=1)
            else:
                tr_emb = tr_emb_outlier
            # algorithm.fit(tr_emb)

            # sep
            algorithm.fit(tr_emb_outlier)
            perform_sep = test_spin_outlier(data, spin, algorithm, device, batch_size,
                                        config_spin['sim_scale'], config['n_seen_class'],
                                        'sep', config_outlier['outlier_outp'],
                                        config_outlier['percentile_min'], config_outlier['percentile_max'],
                                        encoder)
            # combine
            algorithm.fit(tr_emb)
            perform_combine = test_spin_outlier(data, spin, algorithm, device, batch_size,
                                        config_spin['sim_scale'], config['n_seen_class'],
                                        'combine', config_outlier['outlier_outp'],
                                        config_outlier['percentile_min'], config_outlier['percentile_max'],
                                        encoder)

            # if config['combine_mode'] in ['combine', 'sep']:
            #     perform = test_spin_outlier(data, spin, algorithm, device, batch_size,
            #                                 config_spin['sim_scale'], config['n_seen_class'],
            #                                 config['combine_mode'], config_outlier['outlier_outp'],
            #                                 config_outlier['percentile_min'], config_outlier['percentile_max'],
            #                                 encoder)
            # else:
            #     perform = test_spin_outlier(data, spin, algorithm, device, batch_size,
            #                                 config_spin['sim_scale'], config['n_seen_class'],
            #                                 config['combine_mode'], config_outlier['outlier_outp'],
            #                                 config_outlier['percentile_min'], config_outlier['percentile_max'])
    return perform_sep, perform_combine


def test_spin_outlier(data, spin, algorithm, device, batch_size,
                      sim_scale, s_cnum, combine_mode='sep',
                      outlier_outp='predict', percentile_min=25, percentile_max=75,
                      encoder=None):

    # prepare log
    metric_list = ['pre_seen', 'rec_seen', 'f1_seen',
                   'pre_unseen', 'rec_unseen', 'f1_unseen',
                   'pre_all', 'rec_all', 'f1_all',
                   'acc_outlier', 'acc_k_plus1']

    metric_list = (
            # ['macro_' + v for v in metric_list] +
            ['micro_' + v for v in metric_list] +
            ['acc_outlier']
    )

    # index_list = ['ori', 'soft', 'hard']
    index_list = ['ori', 'hard']
    perform = pd.DataFrame(pd.DataFrame(index=index_list, columns=metric_list))

    print('>> spin outlier {} test start <<'.format(combine_mode))

    spin.eval()
    try:
        sim = data['sim'].to(device)
    except:
        sim = torch.from_numpy(get_sim(data, sim_scale)).float().to(device)

    n_batch = data['n_te'] // batch_size + 1
    # te_pred_y_with_outlier = []
    te_pred_y_soft = []
    te_pred_outlier_soft = []
    te_pred_y_hard = []
    te_pred_outlier_hard = []
    te_pred_y_ori = torch.LongTensor([])
    with torch.no_grad():
        for i_batch in range(n_batch):
            index = range(i_batch * batch_size, min((i_batch + 1) * batch_size, data['n_te']))
            batch_x = data['x_te'][index].to(device)

            # model forward
            spin.construct_unseen_CapW(sim)
            te_logits = spin.predict(batch_x).cpu()
            if combine_mode in ['sep', 'combine']:
                te_emb_outlier = encoder(batch_x).detach().cpu().numpy()

            # outlier identification
            if combine_mode in ['merge', 'combine']:
                te_emb = spin.sentence_reduction.detach().cpu().numpy()
                if combine_mode == 'combine':
                    te_emb = np.concatenate((te_emb, te_emb_outlier), axis=1)
                    # te_emb = np.concatenate((te_emb[:, :s_cnum], te_emb_outlier), axis=1)
            else:
                te_emb = te_emb_outlier

            # pred_outlier = getattr(algorithm, outlier_outp)(te_emb)
            # te_pred_with_outlier = predict_with_outlier_scores(pred_outlier, te_logits, s_cnum,
            #                                                    percentile_min, percentile_max)
            # te_pred_y_with_outlier.append(te_pred_with_outlier)
            # ====================================================================================

            # soft outlier prediction
            # pred_soft_outlier = algorithm.score_samples(te_emb)
            # pred_outlier_soft = score_to_outlier(pred_soft_outlier, percentile_min, percentile_max,
            #                                      is_soft=True)
            # te_pred_soft_outlier = predict_with_outlier_scores(pred_soft_outlier, te_logits, s_cnum,
            #                                                    percentile_min, percentile_max,
            #                                                    is_soft=True)
            # te_pred_y_soft.append(te_pred_soft_outlier)
            # te_pred_outlier_soft.append(pred_outlier_soft)

            # hard outlier prediction
            pred_hard_outlier = algorithm.predict(te_emb)
            pred_outlier_hard = score_to_outlier(pred_hard_outlier, percentile_min, percentile_max,
                                                 is_soft=False)
            te_pred_hard_outlier = predict_with_outlier_scores(pred_hard_outlier, te_logits, s_cnum,
                                                               percentile_min, percentile_max,
                                                               is_soft=False)
            te_pred_y_hard.append(te_pred_hard_outlier)
            te_pred_outlier_hard.append(pred_outlier_hard)

            # origin outlier prediction
            te_pred_ori = torch.argmax(te_logits, dim=1)
            te_pred_y_ori = torch.cat([te_pred_y_ori, te_pred_ori])
            # ====================================================================================

        # prediction and evaluation
        # te_pred_y_with_outlier = np.concatenate(te_pred_y_with_outlier, axis=0)
        # print(classification_report(te_target_y, te_pred_y_with_outlier, digits=4))
        # log = precision_recall_fscore_support(te_target_y, te_pred_y_with_outlier)
        # ====================================================================================
        print('')

        log_ori = precision_recall_fscore_support(data['y_te'], te_pred_y_ori)
        pfm = parse_sklearn_log(log_ori, s_cnum)
        pfm['acc_outlier'] = 0
        pfm['acc_k_plus1'] = 0
        perform.loc['ori'] = pfm
        print('ori ||', pfm)
        print('')

        # te_pred_y_soft = np.concatenate(te_pred_y_soft, axis=0)
        # te_pred_outlier_soft = np.concatenate(te_pred_outlier_soft, axis=0)
        # log_soft = precision_recall_fscore_support(data['y_te'], te_pred_y_soft)
        # pfm = parse_sklearn_log(log_soft, s_cnum)
        # _, pfm['acc_outlier'] = evaluate_outlier(data['y_te'], te_pred_outlier_soft, s_cnum)
        # pfm['acc_k_plus1'] = evaluate_kplus(data['y_te'], te_pred_y_ori, te_pred_outlier_soft,
        #                                     s_cnum)['micro_rec_all']
        # perform.loc['soft'] = pfm
        # print('soft ||', pfm)
        # print('')

        te_pred_y_hard = np.concatenate(te_pred_y_hard, axis=0)
        te_pred_outlier_hard = np.concatenate(te_pred_outlier_hard, axis=0)
        log_hard = precision_recall_fscore_support(data['y_te'], te_pred_y_hard)
        pfm = parse_sklearn_log(log_hard, s_cnum)
        _, pfm['acc_outlier'] = evaluate_outlier(data['y_te'], te_pred_outlier_hard, s_cnum)
        pfm['acc_k_plus1'] = evaluate_kplus(data['y_te'], te_pred_y_ori,
                                            te_pred_outlier_hard, s_cnum)['micro_rec_all']
        perform.loc['hard'] = pfm
        print('hard ||', pfm)
        print('')
        # ====================================================================================

    return perform


def get_sim(data, sim_scale):
    # get unseen and seen categories similarity
    vec_seen = normalize(data['sc_vec'])
    vec_test = normalize(data['uc_vec'])
    if vec_seen.shape[0] < vec_test.shape[0]:
        no_seenc = vec_seen.shape[0]
        vec_test = vec_test[no_seenc:, ]
    sim = compute_label_sim(vec_test, vec_seen, sim_scale)
    return sim
