import torch
import time

from sklearn.covariance import EllipticEnvelope
from sklearn import svm
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

torch.cuda.current_device()  # for a wired RuntimeError: CUDA error: unknown error bug


def get_config(data, args):

    config = dict()

    # basic
    config['n_repeat'] = args.repeat
    config['batch_size'] = 50  # bsz for spin, times 4 for outlier
    config['use_gpu'] = True
    config['test_step'] = 50

    # dataset related
    config['dataset'] = data['dataset']
    config['seen_prob'] = args.prob
    config['n_tr_sample'] = data['x_tr'].shape[0]
    config['n_te_sample'] = data['x_te'].shape[0]
    config['n_seen_class'] = len(data['seen_class'])
    config['n_unseen_class'] = len(data['unseen_class'])
    config['n_vocab'] = data['n_vocab']
    config['d_emb'] = data['d_emb']
    config['text_represent'] = data['text_represent']
    config['key_pretrained'] = data['key_pretrained']

    # training
    config['device'] = torch.device("cuda" if torch.cuda.is_available() & config['use_gpu'] else "cpu")

    # outp
    config['experiment_time'] = time.strftime('%y%m%d%I%M%S')

    return config


def set_config_outlier(config, args):

    config['outlier'] = dict(
        n_epoch=args.epoch_outlier,

        # model: outlier detection
        classifier=args.classifier,  # (LGMLoss | LSoftmax)
        use_labels=True,  # use label description as gaussian center
        d_lstm=128,  # lstm softmax
        n_layer=2,  # lstm layer count
        dropout_lstm=0.3,  # dropout rate for lstm
        dropout_fc=0.3,  # dropout rate for fully connected layers
        d_a=10,  # attention dim
        d_r=args.feat_dim,  # reduction dim
        coef_ce=1,  # coefficient for cross entropy loss in the classifier
        coef_cls=1,  # coefficient for classifier specific loss
        lambda_=0.5,  # lambda in LGMLoss
        margin=1,  # alpha in LGMLoss

        # outlier detection
        is_soft=True,
        # algorithm_key (robust_covariance | one_class_svm | isolation_forest | lof)
        # algorithm_key=['robust_covariance', 'one_class_svm', 'isolation_forest', 'lof'],
        algorithm_key=['lof'],
        outlier_fraction=0.05,  # refer to related sklearn documents
        k=5,

        # training
        learning_rate=0.001,

        # output
        visualize_emb=False,  # visualize embedding feeding to outlier detection algorithm (seen/unseen classes)
        visualize_outlier=False,  # visualize outlier detection scatter (is or not outlier)
    )

    str_prob = str(config['seen_prob']).replace('0.', '')
    config['description'] = "%s_%s_%d" % (config['dataset'], str_prob,
                                          config['outlier']['d_r'])
    if config['outlier']['is_soft']:
        config['outlier']['outlier_outp'] = 'score_samples'  # class method name of sklearn.<algorithm> prediction
        config['outlier']['percentile_min'] = 25  # int, percentile lower bound in range [0,100]
        config['outlier']['percentile_max'] = 75  # int, percentile upper bound in range [0,100]
        config['description'] = config['description'] + "_%dth%d" % (config['outlier']['percentile_min'],
                                                                     config['outlier']['percentile_max'])
    else:
        config['outlier']['outlier_outp'] = 'predict'
        config['outlier']['percentile_min'] = 0
        config['outlier']['percentile_max'] = 0

    # output
    config['description'] = config['description'] + '_' + args.caption
    config['ckpt_dir'] = 'saved_models/' + config['description'] + '/'


def set_config_spin(config, args):

    config['spin'] = dict(
        n_epoch=args.epoch_spin,

        # model: zero shot with spin
        dropout_emb=0.5,  # embedding dropout keep rate
        n_layer=2,  # default for bilstm
        d_lstm=32,  # lstm hidden size
        d_a=30,  # self-attention weight hidden units number
        output_atoms=10,  # capsule output atoms
        r=8,  # attention head number
        n_routing=3,  # capsule routing number
        alpha=0.001,  # coefficient of self-attention loss SMP = 0.001
        margin=1,  # ranking loss margin
        sim_scale=0.35,  # sim scale SMP =0.35 SNIPS =0.15

        # training
        learning_rate=0.01,
        lr_step_size=20,  # SMP 20 SNIP 10
        lr_gamma=0.01,
    )

    # output
    # config['description'] = (config['dataset'] + '_' + args.caption)
    # config['ckpt_dir'] = 'saved_models/' + config['description'] + '/'
    pass


def set_config_combine(config, args):

    set_config_outlier(config, args)
    set_config_spin(config, args)
    config['combine_mode'] = args.combine_mode  # (merge | sep | combine)

    if config['combine_mode'] == 'merge':
        config['d_feat'] = config['spin']['r']
    elif config['combine_mode'] == 'combine':
        config['d_feat'] = config['spin']['r'] + args.feat_dim
    else:
        config['d_feat'] = args.feat_dim

    # output
    config['description'] = "%s_%s%d" % (config['description'], config['combine_mode'],
                                         config['d_feat'])
    config['ckpt_dir'] = 'saved_models/' + config['description'] + '/'


def get_data_setting(choose_dataset, seen_class_prob=0.7):

    data_setting = dict()
    data_setting['dataset'] = choose_dataset
    data_setting['test_mode'] = 'general'  # (standard | general)
    data_setting['freeze_class'] = False

    data_setting['seen_class_prob'] = seen_class_prob
    data_setting['sample_in_test_prob'] = 0.3
    # text representation
    # ref https://huggingface.co/transformers/pretrained_models.html for transfomers
    data_setting['text_represent'] = 'w2v'  # (w2v, transformers)

    if choose_dataset == 'SNIP':
        data_setting['data_prefix'] = '../data/SNIP/'
        data_setting['dataset_name'] = 'dataSNIP.txt'
        data_setting['wordvec_name'] = 'wiki.en.vec'
        data_setting['key_pretrained'] = 'bert-base-uncased'
        data_setting['sim_name_withOS'] = 'SNIP_similarity_M_zscore.mat'
        data_setting['sim_name_withS'] = 'SNIP10seen.mat'
        data_setting['unseen_class'] = ['playlist', 'book']
        data_setting['label_order'] = ['search', 'movie', 'music', 'weather',
                                       'restaurant', 'playlist', 'book']

    if choose_dataset == "SMP18":
        data_setting['data_prefix'] = '../data/SMP18/'
        data_setting['dataset_name'] = 'dataSMP18.txt'
        data_setting['wordvec_name'] = 'sgns_merge_subsetSMP.txt'
        data_setting['key_pretrained'] = 'bert-base-chinese'
        data_setting['sim_name_withOS'] = 'SMP_unseen_similarity.mat'
        data_setting['sim_name_withS'] = 'SMP_seen_similarity_5.mat'
        data_setting['unseen_class'] = ['天气', '公交', 'app', '飞机', '电影', '音乐']
        data_setting['label_order'] = ["联络", "股票", "健康", "app", "电台", "翻译",
                                       "飞机", "电话", "谜语", "小说", "公交", "新闻",
                                       "抽奖", "音乐", "电影", "视频", "日程", "网站",
                                       "计算", "短信", "地图", "比赛", "诗歌", "火车",
                                       "时间", "天气", "email", "节目", "电视频道", "食谱"]
    if choose_dataset == "ATIS":
        data_setting['data_prefix'] = '../data/ATIS/'
        data_setting['dataset_name'] = 'dataATIS.txt'
        data_setting['wordvec_name'] = 'glove300_ATIS.txt'
        data_setting['key_pretrained'] = 'distilbert-base-uncased'
        data_setting['sim_name_withOS'] = 'SMP_unseen_similarity.mat'
        data_setting['sim_name_withS'] = 'SMP_seen_similarity_5.mat'
        data_setting['unseen_class'] = ['天气', '公交', 'app', '飞机', '电影', '音乐']
        data_setting['label_order'] = ["联络", "股票", "健康", "app", "电台", "翻译",
                                       "飞机", "电话", "谜语", "小说", "公交", "新闻",
                                       "抽奖", "音乐", "电影", "视频", "日程", "网站",
                                       "计算", "短信", "地图", "比赛", "诗歌", "火车",
                                       "时间", "天气", "email", "节目", "电视频道", "食谱"]

    return data_setting


def get_anomaly_algorithms(algorithm_key, k, outlier_fraction):

    anomaly_algorithms = dict(
        robust_covariance=EllipticEnvelope(contamination=outlier_fraction, support_fraction=0.9999),
        one_class_svm=svm.OneClassSVM(nu=outlier_fraction, kernel="rbf", gamma=0.1),
        isolation_forest=IsolationForest(behaviour='new', contamination=outlier_fraction, random_state=42),
        lof=LocalOutlierFactor(n_neighbors=k, contamination=outlier_fraction, novelty=True, n_jobs=-1),
    )
    return anomaly_algorithms[algorithm_key]


def get_info_transfomer(key_pretrained):
    transfomer_info = dict()
    transfomer_info['distilbert-base-uncased'] = dict(
        architecture='DistilBert',
        d_hidden=768,
    )
    transfomer_info['bert-base-uncased'] = dict(
        architecture='BERT',
        d_hidden=768,
    )
    transfomer_info['bert-base-chinese'] = dict(
        architecture='Bert',
        d_hidden=768,
    )
    return transfomer_info[key_pretrained]
