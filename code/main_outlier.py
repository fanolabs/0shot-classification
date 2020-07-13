import os
import json
from argparse import ArgumentParser
import pandas as pd
import torch
from numpy.random import seed

import input_data
from do_outlier import *

parser = ArgumentParser(description='config for outlier detection')
parser.add_argument('-r', '--repeat', default=5, type=int, help="repeat number in range (0, 10)")
parser.add_argument('-fd', '--feat_dim', default=12, type=int, help="feat_dim for outlier detection")
parser.add_argument('-eo', '--epoch_outlier', default=151, type=int,
                    help="number of epoch for outlier detection")
parser.add_argument('-data', '--dataset', default='SMP18', type=str,
                    help="choose dataset in (SNIP | SMP18 | ATIS)")
parser.add_argument('-cls', '--classifier', default='LGMLoss', type=str,
                    help="select classifier (LGMLoss | LSoftmax | MSP | DOC | LMCL)")
parser.add_argument('-cap', '--caption', default='visualMarker', type=str,
                    help="additional text caption for the experiment")
parser.add_argument('-s', '--seed', default=0, type=int,
                    help="seed (int)")
parser.add_argument('-p', '--prob', default=0.75, type=float,
                    help="proportion of seen classes")
args = parser.parse_args()


if __name__ == '__main__':

    data_setting = get_data_setting(args.dataset, args.prob)
    data_setting['freeze_class'] = False
    data_split = range(0, args.repeat)
    for i_split in data_split:

        sd = args.seed + i_split
        torch.cuda.manual_seed_all(sd)
        torch.manual_seed(sd)
        seed(sd)

        # load data
        data_setting['id_split'] = i_split
        data = input_data.read_datasets(data_setting)

        # load config
        config = get_config(data, args)
        set_config_outlier(config, args)
        if not os.path.exists(config['ckpt_dir']):
            print("making", config['ckpt_dir'])
            os.makedirs(config['ckpt_dir'])

        # prepare log
        metric_list = ['pre_unseen', 'rec_unseen', 'f1_unseen',
                       'pre_seen', 'rec_seen', 'f1_seen',
                       'pre_all', 'rec_all', 'f1_all']
        metric_list = (['macro_' + v for v in metric_list]
                       + ['micro_' + v for v in metric_list]
                       + ['acc_outlier']
                       )

        # # ================ outlier detection baselines ================
        # exp_key = 'outliers_useLabel'
        # config['outlier']['use_labels'] = True
        # # index_list = ['robust_covariance', 'one_class_svm', 'isolation_forest', 'lof']
        # index_list = ['lof']
        # Perform = pd.DataFrame(pd.DataFrame(index=index_list, columns=metric_list))
        # for algorithm_key in index_list:
        #     Perform.loc[algorithm_key] = train_outlier_detection(data, config, algorithm_key,
        #                                                          caption=algorithm_key)
        # filename = "%s%s_%s%s.csv" % (config['ckpt_dir'], exp_key, config['description'],
        #                               time.strftime('%y%m%d%I%M%S'))
        # Perform.to_csv(filename)

        # exp_key = 'outliers_noLabel'
        # config['outlier']['use_labels'] = False
        # index_list = ['robust_covariance', 'one_class_svm', 'isolation_forest', 'lof']
        # Perform = pd.DataFrame(pd.DataFrame(index=index_list, columns=metric_list))
        # for algorithm_key in index_list:
        #     Perform.loc[algorithm_key] = train_outlier_detection(data, config, algorithm_key,
        #                                                          caption=algorithm_key)
        # filename = "%s%s_%s%s.csv" % (config['ckpt_dir'], exp_key, config['description'],
        #                               time.strftime('%y%m%d%I%M%S'))
        # Perform.to_csv(filename)

        # # ================ different classifiers ================
        # exp_key = 'classifiers'
        # index_list = ['LGMLoss', 'LSoftmax']
        # Perform = pd.DataFrame(pd.DataFrame(index=index_list, columns=metric_list))
        # for key in index_list:
        #     config['outlier']['classifier'] = key
        #     Perform.loc[key] = train_outlier_detection(data, config)
        # filename = "%s%s_%s%s.csv" % (config['ckpt_dir'], exp_key, config['description'],
        #                               time.strftime('%y%m%d%I%M%S'))
        # Perform.to_csv(filename)
        # config['outlier']['classifier'] = args.classifier
        #
        # # ================ dim_feature ================
        exp_key = 'dimFeature'
        index_list = [4, 12, 64, 128]
        config['outlier']['classifier'] = 'LGMLoss'

        exp_key = 'dim_useLabel'
        config['outlier']['use_labels'] = True
        Perform = pd.DataFrame(pd.DataFrame(index=index_list, columns=metric_list))
        for key in index_list:
            config['outlier']['d_r'] = key
            Perform.loc[key] = train_outlier_detection(data, config, caption=key)
        filename = "%s%s_%s%s.csv" % (config['ckpt_dir'], exp_key, config['description'],
                                      time.strftime('%y%m%d%I%M%S'))
        Perform.to_csv(filename)

        exp_key = 'dim_noLabel'
        config['outlier']['use_labels'] = False
        Perform = pd.DataFrame(pd.DataFrame(index=index_list, columns=metric_list))
        for key in index_list:
            config['outlier']['d_r'] = key
            Perform.loc[key] = train_outlier_detection(data, config, caption=key)
        filename = "%s%s_%s%s.csv" % (config['ckpt_dir'], exp_key, config['description'],
                                      time.strftime('%y%m%d%I%M%S'))
        Perform.to_csv(filename)

        # # ================ add label as center ================
        # exp_key = 'useLabel'
        # index_list = [True, False]
        # Perform = pd.DataFrame(pd.DataFrame(index=index_list, columns=metric_list))
        # for key in index_list:
        #     config['outlier']['use_labels'] = key
        #     Perform.loc[key] = train_outlier_detection(data, config,  caption='usL' if key else 'noL')
        # filename = "%s%s_%s%s.csv" % (config['ckpt_dir'], exp_key, config['description'],
        #                               time.strftime('%y%m%d%I%M%S'))
        # Perform.to_csv(filename)
        #
        # # ================ hard or soft ================
        # exp_key = 'HardOSoft'
        # index_list = [True, False]
        # Perform = pd.DataFrame(pd.DataFrame(index=index_list, columns=metric_list))
        # for key in index_list:
        #     config['outlier']['is_soft'] = key
        #     if config['outlier']['is_soft']:
        #         config['outlier']['outlier_outp'] = 'score_samples'
        #         config['outlier']['percentile_min'] = 25
        #         config['outlier']['percentile_max'] = 75
        #     else:
        #         config['outlier']['outlier_outp'] = 'predict'
        #         config['outlier']['percentile_min'] = 0
        #         config['outlier']['percentile_max'] = 0
        #     Perform.loc[key] = train_outlier_detection(data, config)
        # filename = "%s%s_%s%s.csv" % (config['ckpt_dir'], exp_key, config['description'],
        #                               time.strftime('%y%m%d%I%M%S'))
        # Perform.to_csv(filename)

        # ================ how soft ================
        # exp_key = 'howSoft'
        # index_list = [20, 25, 30, 35, 40, 45, 50]
        # Perform = pd.DataFrame(pd.DataFrame(index=index_list, columns=metric_list))
        # for key in index_list:
        #     config['outlier']['is_soft'] = key
        #     if config['outlier']['is_soft']:
        #         config['outlier']['outlier_outp'] = 'score_samples'
        #         config['outlier']['percentile_min'] = key
        #         config['outlier']['percentile_max'] = 100 - key
        #     else:
        #         config['outlier']['outlier_outp'] = 'predict'
        #         config['outlier']['percentile_min'] = 0
        #         config['outlier']['percentile_max'] = 0
        #     Perform.loc[key] = train_outlier_detection(data, config)
        # filename = "%s%s_%s%s.csv" % (config['ckpt_dir'], exp_key, config['description'],
        #                               time.strftime('%y%m%d%I%M%S'))
        # Perform.to_csv(filename)

        # ================ baselines_useLabel ================
        # exp_key = 'baselines_useLabel'
        # config['outlier']['use_labels'] = True
        # index_list = ['LGMLoss']
        # Perform = pd.DataFrame(pd.DataFrame(index=index_list, columns=metric_list))
        # for key in index_list:
        #     config['outlier']['classifier'] = key
        #     Perform.loc[key] = train_outlier_detection(data, config, caption=key)
        # filename = "%s%s_%s%s.csv" % (config['ckpt_dir'], exp_key, config['description'],
        #                               time.strftime('%y%m%d%I%M%S'))
        # Perform.to_csv(filename)
        #
        # # # ================ baselines_noLabel ================
        # exp_key = 'baselines_noLabel'
        # config['outlier']['use_labels'] = False
        # index_list = ['MSP', 'DOC', 'LSoftmax', 'LMCL', 'LGMLoss']
        # # index_list = ['MSP', 'DOC']
        # Perform = pd.DataFrame(pd.DataFrame(index=index_list, columns=metric_list))
        # for key in index_list:
        #     config['outlier']['classifier'] = key
        #     Perform.loc[key] = train_outlier_detection(data, config, caption=key)
        # filename = "%s%s_%s%s.csv" % (config['ckpt_dir'], exp_key, config['description'],
        #                               time.strftime('%y%m%d%I%M%S'))
        # Perform.to_csv(filename)
        # # ================ test_doc ================
        # exp_key = 'test_doc'
        # config['outlier']['use_labels'] = False
        # index_list = ['DOC']
        # Perform = pd.DataFrame(pd.DataFrame(index=index_list, columns=metric_list))
        # for key in index_list:
        #     config['outlier']['classifier'] = key
        #     Perform.loc[key] = train_outlier_detection(data, config, caption=key)
        # filename = "%s%s_%s%s.csv" % (config['ckpt_dir'], exp_key, config['description'],
        #                               time.strftime('%y%m%d%I%M%S'))
        # Perform.to_csv(filename)
        # # ================ lof parameter ================
        # index_list = [0.05, 0.1, 0.2, 0.4]
        #
        # exp_key = 'lof_param_noL'
        # config['outlier']['use_labels'] = False
        # Perform = pd.DataFrame(pd.DataFrame(index=index_list, columns=metric_list))
        # for key in index_list:
        #     config['outlier_fraction'] = key
        #     Perform.loc[key] = train_outlier_detection(data, config, caption=key)
        # filename = "%s%s_%s%s.csv" % (config['ckpt_dir'], exp_key, config['description'],
        #                               time.strftime('%y%m%d%I%M%S'))
        # Perform.to_csv(filename)
        #
        # exp_key = 'lof_param_L'
        # config['outlier']['use_labels'] = True
        # Perform = pd.DataFrame(pd.DataFrame(index=index_list, columns=metric_list))
        # for key in index_list:
        #     config['outlier_fraction'] = key
        #     Perform.loc[key] = train_outlier_detection(data, config, caption=key)
        # filename = "%s%s_%s%s.csv" % (config['ckpt_dir'], exp_key, config['description'],
        #                               time.strftime('%y%m%d%I%M%S'))
        # Perform.to_csv(filename)
        # # ================ lambda ================
        # index_list = [0, 0.1, 0.5, 1]
        #
        # exp_key = 'lambda_noLabel'
        # config['outlier']['use_labels'] = False
        # Perform = pd.DataFrame(pd.DataFrame(index=index_list, columns=metric_list))
        # for key in index_list:
        #     config['outlier']['lambda_'] = key
        #     Perform.loc[key] = train_outlier_detection(data, config, caption=key)
        # filename = "%s%s_%s%s.csv" % (config['ckpt_dir'], exp_key, config['description'],
        #                               time.strftime('%y%m%d%I%M%S'))
        # Perform.to_csv(filename)
        #
        # exp_key = 'lambda_useLabel'
        # config['outlier']['use_labels'] = True
        # Perform = pd.DataFrame(pd.DataFrame(index=index_list, columns=metric_list))
        # for key in index_list:
        #     config['outlier']['lambda_'] = key
        #     Perform.loc[key] = train_outlier_detection(data, config, caption=key)
        # filename = "%s%s_%s%s.csv" % (config['ckpt_dir'], exp_key, config['description'],
        #                               time.strftime('%y%m%d%I%M%S'))
        # Perform.to_csv(filename)

        # # ================ margin ================
        # index_list = [0, 0.5, 1, 2, 4]
        #
        # exp_key = 'lambda_noLabel'
        # config['outlier']['use_labels'] = False
        # Perform = pd.DataFrame(pd.DataFrame(index=index_list, columns=metric_list))
        # for key in index_list:
        #     config['outlier']['margin'] = key
        #     Perform.loc[key] = train_outlier_detection(data, config, caption=key)
        # filename = "%s%s_%s%s.csv" % (config['ckpt_dir'], exp_key, config['description'],
        #                               time.strftime('%y%m%d%I%M%S'))
        # Perform.to_csv(filename)
        #
        # exp_key = 'lambda_useLabel'
        # config['outlier']['use_labels'] = True
        # Perform = pd.DataFrame(pd.DataFrame(index=index_list, columns=metric_list))
        # for key in index_list:
        #     config['outlier']['margin'] = key
        #     Perform.loc[key] = train_outlier_detection(data, config, caption=key)
        # filename = "%s%s_%s%s.csv" % (config['ckpt_dir'], exp_key, config['description'],
        #                               time.strftime('%y%m%d%I%M%S'))
        # Perform.to_csv(filename)

        # ================ lof k ================
        # index_list = [5, 10, 20, 40]
        # config['outlier']['classifier'] = 'LGMLoss'
        #
        # exp_key = 'k_noLabel'
        # config['outlier']['use_labels'] = False
        # Perform = pd.DataFrame(pd.DataFrame(index=index_list, columns=metric_list))
        # for key in index_list:
        #     config['outlier']['k'] = key
        #     Perform.loc[key] = train_outlier_detection(data, config, caption=key)
        # filename = "%s%s_%s%s.csv" % (config['ckpt_dir'], exp_key, config['description'],
        #                               time.strftime('%y%m%d%I%M%S'))
        # Perform.to_csv(filename)
        #
        # exp_key = 'k_uselabel'
        # config['outlier']['use_labels'] = True
        # Perform = pd.DataFrame(pd.DataFrame(index=index_list, columns=metric_list))
        # for key in index_list:
        #     config['outlier']['k'] = key
        #     Perform.loc[key] = train_outlier_detection(data, config, caption=key)
        # filename = "%s%s_%s%s.csv" % (config['ckpt_dir'], exp_key, config['description'],
        #                               time.strftime('%y%m%d%I%M%S'))
        # Perform.to_csv(filename)
        #
        # exp_key = 'k_lmcl'
        # config['outlier']['classifier'] = 'LMCL'
        # Perform = pd.DataFrame(pd.DataFrame(index=index_list, columns=metric_list))
        # for key in index_list:
        #     config['outlier']['k'] = key
        #     Perform.loc[key] = train_outlier_detection(data, config, caption=key)
        # filename = "%s%s_%s%s.csv" % (config['ckpt_dir'], exp_key, config['description'],
        #                               time.strftime('%y%m%d%I%M%S'))
        # Perform.to_csv(filename)

    # output config
    filename = "%sconfig_%s.json" % (config['ckpt_dir'], config['description'])
    config['device'] = str(config['device'])
    with open(filename, 'w') as f:
        json.dump(config, f)

    filename = "%sdatasetting_%s.json" % (config['ckpt_dir'], config['description'])
    with open(filename, 'w') as f:
        json.dump(data_setting, f)
