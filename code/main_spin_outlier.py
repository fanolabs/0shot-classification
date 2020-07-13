import os
import json
from argparse import ArgumentParser

from config import *
from input_data import read_datasets
from do_spin_outlier import train_spin_outlier

from do_outlier import *

parser = ArgumentParser(description='seg + recaps')
parser.add_argument('-r', '--repeat', default=10, type=int,
                    help="repeat number in range (0, 10)")
parser.add_argument('-fd', '--feat_dim', default=12, type=int,
                    help="feat_dim for outlier detection")
parser.add_argument('-eo', '--epoch_outlier', default=151, type=int,
                    help="number of epoch for outlier detection")
parser.add_argument('-es', '--epoch_spin', default=101, type=int,
                    help="number of epoch for outlier detection")
parser.add_argument('-data', '--dataset', default='SMP18', type=str,
                    help="choose dataset in (SNIP | SMP18)")
parser.add_argument('-cls', '--classifier', default='LGMLoss', type=str,
                    help="select classifier (LGMLoss | LSoftmax)")
parser.add_argument('-cap', '--caption', default='recapsEff', type=str,
                    help="additional text caption for the experiment")
parser.add_argument('-m', '--combine_mode', default='combine', type=str,
                    help="(merge | sep | combine)")
parser.add_argument('-p', '--prob', default=0.7, type=float,
                    help="proportion of seen classes")
args = parser.parse_args()


if __name__ == '__main__':

    data_setting = get_data_setting(args.dataset)
    data_setting['freeze_class'] = True
    data_split = range(0, args.repeat)
    for i_split in data_split:

        # load data
        data_setting['id_split'] = i_split
        data = read_datasets(data_setting)

        # load config
        config = get_config(data, args)
        set_config_combine(config, args)

        print('---- %s game start ----' % config['description'])

        config['ckpt_dir'] = config['ckpt_dir'].replace(args.combine_mode, "")
        if not os.path.exists(config['ckpt_dir']):
            print("making", config['ckpt_dir'])
            os.makedirs(config['ckpt_dir'])

        # train and test
        # for mode in ['sep', 'merge', 'combine']:
        # ================ useLabel ================
        exp_key = 'useLabel'
        config['outlier']['use_labels'] = True
        seg = train_outlier_detection(data, config, caption=exp_key,
                                      if_test=True, is_block=True)
        perform_sep, perform_combine = train_spin_outlier(data, config, seg, caption=exp_key)
        filename = "%s%s_%s%s_sep.csv" % (config['ckpt_dir'], exp_key, config['description'],
                                          time.time())
        perform_sep.to_csv(filename)
        filename = "%s%s_%s%s_combine.csv" % (config['ckpt_dir'], exp_key, config['description'],
                                      time.time())
        perform_combine.to_csv(filename)

        # ================ noLabel ================
        exp_key = 'noLabel'
        config['outlier']['use_labels'] = False
        seg = train_outlier_detection(data, config, caption=exp_key,
                                      if_test=True, is_block=True)

        perform_sep, perform_combine = train_spin_outlier(data, config, seg, caption=exp_key)
        filename = "%s%s_%s%s_sep.csv" % (config['ckpt_dir'], exp_key, config['description'],
                                      time.time())
        perform_sep.to_csv(filename)
        filename = "%s%s_%s%s_combine.csv" % (config['ckpt_dir'], exp_key, config['description'],
                                      time.time())
        perform_combine.to_csv(filename)

    # output config and summary log
    config['device'] = str(config['device'])
    filename = "%sconfig_%s.json" % (config['ckpt_dir'], config['description'])
    with open(filename, 'w') as f:
        json.dump(config, f)

    filename = "%sdatasetting_%s.json" % (config['ckpt_dir'], config['description'])
    with open(filename, 'w') as f:
        json.dump(data_setting, f)
