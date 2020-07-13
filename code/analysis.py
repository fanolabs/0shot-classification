# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 13:11:18 2019

@author: floatsd
"""

import os
import pandas as pd

def summary_outliers(path, keyword, name=''):
    file = os.listdir(path)
    path_file = [os.path.join(path, f) for f in file if keyword in f] 
    pd_rst = [pd.read_csv(f, index_col=0) for f in path_file]
    rst_init = pd_rst[0]
    
    for rst in pd_rst[1:]:
        rst_init = rst_init.add(rst)
    
    rst_init = rst_init/len(pd_rst)
    rst_init.to_csv(path + '/' + name + '_' + keyword + '_summary.csv')

#%%
#prefix = "../rst/rollBack/"
#prefix = "saved_models/"
#ver = "SNIP_LGMLoss4_test"

#summary_outliers(prefix+ver, keyword='combine_noLabel', name=ver)
#summary_outliers(prefix+ver, keyword='merge_noLabel', name=ver)
#summary_outliers(prefix+ver, keyword='sep_noLabel', name=ver)
#
#summary_outliers(prefix+ver, keyword='combine_useLabel', name=ver)
#summary_outliers(prefix+ver, keyword='merge_useLabel', name=ver)
#summary_outliers(prefix+ver, keyword='sep_useLabel', name=ver)

#summary_outliers(prefix+ver, keyword='useLabel', name=ver)
    
#%%
#prefix = 'E:/bitbucket/GuangfengLOF/gf06_Intent_Detection/result/follow/'
#tag = 'SNIPS50'
#summary_outliers(prefix, keyword=tag, name=tag)
    
    
#%%
#prefix = 'C:/Users/floatsd/Documents/git/spin_outlier/code/saved_models/SMP18_LGMLoss12_25'
#tag = 'useLabel'
#summary_outliers(prefix, keyword=tag, name=tag)
#    
#tag = 'noLabel'
#summary_outliers(prefix, keyword=tag, name=tag)
#    
#%%
    
ver = 'SMP18_LGMLoss12_macro75baseline'
prefix = 'C:/Users/floatsd/Documents/git/spin_outlier/rst/macro/' + ver
tag = 'baselines_useLabel'
summary_outliers(prefix, keyword=tag, name=ver)
tag = 'baselines_noLabel'
summary_outliers(prefix, keyword=tag, name=ver)

tag = 'outliers_useLabel'
summary_outliers(prefix, keyword=tag, name=ver)
tag = 'outliers_noLabel'
summary_outliers(prefix, keyword=tag, name=ver)
    