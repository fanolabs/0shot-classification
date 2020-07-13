""" input data preprocess.
"""
import scipy.io as sio
import random

import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import math
from gensim.models.keyedvectors import KeyedVectors

import tool


def load_data(file_path):
    """
    :param file_path: str, data file name with extension
                    format: <label>\t<query>, split word by space, one sample one line
    :return: x_text: [str], y_text: [str],
            y: torch.tensor[index], class_dict[class: class_index]
    """
    print(">> load data %s" % file_path)
    x_text = []
    y_text = []
    class_dict = {}
    for line in open(file_path, 'rb'):
        arr = str(line.strip(), 'utf-8')
        arr = arr.split('\t')
        x_text.append(arr[1])
        y_text.append(arr[0])
        if not (y_text[-1] in class_dict):
            class_dict[y_text[-1]] = len(class_dict)

    y = [class_dict[label] for label in y_text]
    y = torch.tensor(y)

    return x_text, y_text, y, class_dict


def load_w2v(file_path):
    """ load w2v model
        input: model file name
        output: w2v model
    """
    w2v = KeyedVectors.load_word2vec_format(file_path, binary=False)
    return w2v


def tokenizer_w2v(query, w2v):
    """
    input a text query and output its token index
    :param query: text query
    :param w2v: a gensim word2vec object
    :return: torch.longtensor[query length] tokenized tensor
    """
    query_token_id = []
    query = query.split(" ")
    size_vocab, size_emb = w2v.syn0.shape
    for w in query:
        if w in w2v.vocab:
            query_token_id.append(w2v.vocab[w].index)
        else:
            weight = np.random.uniform(low=-0.5, high=0.5, size=(size_emb,))
            w2v.add(w, weight.astype(np.float32), replace=False)
            query_token_id.append(w2v.vocab[w].index)

    token_id = torch.tensor(query_token_id)

    return token_id


def tokenize_w2v(text, w2v):
    """
    tokenize input text, add random initialize emb to w2v vocab if word not in w2v
    :param text: [str], list of text query, split by space
    :param w2v: a gensim word2vec object
    :return: pad_token_id[n, max_len], n_vocab, d_emb
    """
    n_vocab, d_emb = w2v.syn0.shape
    token_id = [tokenizer_w2v(query, w2v).long() for query in text]
    pad_token_id = pad_sequence(token_id, batch_first=True, padding_value=0)

    return pad_token_id, n_vocab, d_emb


def tokenize_transformers(text, key_pretrained):
    """
    reference: https://huggingface.co/transformers/pretrained_models.html
    :param text: [str], list of text query, split by space
    :param key_pretrained: reference@Shortcut_name
    :return: pad_token_id[n, max_len], n_vocab, d_emb
    """
    import transformers
    from config import get_info_transfomer
    key_architecture = get_info_transfomer(key_pretrained)['architecture']
    d_emb = get_info_transfomer(key_pretrained)

    print("____ tokenize", key_architecture, key_pretrained, " ____")
    key_tokenizer = key_architecture + "Tokenizer"
    tokenizer = getattr(transformers, key_tokenizer).from_pretrained(key_pretrained)

    text_processed = ['[CLS]' + query + '[SEP]' for query in text]
    token = [tokenizer.tokenize(query) for query in text_processed]
    token_id = [tokenizer.convert_tokens_to_ids(query) for query in token]
    n_vocab = len(set(tool.flatten(token_id)))
    token_id = [torch.tensor(query).long() for query in token_id]
    pad_token_id = pad_sequence(token_id, batch_first=True, padding_value=0)

    return pad_token_id, n_vocab, d_emb


def process_label(intents, w2v, class_id_startpoint=0):
    """ pre process class labels
        input: class label list, w2v model
        output: class dict and label vectors
    """
    class_dict = {}
    label_vec = []
    class_id = class_id_startpoint
    for line in intents:
        # check whether all the words in w2v dict
        label = line.split(' ')
        for w in label:
            if w not in w2v.vocab:
                print('not in w2v dict', w)

        # compute label vec
        label_sum = np.sum([w2v[w] for w in label], axis=0)
        label_vec.append(label_sum)
        # store class names => index
        class_dict[' '.join(label)] = class_id
        class_id = class_id + 1
    return class_dict, np.asarray(label_vec)


def read_datasets(data_setting):
    print("------------------read datasets begin-------------------")

    data = dict()
    data['dataset'] = data_setting['dataset']

    # load data
    data_path = data_setting['data_prefix'] + data_setting['dataset_name']
    x_text, y_text, y, class_dict = load_data(data_path)

    # tokenize
    data['text_represent'] = data_setting['text_represent']
    data['key_pretrained'] = data_setting['key_pretrained']

    word2vec_path = data_setting['data_prefix'] + data_setting['wordvec_name']
    w2v = load_w2v(word2vec_path)
    if data['text_represent'] == 'w2v':
        x_pad, _, _ = tokenize_w2v(x_text, w2v)
        y_pad, data['n_vocab'], data['d_emb'] = tokenize_w2v(y_text, w2v)
        data['embedding'] = torch.from_numpy(tool.norm_matrix(w2v.syn0))
    elif data['text_represent'] == 'transformers':
        x_pad, data['n_vocab'], data['d_emb'] = tokenize_transformers(x_text, data['key_pretrained'])
        # TODO n_vocab not update by y
        y_pad, _, _ = tokenize_transformers(y_text, data['key_pretrained'])

    # split dataset
    print('split dataset')
    if data_setting['freeze_class']:
        label_order = data_setting['label_order']
        unseen_class = data_setting['unseen_class']
        seen_class = [x for x in label_order if x not in unseen_class]
    else:
        if data_setting['dataset'] == 'SMP18':
            del class_dict['聊天']
        class_freq_dict = {k: (y == class_dict[k]).nonzero().shape[0] for k in class_dict.keys()}
        class_freq_dict = {k: v for k, v in class_freq_dict.items() if v > 2}
        label_order = list(class_freq_dict.keys())
        label_freq_np = np.array([class_freq_dict[l] for l in label_order])
        label_freq_np = label_freq_np/sum(label_freq_np)
        n_c_tr = math.ceil(len(label_order) * data_setting['seen_class_prob'])
        seen_class = list(np.random.choice(label_order, size=n_c_tr, replace=False, p=label_freq_np))
        unseen_class = [x for x in label_order if x not in seen_class]
        print("unseen_class:\n", unseen_class)

    # update class_dict and y (first seen and then unseen classes in class_dict)
    class_dict = dict()
    for c in seen_class:
        class_dict[c] = len(class_dict)
    for c in unseen_class:
        class_dict[c] = len(class_dict)
    y = [class_dict[label] if label in class_dict.keys() else -1 for label in y_text]
    y = torch.tensor(y)

    # get split index
    if data_setting['freeze_class']:
        data['id_split'] = data_setting['id_split']
        matlab_data = sio.loadmat(data_setting['data_prefix'] + data_setting['sim_name_withS'])

        idx_tr = (matlab_data['train_ind'][0, data_setting['id_split']]-1).tolist()[0]
        idx_te = (matlab_data['test_ind'][0, data_setting['id_split']]-1).tolist()[0]
        idx_tr = torch.LongTensor(idx_tr)
        idx_te = torch.LongTensor(idx_te)

        data['sim'] = matlab_data['similarity'][0, data_setting['id_split']]
        data['sim'] = torch.from_numpy(data['sim'])
    else:
        idx_tr = torch.tensor([]).long()
        idx_te = torch.tensor([]).long()
        for c in unseen_class:
            idx_c = (y == class_dict[c]).nonzero().squeeze(-1)
            if data_setting['test_mode'] == 'standard':
                idx_te = torch.cat([idx_te, idx_c], dim=0)
            else:
                # idx_te = torch.cat([idx_te, idx_c], dim=0)
                n_unseen_in_test = int(idx_c.shape[0] * data_setting['sample_in_test_prob'])
                idx_te = torch.cat([idx_te, idx_c[:n_unseen_in_test]], dim=0)
        for c in seen_class:
            idx_c = (y == class_dict[c]).nonzero().squeeze(-1)
            if data_setting['test_mode'] == 'standard':
                idx_tr = torch.cat([idx_tr, idx_c], dim=0)
            elif data_setting['test_mode'] == 'general':
                idx_perm = torch.randperm(idx_c.shape[0])
                idx_c = idx_c[idx_perm]
                n_seen_in_test = int(idx_c.shape[0] * data_setting['sample_in_test_prob'])
                idx_tr = torch.cat([idx_tr, idx_c[n_seen_in_test:]])
                idx_te = torch.cat([idx_te, idx_c[:n_seen_in_test]])

    # shuffle data
    idx_tr = idx_tr[torch.randperm(idx_tr.shape[0])]
    idx_te = idx_te[torch.randperm(idx_te.shape[0])]

    # get padded class represent
    if data['text_represent'] == 'w2v':
        # class_pad, data['n_vocab'], data['d_emb'] = tokenize_w2v(seen_class + unseen_class, w2v)
        class_pad, data['n_vocab'], data['d_emb'] = tokenize_w2v(seen_class, w2v)
    else:
        class_pad, _, _ = tokenize_transformers(seen_class + unseen_class, data['key_pretrained'])

    # pre process seen and unseen labels
    # if data_setting['text_represent'] == 'w2v':
    class_id_startpoint = 0
    sc_dict, sc_vec = process_label(seen_class, w2v, class_id_startpoint)
    if data_setting['test_mode'] == 'general':
        uc_dict, uc_vec = process_label(unseen_class, w2v, class_id_startpoint+len(sc_dict))
        uc_dict = dict(sc_dict, **uc_dict)
        uc_vec = np.concatenate([sc_vec, uc_vec], axis=0)
    else:
        uc_dict, uc_vec = process_label(unseen_class, w2v, class_id_startpoint)
    data['sc_dict'] = sc_dict
    data['sc_vec'] = sc_vec
    data['uc_dict'] = uc_dict
    data['uc_vec'] = uc_vec

    # finalize data package
    data['n_tr'] = idx_tr.shape[0]
    data['x_tr'] = x_pad[idx_tr]
    data['y_tr'] = y[idx_tr]
    data['y_ind'] = torch.zeros(data['n_tr'], len(seen_class)).scatter_(1, data['y_tr'].unsqueeze(1), 1)

    data['n_te'] = idx_te.shape[0]
    data['x_te'] = x_pad[idx_te]
    data['y_te'] = y[idx_te]

    data['seen_class'] = seen_class
    data['unseen_class'] = unseen_class
    data['class_padded'] = class_pad

    print("------------------read dataset end---------------------")
    return data
