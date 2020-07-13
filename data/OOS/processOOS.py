# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 22:30:14 2019

@author: floatsd
"""
import numpy as np
from gensim.models.keyedvectors import KeyedVectors
import re

flatten = lambda l: [item for sublist in l for item in sublist]

def restrictW2V(w2v, restricted_word_set):
    
    print("restrict w2v")
    new_vectors = []
    new_vocab = {}
    new_index2entity = []
    
    for i in range(len(w2v.vocab)):
        word = w2v.index2entity[i]
        vec = w2v.vectors[i]
        vocab = w2v.vocab[word]
        if word in restricted_word_set:
            vocab.index = len(new_index2entity)
            new_index2entity.append(word)
            new_vocab[word] = vocab
            new_vectors.append(vec)

    w2v.vocab = new_vocab
    w2v.vectors = np.array(new_vectors)
    w2v.index2entity = np.array(new_index2entity)
    w2v.index2word = np.array(new_index2entity)

    return w2v.vocab, w2v.vectors

def load_w2v(file_name):
    print("loading", file_name)
    w2v = KeyedVectors.load_word2vec_format(file_name, binary=False)
    return w2v

if __name__ == '__main__':
    
    file_path = 'dataOOS.txt'
    w2v_path = 'F:/1_DATABASE/NLPGeneral/wordVector/vec/crawl-300d-2M-subword.vec'
    outp_w2v_path = "w2v_oos_subword" + ".vec"
    
    labels = []
    queries = []
    
        
    p_left = re.compile(r' (?![&])[\W]')
    p_right = re.compile(r'(?![&])[\W] ')
    p_insert = re.compile(r'(\w)([\:\+\-\/\%])(\w)')
    p_x = re.compile(r'(\d)(\\x)(\d)')
    p_s = re.compile(r'([’\'])(s\b)')    
    p_end_punc = re.compile(r'\W$')
    p_sub = [(r'[’\']ll', ' will'), (r'[’\']d', ' would'), (r'[’\']ve', ' have'),
             (r'n[’\']t', ' not'), (r'i[’\']m', 'i am'), (r'[’\']re', ' are')]
    for line in open(file_path,'r', encoding='utf-8'):
        arr =str(line.strip())
        arr = arr.split('\t')
        labels.append([w for w in arr[0].split(' ')])
        query = arr[1]
        query = re.sub(p_left, r'\g<0> ', query)
        query = re.sub(p_right, r' \g<0>', query)
        query = re.sub(p_insert, '\g<1> \g<2> \g<3>', query)
        query = re.sub(p_insert, '\g<1> \g<2> \g<3>', query)
        query = re.sub(p_x, '\g<1> \g<2> \g<3>', query)
        query = re.sub(p_s, ' \g<1> \g<2>', query)
        query = re.sub(p_end_punc, ' \g<0>', query)
        for p in p_sub:
            query = re.sub(p[0], p[1], query)
        query = re.sub(' +', ' ', query)
        queries.append([w for w in query.split(' ')])
    
    print("statistics vocabulary")
    vocab_labels = list(set(flatten(labels)))
    vocab_queries = list(set(flatten(queries)))
    vocab_all = list(set(vocab_labels + vocab_queries))
    
    file_outp = "dataOOS_1024.txt"
    print("writing",file_outp)
    with open(file_outp,'w',encoding='utf-8') as f:
        for i,q in enumerate(queries):
            f.write(" ".join(labels[i]) + "\t" + " ".join(q) + '\n')
    #%%
    w2v = load_w2v(w2v_path)
    vocab_w2v = list(w2v.vocab)
    vocab_not_in = [w for w in vocab_all if w not in vocab_w2v]
    print(len(vocab_not_in))
    word_class, vec_class = restrictW2V(w2v,vocab_all)
    print("saving",outp_w2v_path)
    w2v.save_word2vec_format(outp_w2v_path)
    
    #%%
    set_labels = list(set([" ".join(l) for l in labels]))
    file_outp = "vocab_intent_oos.txt"
    print("writing",file_outp)
    with open(file_outp,'w',encoding='utf-8') as f:
        for l in set_labels:
            f.write(l + '\n')
    
    