# coding=utf-8

'''
Load data given in .csv file , do some preprocessing 
and then save to .pkl files
code for python 3.x
'''

import csv
import pickle
import nltk
import numpy as np

BOS = '<p>'
EOS = '</p>'
UNK = '<unk>'
BOS_ID = 0
EOS_ID = 1
UNK_ID = 2
EMBED_DIM = 300

def add_freq(word, freq_dict):
    if word not in freq_dict.keys():
        freq_dict[word] = 1
    else:
        freq_dict[word] += 1

def build_vocab_list(file_path_list=['train.csv', 'val.csv', 'test.csv'], encoding='utf-8', out_file=None):
    vocab_freq = dict()
    for file_path in file_path_list:
        with open(file_path, 'r', encoding=encoding) as f:
            csvreader = csv.reader(f)
            next(csvreader)     # 跳过csv文件的标题行
            for entry in csvreader:
                entry = entry[:-1] if file_path == 'val.csv' else entry
                for sent in entry:
                    words = nltk.tokenize.word_tokenize(sent)
                    add_freq(BOS, vocab_freq)
                    add_freq(EOS, vocab_freq)
                    for word in words:
                        add_freq(word, vocab_freq)

    return vocab_freq

def build_word_vec(vocab_freq, w2v_path='../embedding/InferSent/dataset/fastText/crawl-300d-2M.vec'):
    word_vec = dict()
    w2i = dict()
    i2w = dict()
    i2v = []
    idx = UNK_ID + 1    # 当前分配的id号
    with open(w2v_path, 'r', encoding='utf-8') as f:
        line = f.readline().strip()
        for line in f:
            word, vec = line.split(' ', 1)
            if word in vocab_freq.keys():
                word_vec[word] = np.fromstring(vec, sep=' ')
        word_vec[UNK] = np.zeros(EMBED_DIM)  # 用全0向量作为OOV词汇的词向量
    print('Found %s(/%s) words with w2v vectors' % (len(word_vec), len(vocab_freq)))
    sorted_freq = sorted(vocab_freq.items(), key=lambda x:x[1], reverse=True) # 按频率排序
    w2i[BOS] = BOS_ID
    w2i[EOS] = EOS_ID
    w2i[UNK] = UNK_ID
    for (word, freq) in sorted_freq:  # 按出现频率分配词的id号，建立w2i索引
        if word in {BOS, EOS, UNK}:  # 已经分配了index的特殊标记
            continue
        if word in word_vec.keys():
            w2i[word] = idx
            idx += 1
    for i in w2i.items():   # 反向建立一个i2w索引
        i2w[i[1]] = i[0]
    for i in range(idx):    # 建立id号到词向量的矩阵
        i2v.append(word_vec[i2w[i]])
    i2v = np.array(i2v)
    print('Vocab size: %d, including {<p>, </p>, <unk>}' % (idx))
    return w2i, i2w, i2v

def corpus_tokenize(corpus_path, split, encoding='utf-8'):
    '''
    提前将分词完成的训练数据保存，避免每次读入数据都要现场分词
    '''
    assert split in {'train', 'test', 'val'}
    corpus = dict()
    corpus['data'] = []
    corpus['split'] = split
    if split == 'val':
        corpus['label'] = []
    with open(corpus_path, 'r', encoding=encoding) as f:
        csvreader = csv.reader(f)
        next(csvreader)         # 跳过csv文件的标题行
        for entry_raw in csvreader:
            entry = entry_raw[:-1] if split == 'val' else entry_raw
            sents = []
            for sent in entry:
                words = nltk.tokenize.word_tokenize(sent)
                sents.append(words)
            if split == 'val':     # 需要从csv文件中读入label
                label = [1, 0] if entry_raw[-1] == '1' else [0, 1]
                corpus['data'].append(sents)
                corpus['label'].append(label)
            else:
                corpus['data'].append(sents)
    return corpus

if __name__ == "__main__":
    # vocab = build_vocab_list()
    # with open('vocab_freq.pkl', 'wb') as f:
    #     pickle.dump(vocab, file=f)
    # with open('vocab_freq.pkl', 'rb') as f:
    #     vocab = pickle.load(f)
    # w2i, i2w, i2v = build_word_vec(vocab)
    # with open('vocab.pkl', 'wb') as f:
    #     pickle.dump((w2i, i2w), file=f)
    # with open('vocab_embed.pkl', 'wb') as f:
    #     pickle.dump(i2v, file=f)
    # with open('vocab.pkl', 'rb') as f:
    #     w2i, i2w = pickle.load(f)
    # with open('vocab_embed.pkl', 'rb') as f:
    #     i2v = pickle.load(f)
    # print(w2i, i2w, i2v)
    # corpus = corpus_tokenize('./train.csv', 'train')
    # with open('train.pkl', 'wb') as f:
    #     pickle.dump(corpus, file=f)
    # print(corpus)
    pass