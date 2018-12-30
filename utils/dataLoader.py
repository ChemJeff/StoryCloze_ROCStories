# coding=utf-8

'''
Load preprocessed data given in .pkl file and convert it
to the format used in pyTorch network 
code for python 3.x and pyTorch == 0.4
'''

import os
import argparse
import csv
import numpy as np
import random
import torch
import torch.utils.data as data
import pickle
import multiprocessing


class DataSet(data.Dataset) :
    '''
    自定义DataSet类型，需要实现__init__,__getitem__,__len__方法
    '''

    def __init__(self, opt) :
        self.opt = opt
        # self.split = self.opt.split     # in {'train', 'test', 'val'}
        with open(self.opt.vocab, 'rb') as f :
            self.word2id, self.id2word = pickle.load(f)
        print("Vocab from \'%s\' loaded" % (self.opt.vocab))
        self.vocab = self.word2id.keys()
        self.vocab_size = len(self.vocab)
        print("Vocab size: %d\n" % (self.vocab_size))       
        self.dataset = self.corpus_covert(self.opt.corpus)
        self.datalen = len(self.dataset)
        print("Dataset from \'%s\' loaded" % (self.opt.corpus))
        print("Dataset split: %s" %(self.split))
        print("Dataset size: %s\n" %(self.datalen))


    def corpus_covert(self, corpus_path):
        corpus = []
        with open(corpus_path, 'rb') as f:
            corpus_raw = pickle.load(f)
            self.split = corpus_raw['split']
            for idx, sents in enumerate(corpus_raw['data']):
                sent_ids = []
                for sent in sents:
                    ids = [self.word2id[self.opt.BOS]] + \
                        [self.word2id[word] \
                        if word in self.vocab \
                        else self.word2id[self.opt.UNK]
                        for word in sent] \
                        + [self.word2id[self.opt.EOS]]
                    sent_ids.append(ids)
                if self.split == 'val':     #这个数据集包含label的信息
                    label = corpus_raw['label'][idx]
                    corpus.append([sent_ids, label])
                else:
                    corpus.append(sent_ids)
        return corpus

    def __getitem__(self, index) :
        '''
        根据数据集划分的不同，处理也有所不同，
        如果是train，那么需要随机抽取一个错误的结尾，
        并且随机放在1号句子或者2号句子的位置，
        然后生成对应的label。
        如果是test，那么没有对应的label返回
        如果是val，直接返回6个句子以及对应label
        '''
        if self.split == 'val':
            sent_ids, label = self.dataset[index]
            return (sent_ids, label)
        elif self.split == 'test':
            sent_ids = self.dataset[index]
            return (sent_ids, )     # 额外包一层tuple方便后续统一处理
        else:
            assert self.split == 'train'
            sent_ids = self.dataset[index][:4]
            rand_idx = index
            while(rand_idx == index):
                rand_idx = random.randint(0, self.datalen - 1)
            prob_front = random.random()    
            # 将正确答案平均分布在两个位置上，避免模型直接学习答案分布
            if prob_front <= 0.5:
                sent_ids.append(self.dataset[index][4])
                sent_ids.append(self.dataset[rand_idx][4])
                label = [1, 0]
            else:
                sent_ids.append(self.dataset[rand_idx][4])
                sent_ids.append(self.dataset[index][4])
                label = [0, 1]
            return (sent_ids, label)

    def __len__(self) :
        return self.datalen


def packBatch(batch_in_list) :
    '''
    return a torch.nn.utils.rnn.PackedSequence and indices to restore 
    the original order for a batch sorted by length in a decreasing order
    '''
    item_count = len(batch_in_list[0])
    split = 'train' if item_count == 2 else 'test'
    batch_len = [x for x in map(lambda item:len(item[0]), batch_in_list)]
    # print(batch_len)
    _, idx_sort = torch.sort(torch.tensor(batch_len), dim=0, descending=True)
    _, idx_unsort = torch.sort(idx_sort, dim=0)
    word_batch_in_list = [batch_in_list[i][0] for i in idx_sort]
    # batch_in_list.sort(key=lambda x:len(x), reverse=True)
    # print(word_batch_in_list)
    packed_words = torch.nn.utils.rnn.pack_sequence(word_batch_in_list)
    # print(packed_words)
    if split == 'train' :
        tag_batch_in_list = [batch_in_list[i][1] for i in idx_sort]
        packed_tags = torch.nn.utils.rnn.pack_sequence(tag_batch_in_list)
        return (packed_words, packed_tags), idx_unsort
    return packed_words, idx_unsort

def collate(batch_in_list) :
    '''
    collate function to convert raw data fetched from dataset
    to a torch.nn.utils.rnn.PackedSequence for LSTM input
    considering usage, return a lot of auxiliary info as well
    '''
    batch_size = len(batch_in_list)
    words = [item[1] for item in batch_in_list]
    for i in range(batch_size) :
        batch_in_list[i] = [torch.tensor(item, dtype=torch.int64) for item in batch_in_list[i][0]]
    packed, idx_unsort = packBatch(batch_in_list)
    return packed, idx_unsort, words

def testcollate(data_in):
    return data_in[0]

if __name__ == "__main__" :
    opt = argparse.Namespace()
    opt.BOS = '<p>'
    opt.EOS = '</p>'
    opt.UNK = '<unk>'
    opt.corpus = './train.pkl'
    opt.vocab = './vocab.pkl'
    opt.word_embedding = './vocab_embed.pkl' 
    opt.encoding = 'utf-8'
    dataset = DataSet(opt)
    dataloader = data.DataLoader(dataset, batch_size=1, collate_fn=testcollate)
    sample = 10
    for item in dataloader:
        sample -= 1
        if sample < 0 :
            continue
        print(item, sep='\n')
    pass