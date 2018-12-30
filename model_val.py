# coding=utf-8
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

import argparse
import datetime
import pickle
import time
import os
import sys

from model.model import *
from utils.utils import *
import utils.dataLoader as dataLoader

try:
    import tensorflow as tf
except ImportError:
    print("Tensorflow not installed; No tensorboard logging.")
    tf = None

class BiLSTM_MLP(nn.Module):
    '''
    将模块拼接起来
    '''
    def __init__(self, opt):
        super(BiLSTM_MLP, self).__init__()
        self.opt = opt
        self.device = self.opt.device
        self.word_embed = WordEmbedding(self.opt)
        self.sent_embed_lstm = BiLSTM(self.opt, self.opt.word_embed_dim, \
            self.opt.sent_embed_dim, batch_size=6)
        self.story_embed_lstm = BiLSTM(self.opt, self.opt.sent_embed_dim, \
            self.opt.story_embed_dim)
        self.classifier = MLP(self.opt, self.opt.story_embed_dim + self.opt.sent_embed_dim*2, \
             self.opt.fc_hidden_dims, self.opt.num_classes)

    def _get_context(self, sent_list):
        '''
        输入list:包含六个句子对应id的list
        输出torch.Tensor:(1024)上下文向量，包含故事和候选句子的信息，用于通过分类器
        '''
        word_embed_list = [self.word_embed(torch.tensor(item, dtype=torch.long).to(self.device)) \
            for item in sent_list]
        packed_word_embed, idx_unsort = packBatch(word_embed_list, self.opt)
        sent_embed_list = self.sent_embed_lstm(packed_word_embed)
        story_sent_embed = [sent_embed_list[0][idx_unsort[i]] \
            for i in range(4)]  # 前四句是确定的故事内的句子
        story_sent_embed = torch.stack(story_sent_embed, dim=0)
        story_sent_embed = story_sent_embed.view(1, 4, -1)
        candidate1 = sent_embed_list[0][idx_unsort[4]].view(-1)
        candidate2 = sent_embed_list[0][idx_unsort[5]].view(-1)
        story_embed = self.story_embed_lstm(story_sent_embed).view(-1)
        context = torch.cat((story_embed, candidate1, candidate2), 0)
        return context

    def nll_loss(self, sent_list, label):
        context = self._get_context(sent_list)
        loss = self.classifier.nll_loss(context, label)
        return loss

    def forward(self, sent_list):
        '''
        输入list:包含六个句子对应id的list
        输出list:对应两个candidate的正确概率预测，加和为1
        '''
        context = self._get_context(sent_list)
        label_pred = self.classifier(context).tolist()
        return label_pred


if __name__ == "__main__":

    ckpt_path = "./base_ckpt/"
    data_path = "./data/"
    log_dir = "./log/"

    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    time_suffix = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    log_dir = log_dir + "run_%s/" % (time_suffix)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    stime = time.time()
    sys.stdout = Logger(log_dir + "log")
    tf_summary_writer = tf and tf.summary.FileWriter(log_dir + "tflog")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nDevice: %s\n" % (device))

    opt = argparse.Namespace()
    opt.device = device
    opt.BOS = '<p>'
    opt.EOS = '</p>'
    opt.UNK = '<unk>'
    opt.corpus = data_path + 'val.pkl'
    opt.vocab = data_path + 'vocab.pkl'
    opt.word_embedding = data_path + './vocab_embed.pkl' 
    opt.word_embedding_fixed = True
    opt.encoding = 'utf-8'
    opt.word_embed_dim = 300
    opt.sent_embed_dim = 256
    opt.story_embed_dim = 512
    opt.fc_hidden_dims = [256, 64, 16]  # 三个全连接层的输出维度
    opt.num_classes = 2
    opt.lr = 1e-2
    opt.weight_decay = 1e-4
    opt.iter_cnt = 0       # if non-zero, load checkpoint at iter (#iter_cnt)
    opt.train_epoch = 100
    opt.save_every = 10000

    with open(opt.vocab, 'rb') as f:
        word_to_ix, ix_to_word = pickle.load(f)

    dataset = dataLoader.DataSet(opt)
    dataloader = torch.utils.data.DataLoader(
        dataset, collate_fn=dataLoader.testcollate, shuffle=True)

    testdataset = dataLoader.DataSet(opt)
    testdataloader = torch.utils.data.DataLoader(
        testdataset, collate_fn=dataLoader.testcollate)

    print("All necessites prepared, time used: %f s\n" % (time.time() - stime))
    model = BiLSTM_MLP(opt).to(device)

    from_ckpt = False
    if opt.iter_cnt > 0:
        try:
            print("Load checkpoint at %s" %(ckpt_path + "bilstm-mlp_300w_256s_1024d_iter%d.cpkt" % (opt.iter_cnt)))
            # load parameters from checkpoint given
            model.load_state_dict(torch.load(ckpt_path + "bilstm-mlp_300w_256s_1024d_iter%d.cpkt" % (opt.iter_cnt)))
            print("Success\n")
            from_ckpt = True
        except Exception as e:
            print("Failed, check the path and permission of the checkpoint")
            exit(0)

    optimizer = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=opt.lr, weight_decay=opt.weight_decay)

    # Check predictions before training
    with torch.no_grad():
        for sent_ids, label in testdataloader:
            label_pred = model(sent_ids)
            visualize(sent_ids, label_pred, ix_to_word, label)
            break

    iter_cnt = opt.iter_cnt if from_ckpt is True else 0
    for epoch in range(opt.train_epoch):
        stime = time.time()
        loss_add = 0.0
        for sent_ids, label in dataloader:
            model.zero_grad()

            loss = model.nll_loss(sent_ids, label)
            loss_add += loss.item()

            loss.backward()
            optimizer.step()
            iter_cnt += 1
            if iter_cnt % 100 == 0 :
                print("%s  last 100 iters: %d s, iter = %d, average loss = %f" %(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S, %Z'), time.time() - stime, iter_cnt, loss_add/100))
                loss_add = 0.0
                sys.stdout.flush()
                add_summary_value(tf_summary_writer, "train_loss", loss.item(), iter_cnt)
                add_summary_value(tf_summary_writer, 'learning_rate', get_lr(optimizer), iter_cnt)
                tf_summary_writer.flush()
                stime = time.time()
            if iter_cnt % opt.save_every == 0 :
                try:
                    torch.save(model.state_dict(), ckpt_path + "bilstm-mlp_300w_256s_1024d_iter%d.cpkt" % (iter_cnt))
                    print("checkpoint saved at \'%s\'" % (ckpt_path + "bilstm-mlp_300w_256s_1024d_iter%d.cpkt" % (iter_cnt)))
                except Exception as e:
                    print(e)
                with torch.no_grad():
                    for sent_ids, label in testdataloader:
                        label_pred = model(sent_ids)
                        visualize(sent_ids, label_pred, ix_to_word, label)
                        break

        if (iter_cnt % 100 > 0):    # divide-by-zero bug fixed
            print("%s  last %d iters: %d s, average loss = %f" % (datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S, %Z'), iter_cnt % 100, time.time() - stime, loss_add/(iter_cnt % 100)))
            # loss_add = 0.0  fix bug when compute average loss
            with torch.no_grad():
                for sent_ids, label in testdataloader:
                    label_pred = model(sent_ids)
                    visualize(sent_ids, label_pred, ix_to_word, label)
                    break
