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

import embedding.sent_embedding as sent_embedding
from model.model import *
from utils.utils import *
import utils.dataLoader as dataLoader

class BiLSTM_MLP(nn.Module):
    '''
    将模块拼接起来
    '''
    def __init__(self, opt):
        super(BiLSTM_MLP, self).__init__()
        self.opt = opt
        self.device = self.opt.device
        self.infersent = sent_embedding.infersent_init(self.opt)
        self.sent_embed = sent_embedding.sent_embedding
        self.story_embed_lstm = BiLSTM(self.opt, self.opt.sent_embed_dim, \
            self.opt.story_embed_dim)
        self.classifier = MLP(self.opt, self.opt.story_embed_dim + self.opt.sent_embed_dim*2, \
             self.opt.fc_hidden_dims, self.opt.num_classes)

    def _get_context(self, sent_list):
        '''
        输入list:包含六个句子对应id的list
        输出torch.Tensor:(4096*3)上下文向量，包含故事和候选句子的信息，用于通过分类器
        '''
        sent_embed_list = self.sent_embed(self.infersent, sent_list, tokenize=False, verbose=False, bsize=6)

        story_sent_embed = [torch.tensor(sent_embed) \
                            for sent_embed in sent_embed_list[:4]]  # 前四句是确定的故事内的句子
        story_sent_embed = torch.stack(story_sent_embed, dim=0)
        story_sent_embed = story_sent_embed.view(1, 4, -1).to(self.device)
        candidate1 = torch.tensor(sent_embed_list[4]).view(-1).to(self.device)
        candidate2 = torch.tensor(sent_embed_list[5]).view(-1).to(self.device)
        story_embed = self.story_embed_lstm(story_sent_embed).view(-1)
        context = torch.cat((story_embed, candidate1, candidate2), 0).to(self.device)
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

    ckpt_path = "./checkpoint/"
    data_path = "./data/"
    log_dir = "./log/test/"

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\nDevice: %s\n" % (device))

    opt = argparse.Namespace()
    opt.device = device
    opt.BOS = '<p>'
    opt.EOS = '</p>'
    opt.UNK = '<unk>'
    opt.InferSent_V = 2
    opt.InferSent_MODEL_PATH = './embedding/InferSent/encoder/infersent%d.pkl' % opt.InferSent_V
    opt.InferSent_params_model = {'bsize': 128, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                'pool_type': 'max', 'dpout_model': 0.0, 'version': opt.InferSent_V}
    opt.InferSent_W2V_PATH = './embedding/InferSent/dataset/fastText/crawl-300d-2M.vec'
    opt.rawdata = True # 不用将word转换为对应id
    opt.corpus = data_path + 'train.pkl'
    opt.vocab = data_path + 'vocab.pkl'
    opt.word_embedding = data_path + './vocab_embed.pkl' 
    opt.word_embedding_fixed = True
    opt.encoding = 'utf-8'
    opt.word_embed_dim = 300
    opt.sent_embed_dim = 4096
    opt.story_embed_dim = 4096
    opt.fc_hidden_dims = [2048, 256, 32]  # 三个全连接层的输出维度
    opt.num_classes = 2
    opt.lr = 1e-2
    opt.weight_decay = 1e-4
    opt.iter_cnt = 95000       # if non-zero, load checkpoint at iter (#iter_cnt)
    opt.train_epoch = 100
    opt.save_every = 10000
    opt.sample = 100    # 输出的样本故事个数
    opt.answer = True  # 是否有正确答案用于计算准确率

    with open(opt.vocab, 'rb') as f:
        word_to_ix, ix_to_word = pickle.load(f)

    testdataset = dataLoader.DataSet(opt)
    testdataloader = torch.utils.data.DataLoader(
        testdataset, collate_fn=dataLoader.testcollate)

    print("All necessites prepared, time used: %f s\n" % (time.time() - stime))
    model = BiLSTM_MLP(opt).to(device)
    assert opt.iter_cnt > 0

    if opt.iter_cnt > 0:
        try:
            print("Load checkpoint at %s" %(ckpt_path + "Infersent-bilstm-mlp_300w_4096s_iter%d.cpkt" % (opt.iter_cnt)))
            # load parameters from checkpoint given
            model.load_state_dict(torch.load(ckpt_path + "Infersent-bilstm-mlp_300w_4096s_iter%d.cpkt" % (opt.iter_cnt)))
            print("Success\n")
        except Exception as e:
            print("Failed, check the path and permission of the checkpoint")
            exit(0)

    # Test prediction from model in the checkpoint
    with torch.no_grad():
        right_cnt = 0       # 预测正确的样本数
        total_cnt = 0       # 所有的样本数量
        sample_cnt = opt.sample
        if opt.answer:
            for sent_ids, label in testdataloader:
                label_pred = model(sent_ids)
                judge = (label_pred[0]*label[0] + label_pred[1]*label[1] > 0.5)
                total_cnt += 1
                if judge is True:
                    right_cnt += 1
                if sample_cnt > 0 :
                    visualize(sent_ids, label_pred, ix_to_word, label=label, rawdata=opt.rawdata)
                    sample_cnt -= 1
            print("\n***************Summary*****************")
            print("Precision: %f (%d/%d)" % (right_cnt/total_cnt, right_cnt, total_cnt))
        else:
            # 同时将预测结果保存至文件
            pred_out = open(log_dir + "prediction.txt", 'w', encoding='utf-8')
            for (sent_ids, ) in testdataloader:
                label_pred = model(sent_ids)
                pred_out.write("1\n" if label_pred[0]> label_pred[1] else "2\n")
                if sample_cnt > 0 :
                    visualize(sent_ids, label_pred, ix_to_word, rawdata=opt.rawdata)
                    sample_cnt -= 1
