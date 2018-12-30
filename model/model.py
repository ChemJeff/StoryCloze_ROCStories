# coding=utf-8
'''
pyTorch version implementation of some model
and classifier for classification tasks
code for python 3.x and pyTorch == 0.4
'''
import argparse
import pickle

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torch.autograd import Variable

class WordEmbedding(nn.Module):
    '''
    所使用的词向量，载入预训练的词向量(fixed or fine-tune)
    输入 torch.nn.utils.rnn.packedSequence:按batch打包好的句子的word_id
    输出 torch.nn.utils.rnn.packedSequence:按batch打包好的句子的word_embedding
    '''
    def __init__(self, opt):
        super(WordEmbedding, self).__init__()
        self.opt = opt
        self.device = self.opt.device
        with open(self.opt.word_embedding, 'rb') as f:
            self.word_embeds = nn.Embedding.from_pretrained(
                torch.Tensor(pickle.load(f)),
              freeze=self.opt.word_embedding_fixed)
    
    def forward(self, sentence):
        embeds = self.word_embeds(sentence)
        return embeds

class BiLSTM(nn.Module):
    '''
    用于特征提取部分的BiLSTM网络
    输入 torch.nn.utils.rnn.packedSequence:按batch打包好的词embedding
        或者 torch.tensor:(batch=1, seq_len=4, embedding_dim)的句子embedding序列
    输出 torch.tensor:(batch=1, seq_len, hidden_dim*num_layer)最后一个隐状态作为序列的embedding
    '''
    def __init__(self, opt, embedding_dim, hidden_dim, batch_size=1):
        super(BiLSTM, self).__init__()
        self.opt = opt
        self.device = self.opt.device
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim   # 同时也是输出维度
        self.batch_size = batch_size # 硬编码
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first=True)
        self.hidden = self.init_hidden() # 固定初始输入

    def init_hidden(self):
        # 用全0的向量作为(h_0, c_0)初始输入
        return (torch.zeros(2, self.batch_size, self.hidden_dim // 2, requires_grad=False, device=self.device),
                torch.zeros(2, self.batch_size, self.hidden_dim // 2, requires_grad=False, device=self.device))

    def forward(self, data_in):
        # 整个模型中统一使用batch_first的顺序
        _, (h_t, c_t) = self.lstm(data_in, self.hidden)
        # 只需要(h_t, c_t)，因此直接覆盖
        return h_t.view(1, -1, self.hidden_dim)

class MLP(nn.Module):
    """
    FC-ReLU-FC-ReLU-FC-ReLU-FC-Softmax
    多层前馈神经网络，用于分类，判断两个候选句子哪个更有可能是句子正确结尾
    (由于数据集的处理，两句中有且只有一句是正确的)
    """
    def __init__(self, opt, input_size, hidden_sizes, output_size=2):
        super(MLP, self).__init__()
        self.opt = opt
        self.device = self.opt.device
        self.hidden_sizes = hidden_sizes
        self.input_size = input_size
        self.output_size = output_size

        self.layer1 = nn.Linear(self.input_size, self.hidden_sizes[0])
        self.layer2 = nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1])
        self.layer3 = nn.Linear(self.hidden_sizes[1], self.hidden_sizes[2])
        self.output_layer = nn.Linear(self.hidden_sizes[-1], self.output_size)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=0)
        self.logsoftmax = nn.LogSoftmax(dim=0)

        nn.init.xavier_uniform_(self.layer1.weight.data)
        nn.init.xavier_uniform_(self.layer2.weight.data)
        nn.init.xavier_uniform_(self.layer3.weight.data)
        nn.init.xavier_uniform_(self.output_layer.weight.data)

    def _get_score(self, input):
        # 给定输出，输出一个得分形式(未正规化)
        output = self.relu(self.layer1(input))
        output = self.relu(self.layer2(output))
        output = self.relu(self.layer3(output))
        output = self.output_layer(output)
        output = output.view(-1)    # 拉直为一维向量，每个分量对应一个预测的得分
        return output
    
    def nll_loss(self, input, label):
        # 计算分类的交叉熵loss
        score = self._get_score(input)
        log_prob = self.logsoftmax(score)
        label = torch.Tensor(label).to(self.device)
        loss = -torch.sum(log_prob.mul(label))  # 手动计算，注意符号(有负号)
        return loss

    def forward(self, input):
        # 用于预测，分别给定两个句子可能为正确结尾的概率
        # 有且只有一句是正确的结尾
        score = self._get_score(input)
        return self.softmax(score)
