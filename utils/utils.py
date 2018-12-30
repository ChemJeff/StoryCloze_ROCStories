# coding=utf-8

'''
Some simple utility functions used for Story Cloze task
for python 3.x
'''
import sys
import torch
try:
    import tensorflow as tf
except ImportError:
    tf = None

def add_summary_value(writer, key, value, iteration):
    if writer is None:
        return
    summary = tf.Summary(value=[tf.Summary.Value(tag=key, simple_value=value)])
    writer.add_summary(summary, iteration)

def get_lr(optimizer):
    for group in optimizer.param_groups:
        return group['lr']

def packBatch(batch_in_list, opt=None) :
    '''
    return a torch.nn.utils.rnn.PackedSequence and indices to restore 
    the original order for a batch sorted by length in a decreasing order
    '''
    device = opt.device if opt is not None else 'cpu'
    batch_len = [x for x in map(lambda item:len(item), batch_in_list)]
    _, idx_sort = torch.sort(torch.tensor(batch_len).to(device), dim=0, descending=True)
    _, idx_unsort = torch.sort(idx_sort, dim=0)
    sorted_batch = [batch_in_list[i].to(device) for i in idx_sort]
    packed_batch = torch.nn.utils.rnn.pack_sequence(sorted_batch)
    return packed_batch, idx_unsort

class Logger(object):
    def __init__(self, file_name):
        self.terminal = sys.stdout
        self.log = open(file_name, 'a')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    def flush(self):
        self.terminal.flush()
        self.log.flush()

def visualize(sent_ids, label_pred, idx2word, label=None):
    '''
    打印原始的故事文本(部分不在vocab的词会被替换为<unk>)
    以及候选句子和对应的预测概率、正确答案(如果有)
    '''
    print("Story:\n    ", end="")
    for sent in sent_ids[:4]:
        sent = sent[1:-1]   # 去掉<p></p>标记
        print(" ".join([idx2word[idx] for idx in sent]), end=" ")
    print("\nCandidate 1:\n    ",end="")
    print(" ".join([idx2word[idx] for idx in sent_ids[4][1:-1]]))
    print("\nCandidate 2:\n    ",end="")
    print(" ".join([idx2word[idx] for idx in sent_ids[5][1:-1]]))
    print("\nPrediction: ", end="")
    print(list(label_pred))
    if label is not None:
        print("\nAnswer: ", end="")
        print(label, "(Prediction Right)" if \
            (label_pred[0]*label[0] + label_pred[1]*label[1]) > 0.5 # 超过0.5即认为预测正确(弱标准)
            else "(Prediction Wrong)")




