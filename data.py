# coding=utf-8
from utils.dataPreprocess import *

if __name__ == "__main__":
    data_path = "./data/"

    vocab = build_vocab_list([data_path + 'train.csv', data_path + 'test.csv', data_path + 'val.csv'])
    with open(data_path + 'vocab_freq.pkl', 'wb') as f:
        pickle.dump(vocab, file=f)
    w2i, i2w, i2v = build_word_vec(vocab, w2v_path='./embedding/InferSent/dataset/fastText/crawl-300d-2M.vec')
    with open(data_path + 'vocab.pkl', 'wb') as f:
        pickle.dump((w2i, i2w), file=f)
    with open(data_path + 'vocab_embed.pkl', 'wb') as f:
        pickle.dump(i2v, file=f)
    train_corpus = corpus_tokenize(data_path + 'train.csv', 'train')
    with open(data_path + 'train.pkl', 'wb') as f:
        pickle.dump(train_corpus, file=f)
    test_corpus = corpus_tokenize(data_path + 'test.csv', 'test')
    with open(data_path + 'test.pkl', 'wb') as f:
        pickle.dump(test_corpus, file=f)
    val_corpus = corpus_tokenize(data_path + 'val.csv', 'val')
    with open(data_path + 'val.pkl', 'wb') as f:
        pickle.dump(val_corpus, file=f)