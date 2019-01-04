import nltk
import numpy as np
import torch
from .InferSent.models import InferSent

def infersent_init(opt):
    print("initializing InferSent model...", end=" ", flush=True)
    infersent = InferSent(opt.InferSent_params_model).to(opt.device)
    infersent.load_state_dict(torch.load(opt.InferSent_MODEL_PATH))
    print("done")

    print("building vocabulary...")
    infersent.set_w2v_path(opt.InferSent_W2V_PATH)
    infersent.build_vocab_k_words(K=100000)
    return infersent

# wrapper function
def sent_embedding(model, sentences, tokenize = True, verbose = False, bsize = 64) :
    if verbose :
        print('now encoding sentences...')
    return model.encode(sentences, tokenize=tokenize, verbose=verbose, bsize=bsize)

if __name__ == '__main__' :
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("pyTorch will be running on device: "+str(device))

    V = 2
    MODEL_PATH = './InferSent/encoder/infersent%s.pkl' % V
    params_model = {'bsize': 128, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
                    'pool_type': 'max', 'dpout_model': 0.0, 'version': V}

    print("initializing InferSent model...", end=" ", flush=True)
    infersent = InferSent(params_model).to(device)
    infersent.load_state_dict(torch.load(MODEL_PATH))
    print("done")

    print("building vocabulary...")
    W2V_PATH = './InferSent/dataset/fastText/crawl-300d-2M.vec'
    infersent.set_w2v_path(W2V_PATH)
    infersent.build_vocab_k_words(K=100000)

    sentences = []
    with open('InferSent/encoder/samples.txt') as f:
        for line in f:
            sentences.append(line.strip())
    print("num of sentences : "+str(len(sentences)))
    print('now encoding sentences...')
    embeddings = infersent.encode(sentences, tokenize=True, verbose=True, bsize=128)
    infersent.visualize('A man plays an instrument.', tokenize=True)
