# StoryCloze_ROCStories

pyTorch implementation of story cloze task on ROCStories Corpora

### 运行环境

python==3.x

torch==0.4

nltk (with 'punkt' installed using ```nltk.download('punkt')```)

pickle

tensorflow (optional, for tensorboard logging)

### 使用方法

训练数据、验证数据与测试数据均在./data目录下

首先执行data.py预处理数据，生成vocab.freq.pkl、vocab.pkl、vocab_embed.pkl、train.pkl、test.pkl和val.pkl

使用InferSent的准备：

​	下载fastText词向量并解压至./embedding/InferSent/dataset/fastText目录下

```bash
mkdir embedding/InferSent/dataset/fastText
curl -Lo embedding/InferSent/dataset/fastText/crawl-300d-2M.vec.zip https://s3-us-west-1.amazonaws.com/fasttext-vectors/crawl-300d-2M.vec.zip
unzip embedding/InferSent/dataset/fastText/crawl-300d-2M.vec.zip -d embedding/InferSent/dataset/fastText/
```

​	下载预训练的InferSent（V2）模型至./embedding/InferSent/encoder目录下

```bash
curl -Lo embedding/InferSent/encoder/infersent2.pkl https://s3.amazonaws.com/senteval/infersent/infersent2.pkl
```


然后就可以直接执行train.py进行模型的训练

或者使用test.py加载预训练的模型进行故事结尾的预测