# Sentiment Analysis on Reddit Data using BERT (Summer 2019)

We are interested in understanding user opinions about Activision titles on social media data. In this project, we aim to predict sentiment on Reddit data. 

__Method__

* Since there are no labels on the Reddit data, we look into _transfer learning_ techniques by first training on other related datasets then transfer to Reddit.

* We use the original Google [BERT](https://github.com/google-research/bert) model for training. This repo is a fork of the [Multi-GPU BERT](https://github.com/lambdal/bert) (see below)

__Dataset__

* [Metacritic Game Review](http://www.metacritic.com/game/): provided in the data folder as example

* [Amazon review data](http://jmcauley.ucsd.edu/data/amazon/)

* Reddit Dataset

* [IMDB](http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz)

* Data in the format of 

| Review text | Sentiment |
| ------------|:---------:|
| ...         | ...       |  


## To train

For example, fine-tune the BERT model using Metacritic data:

```shell
./run_meta.sh
```

## To generate prediction results using a trained model

For example, predict sentiment using a trained model

```shell
./run_predict.sh
```

## To pre-train

For example, run pre-training using reddit data

```shell
./run_pretrain.sh
```

## Results and some analysis

See **BERT_analysis.ipynb**

View my final presentation slides [here](https://drive.google.com/file/d/15tiV3BNXfZa0ftkCiNnT4jiehEK_TRPr/view?usp=sharing)

---

# Multi-GPU Ready BERT

This is a fork of the original (Google's) BERT implementation. 

* Add Multi-GPU support with Horovod

This [blog](https://lambdalabs.com/blog/bert-multi-gpu-implementation-using-tensorflow-and-horovod-with-code/) explains all the changes we made to the original implementation.

__Install__
Please first [install Horovod](https://github.com/uber/horovod#install)

__Run__
See the commands in each section to run BERT with Multi-GPUs:

* [Sentence (and sentence-pair) classification tasks](#sentencepair)

* [SQuAD 1.1](#squad1.1)

* [SQuAD 2.0](#squad2.0)

* [Pre-training](#pretraining)


# BERT

**\*\*\*\*\* New November 23rd, 2018: Un-normalized multilingual model + Thai +
Mongolian \*\*\*\*\***

We uploaded a new multilingual model which does *not* perform any normalization
on the input (no lower casing, accent stripping, or Unicode normalization), and
additionally inclues Thai and Mongolian.

**It is recommended to use this version for developing multilingual models,
especially on languages with non-Latin alphabets.**

This does not require any code changes, and can be downloaded here:

*   **[`BERT-Base, Multilingual Cased`](https://storage.googleapis.com/bert_models/2018_11_23/multi_cased_L-12_H-768_A-12.zip)**:
    104 languages, 12-layer, 768-hidden, 12-heads, 110M parameters

**\*\*\*\*\* New November 15th, 2018: SOTA SQuAD 2.0 System \*\*\*\*\***

We released code changes to reproduce our 83% F1 SQuAD 2.0 system, which is
currently 1st place on the leaderboard by 3%. See the SQuAD 2.0 section of the
README for details.

**\*\*\*\*\* New November 5th, 2018: Third-party PyTorch and Chainer versions of
BERT available \*\*\*\*\***

NLP researchers from HuggingFace made a
[PyTorch version of BERT available](https://github.com/huggingface/pytorch-pretrained-BERT)
which is compatible with our pre-trained checkpoints and is able to reproduce
our results. Sosuke Kobayashi also made a
[Chainer version of BERT available](https://github.com/soskek/bert-chainer)
(Thanks!) We were not involved in the creation or maintenance of the PyTorch
implementation so please direct any questions towards the authors of that
repository.

**\*\*\*\*\* New November 3rd, 2018: Multilingual and Chinese models available
\*\*\*\*\***

We have made two new BERT models available:

*   **[`BERT-Base, Multilingual`](https://storage.googleapis.com/bert_models/2018_11_03/multilingual_L-12_H-768_A-12.zip)
    (Not recommended, use `Multilingual Cased` instead)**: 102 languages,
    12-layer, 768-hidden, 12-heads, 110M parameters
*   **[`BERT-Base, Chinese`](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip)**:
    Chinese Simplified and Traditional, 12-layer, 768-hidden, 12-heads, 110M
    parameters

We use character-based tokenization for Chinese, and WordPiece tokenization for
all other languages. Both models should work out-of-the-box without any code
changes. We did update the implementation of `BasicTokenizer` in
`tokenization.py` to support Chinese character tokenization, so please update if
you forked it. However, we did not change the tokenization API.

For more, see the
[Multilingual README](https://github.com/google-research/bert/blob/master/multilingual.md).

**\*\*\*\*\* End new information \*\*\*\*\***

## Introduction

**BERT**, or **B**idirectional **E**ncoder **R**epresentations from
**T**ransformers, is a new method of pre-training language representations which
obtains state-of-the-art results on a wide array of Natural Language Processing
(NLP) tasks.

Our academic paper which describes BERT in detail and provides full results on a
number of tasks can be found here:
[https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805).

To give a few numbers, here are the results on the
[SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/) question answering
task:

SQuAD v1.1 Leaderboard (Oct 8th 2018) | Test EM  | Test F1
------------------------------------- | :------: | :------:
1st Place Ensemble - BERT             | **87.4** | **93.2**
2nd Place Ensemble - nlnet            | 86.0     | 91.7
1st Place Single Model - BERT         | **85.1** | **91.8**
2nd Place Single Model - nlnet        | 83.5     | 90.1

And several natural language inference tasks:

System                  | MultiNLI | Question NLI | SWAG
----------------------- | :------: | :----------: | :------:
BERT                    | **86.7** | **91.1**     | **86.3**
OpenAI GPT (Prev. SOTA) | 82.2     | 88.1         | 75.0

Plus many other tasks.

Moreover, these results were all obtained with almost no task-specific neural
network architecture design.

If you already know what BERT is and you just want to get started, you can
[download the pre-trained models](#pre-trained-models) and
[run a state-of-the-art fine-tuning](#fine-tuning-with-bert) in only a few
minutes.

## What has been released in this repository?

We are releasing the following:

*   TensorFlow code for the BERT model architecture (which is mostly a standard
    [Transformer](https://arxiv.org/abs/1706.03762) architecture).
*   Pre-trained checkpoints for both the lowercase and cased version of
    `BERT-Base` and `BERT-Large` from the paper.
*   TensorFlow code for push-button replication of the most important
    fine-tuning experiments from the paper, including SQuAD, MultiNLI, and MRPC.

All of the code in this repository works out-of-the-box with CPU, GPU, and Cloud
TPU.
