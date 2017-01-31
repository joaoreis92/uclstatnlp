#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 17:49:06 2017

@author: joao
"""

import sys, os
# docker image contains tensorflow 0.10.0rc0. We will support execution of only that version!
import statnlpbook.nn as nn

import tensorflow as tf
import numpy as np

#%%
data_path = "data/nn/"
data_train = nn.load_corpus(data_path + "train.tsv")
data_dev = nn.load_corpus(data_path + "dev.tsv")
assert(len(data_train) == 45502)

#%%
data_train[0]

#%%
train_stories, train_orders, vocab = nn.pipeline(data_train)

#%%
max_sent_len = train_stories.shape[2]
dev_stories, dev_orders, _ = nn.pipeline(data_dev, vocab=vocab, max_sent_len_=max_sent_len)

#%%
### MODEL PARAMETERS ###
target_size = 5
vocab_size = len(vocab)
input_size = 10
# n = len(train_stories)
output_size = 5

#%%
### MODEL ###

## PLACEHOLDERS
story = tf.placeholder(tf.int64, [None, None, None], "story")        # [batch_size x 5 x max_length]
order = tf.placeholder(tf.int64, [None, None], "order")              # [batch_size x 5]

batch_size = tf.shape(story)[0]

sentences = [tf.reshape(x, [batch_size, -1]) for x in tf.split(1, 5, story)]  # 5 times [batch_size x max_length]

# Word embeddings
initializer = tf.random_uniform_initializer(-0.1, 0.1)
embeddings = tf.get_variable("W", [vocab_size, input_size], initializer=initializer)

sentences_embedded = [tf.nn.embedding_lookup(embeddings, sentence)   # [batch_size x max_seq_length x input_size]
                      for sentence in sentences]

hs = [tf.reduce_sum(sentence, 1) for sentence in sentences_embedded] # 5 times [batch_size x input_size]

h = tf.concat(1, hs)    # [batch_size x 5*input_size]
h = tf.reshape(h, [batch_size, 5*input_size])

logits_flat = tf.contrib.layers.linear(h, 5 * target_size)    # [batch_size x 5*target_size]
logits = tf.reshape(logits_flat, [-1, 5, target_size])        # [batch_size x 5 x target_size]

# loss 
loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, order))

# prediction function
unpacked_logits = [tensor for tensor in tf.unpack(logits, axis=1)]
softmaxes = [tf.nn.softmax(tensor) for tensor in unpacked_logits]
softmaxed_logits = tf.pack(softmaxes, axis=1)
predict = tf.arg_max(softmaxed_logits, 2)