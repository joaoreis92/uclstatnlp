#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 13:34:54 2017

@author: joao
"""

#%%! SETUP 1 - DO NOT CHANGE, MOVE NOR COPY
import sys, os
_snlp_book_dir = "/home/joao/stat-nlp-book/"
sys.path.append(_snlp_book_dir)
# docker image contains tensorflow0.10.0rc0. We will support execution of only that version!
import statnlpbook.nn as nn

import tensorflow as tf
import numpy as np

#%%! SETUP 2 - DO NOT CHANGE, MOVE NOR COPY
data_path = _snlp_book_dir + "data/nn/"
data_train = nn.load_corpus(data_path + "train.tsv")
data_dev = nn.load_corpus(data_path + "dev.tsv")
assert(len(data_train) == 45502)

#%%
data_train[0]

#%%
# convert train set to integer IDs
train_stories, train_orders, vocab = nn.pipeline(data_train)


#%%
# get the length of the longest sentence
max_sent_len = train_stories.shape[2]

# convert dev set to integer IDs, based on the train vocabulary and max_sent_len
dev_stories, dev_orders, _ = nn.pipeline(data_dev, vocab=vocab, max_sent_len_=max_sent_len)


#%%
nn.show_data_instance(dev_stories, dev_orders, vocab, 155)


#%%
### MODEL PARAMETERS ###
target_size = 5
vocab_size = len(vocab)
input_size = 10
# n = len(train_stories)
output_size = 5
n_hidden = 100
lambda_l2 = 0.001
state_size = 100


#%%
### MODEL ###
tf.reset_default_graph()
## PLACEHOLDERS
story = tf.placeholder(tf.int64, [None, None, None], "story")        # [batch_size x 5 x max_length]
order = tf.placeholder(tf.int64, [None, None], "order")              # [batch_size x 5]

batch_size = tf.shape(story)[0]

sentences = [tf.reshape(x, [batch_size, -1]) for x in tf.split(1, 5, story)]  # 5 times [batch_size x max_length]
### WEIGHTS AND BIASES ######

weights = {

    'out': tf.get_variable(name='whout',shape=[state_size, 5 * target_size],initializer=tf.contrib.layers.xavier_initializer())
}

biases ={

    'out': tf.get_variable(name='bout',shape=[1,5 * target_size],initializer=tf.contrib.layers.xavier_initializer())
}


# Word embeddings
initializer = tf.random_uniform_initializer(-0.1, 0.1)
embeddings = tf.get_variable("W", [vocab_size, input_size], initializer=initializer)


sentences_embedded = [tf.nn.embedding_lookup(embeddings, sentence)   #5 times [batch_size x max_seq_length x input_size]
                      for sentence in sentences]

hs = [tf.reduce_sum(sentence, 1) for sentence in sentences_embedded] # 5 times [batch_size x input_size]

h = tf.pack(hs)
h = tf.transpose(hs,[1,0,2])

#h = tf.concat(1, hs)    # [batch_size x 5*input_size]
#h = tf.reshape(h, [batch_size, 5*input_size])



###### Layers ######
cell = tf.nn.rnn_cell.LSTMCell(state_size,state_is_tuple=True)
output,_ =  tf.nn.dynamic_rnn(cell, h, dtype=tf.float32,sequence_length=tf.tile([5],[tf.to_int32(tf.shape(h)[0])])) 
#output = tf.transpose(output,[1,0,2])
outputs = tf.unpack(output,axis=1)
output = outputs[4]
#output = tf.gather(output,[4])

output = tf.reshape(output,[-1,state_size])


logits_flat = tf.add(tf.matmul(output,weights['out']),biases['out'])   # [batch_size x 5*target_size]
logits = tf.reshape(logits_flat, [-1, 5, target_size])        # [batch_size x 5 x target_size]

# loss 
loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits, order) +
    lambda_l2 * tf.nn.l2_loss(embeddings)+
    lambda_l2 * tf.nn.l2_loss(weights['out'])+
    lambda_l2 * tf.nn.l2_loss(biases['out']))
                    

# prediction function
unpacked_logits = [tensor for tensor in tf.unpack(logits, axis=1)]
softmaxes = [tf.nn.softmax(tensor) for tensor in unpacked_logits]
softmaxed_logits = tf.pack(softmaxes, axis=1)
predict = tf.arg_max(softmaxed_logits, 2)


#%%
opt_op = tf.train.AdamOptimizer(0.001).minimize(loss)

#%%
BATCH_SIZE = 32
from random import shuffle
import time
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    n = train_stories.shape[0]

    for epoch in range(10):
        print('----- Epoch', epoch, '-----')
        total_loss = 0
        t = time.time()
        for i in range(n // BATCH_SIZE):
            if i%100 == 0:
                train_stories_shuf = []
                train_orders_shuf = []
                index_shuf = list(range(len(train_stories)))
                shuffle(index_shuf)
                for j in index_shuf:
                    train_stories_shuf.append(train_stories[j])
                    train_orders_shuf.append(train_orders[j])
            
            inst_story = train_stories_shuf[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
            inst_order = train_orders_shuf[i * BATCH_SIZE: (i + 1) * BATCH_SIZE]
            feed_dict = {story: inst_story, order: inst_order}
            _, current_loss = sess.run([opt_op, loss], feed_dict=feed_dict)

            total_loss += current_loss

        print(' Train loss:', total_loss / n)

        train_feed_dict = {story: train_stories, order: train_orders}
        train_predicted = sess.run(predict, feed_dict=train_feed_dict)
        train_accuracy = nn.calculate_accuracy(train_orders, train_predicted)
        print(' Train accuracy:', train_accuracy)
        
        dev_feed_dict = {story: dev_stories, order: dev_orders}
        dev_predicted = sess.run(predict, feed_dict=dev_feed_dict)
        dev_accuracy = nn.calculate_accuracy(dev_orders, dev_predicted)
        print(' Dev accuracy:', dev_accuracy)
        print(time.time()-t)
        
    
    nn.save_model(sess)