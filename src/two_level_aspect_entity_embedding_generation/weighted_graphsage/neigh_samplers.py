from __future__ import division
from __future__ import print_function

from graphsage.layers import Layer

import tensorflow as tf
flags = tf.app.flags
FLAGS = flags.FLAGS


"""
Classes that are used to sample node neighborhoods
"""

class UniformNeighborSampler(Layer):
    """
    Uniformly samples neighbors.
    Assumes that adj lists are padded with random re-sampling
    """
    def __init__(self, adj_info, prob_matrix, **kwargs):
        super(UniformNeighborSampler, self).__init__(**kwargs)
        self.adj_info = adj_info
        self.prob_matrix = prob_matrix

    def _call(self, inputs):
        ids, num_samples, num = inputs
        adj_lists = tf.nn.embedding_lookup(self.adj_info, ids)
        adj_weights = tf.nn.embedding_lookup(self.prob_matrix, ids)
        gather = tf.gather(adj_weights, adj_lists, axis=1)
        indices = tf.reshape(tf.range(tf.shape(ids)[0]), (-1, 1))

        # Use tf.shape() to make this work with dynamic shapes.
        batch_size = tf.shape(gather)[0]
        rows_per_batch = tf.shape(gather)[1]
        indices_per_batch = tf.shape(indices)[1]

        # Offset to add to each row in indices. We use `tf.expand_dims()` to make 
        # this broadcast appropriately.
        offset = tf.expand_dims(tf.range(0, batch_size) * rows_per_batch, 1)

        # Convert indices and logits into appropriate form for `tf.gather()`. 
        flattened_indices = tf.reshape(indices + offset, [-1])
        flattened_logits = tf.reshape(gather, tf.concat([[-1], tf.shape(gather)[2:]], axis=0))

        selected_rows = tf.gather(flattened_logits, flattened_indices)
        #selected_rows = tf.random.shuffle(selected_rows)
        _, top_k = tf.math.top_k(selected_rows, k=num_samples)
        sample_val = tf.gather(adj_lists, top_k, axis=1)
        indices = tf.reshape(tf.range(tf.shape(ids)[0]), (-1, 1))

        # Use tf.shape() to make this work with dynamic shapes.
        batch_size = tf.shape(sample_val)[0]
        rows_per_batch = tf.shape(sample_val)[1]
        indices_per_batch = tf.shape(top_k)[1]

        # Offset to add to each row in indices. We use `tf.expand_dims()` to make 
        # this broadcast appropriately.
        offset = tf.expand_dims(tf.range(0, batch_size) * rows_per_batch, 1)

        # Convert indices and logits into appropriate form for `tf.gather()`. 
        flattened_indices = tf.reshape(indices + offset, [-1])
        flattened_logits = tf.reshape(sample_val, tf.concat([[-1], tf.shape(sample_val)[2:]], axis=0))

        selected_rows = tf.gather(flattened_logits, flattened_indices)
        return selected_rows
