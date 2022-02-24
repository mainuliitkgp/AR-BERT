from __future__ import division
from __future__ import print_function

import os
import time
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import random
import pickle

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS

tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
#core params..
flags.DEFINE_float('learning_rate', 0.00001, 'initial learning rate.')
flags.DEFINE_string('train_file_path', '', 'name of the object file that stores the training data. must be specified.')
flags.DEFINE_string('output_dir', '', 'path of the output directory.')
flags.DEFINE_string('ds_name', '', 'name of the dataset.')

# left to default values in main experiments
flags.DEFINE_float('weight_decay', 0.01, 'weight for l2 loss on probe.') 
flags.DEFINE_integer('probe_hidden_size', 100, 'probe weight shape.')
flags.DEFINE_integer('epochs', 100, 'number of epochs to train.')
flags.DEFINE_integer('batch_size', 128, 'minibatch size.')

flags.DEFINE_integer('gpu', 0, "which gpu to use.")

os.environ["CUDA_VISIBLE_DEVICES"]=str(FLAGS.gpu)

GPU_MEM_FRACTION = 0.8

def load_data(fp):
    data_examples = np.load(fp)
    data_examples_with_indices_tuple = []
    for i in range(data_examples.shape[0]):
        data_examples_with_indices_tuple.append((i, data_examples[i]))
    return data_examples_with_indices_tuple

def minibatch_end(batch_num, batch_size, train_features):
  return batch_num * batch_size >= len(train_features)

def next_minibatch_feed_dict(batch_num, batch_size, train_features):
    start_idx = batch_num * batch_size
    end_idx = min(start_idx + batch_size, len(train_features))
    batch_features = train_features[start_idx : end_idx]
    return batch_feed_dict(batch_features)

def batch_feed_dict(batch_features):
    embedding_array, indices_list = [], []
    for i in range(len(batch_features)):
        embedding_array.append(batch_features[i][1])
        indices_list.append(batch_features[i][0])
    embedding_array = np.array(embedding_array, dtype = np.float32)
    embedding_i = embedding_array[:, 0:768]
    embedding_j = embedding_array[:, 768:1536]
    embedding_k = embedding_array[:, 1536:2304]
    return embedding_i, embedding_j, embedding_k, indices_list

def train(train_data):
    # Define placeholders
    placeholders = {'h_i' : tf.placeholder(tf.float32, shape=(None, 768), name='bert_embedding_of_entity_i'),
        'h_j' : tf.placeholder(tf.float32, shape=(None, 768), name='bert_embedding_of_entity_j'),
        'h_k' : tf.placeholder(tf.float32, shape=(None, 768), name='bert_embedding_of_entity_k'), }

    B = tf.get_variable(
        "probe_weights", [768, FLAGS.probe_hidden_size],
        initializer=tf.truncated_normal_initializer(stddev=0.02),
        trainable= True)

    # build model
    diff_i_j_in_bert_space = tf.subtract(placeholders['h_i'], placeholders['h_j']) # i and j are closer. h_i-h_j (bs*768)
    diff_i_k_in_bert_space = tf.subtract(placeholders['h_i'], placeholders['h_k']) # i and k are far away. h_i-h_k (bs*768)

    transformed_diff_i_j_in_bert_space = tf.matmul(diff_i_j_in_bert_space, B) # B(h_i-h_j) (bs*100)
    transformed_diff_i_k_in_bert_space = tf.matmul(diff_i_k_in_bert_space, B) # B(h_i-h_k) (bs*100)
 
    # shape: (bs*100)
    dist_i_j_in_B_space = tf.multiply(transformed_diff_i_j_in_bert_space, transformed_diff_i_j_in_bert_space) # (B(h_i-h_j))^T(B(h_i-h_j))
    dist_i_k_in_B_space = tf.multiply(transformed_diff_i_k_in_bert_space, transformed_diff_i_k_in_bert_space) # (B(h_i-h_j))^T(B(h_i-h_j))

    per_example_dist_i_j_in_B_space = tf.reduce_sum(dist_i_j_in_B_space, axis = -1) #(bs)
    per_example_dist_i_k_in_B_space = tf.reduce_sum(dist_i_k_in_B_space, axis = -1) #(bs)

    per_example_dist_diff = per_example_dist_i_k_in_B_space - per_example_dist_i_j_in_B_space # (bs)

    dist_diff = tf.reduce_mean(per_example_dist_diff)
   
    loss = -dist_diff + FLAGS.weight_decay*tf.nn.l2_loss(B)

    optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

    train_op = optimizer.minimize(loss)

    # gpu configuration
    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = GPU_MEM_FRACTION
    config.allow_soft_placement = True

    # Initialize session
    sess = tf.Session(config=config)
    merged = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(FLAGS.output_dir, sess.graph)

    # Init variables
    sess.run(tf.global_variables_initializer())

    # Train model

    train_epoch_loss_list, train_epoch_dist_diff_list = [], []

    for epoch in range(FLAGS.epochs):
        #np.random.shuffle(train_data)
        random.shuffle(train_data)

        iter_num = 0
        train_epoch_loss = 0.0
        train_epoch_dist_diff = 0.0
        train_epoch_per_example_dist_diff = []
        train_epoch_indices_list = []
        print('Epoch: %04d' % (epoch + 1))
        while not minibatch_end(iter_num, FLAGS.batch_size, train_data):
            # Construct feed dictionary
            embedding_i, embedding_j, embedding_k, batch_indices_list = next_minibatch_feed_dict(iter_num, FLAGS.batch_size, train_data)
            batch_feed_dict = {placeholders['h_i']:embedding_i, placeholders['h_j']:embedding_j, placeholders['h_k']:embedding_k}

            outs = sess.run([train_op, per_example_dist_diff, dist_diff, loss], feed_dict = batch_feed_dict)

            train_epoch_per_example_dist_diff.extend(outs[1])
            train_epoch_dist_diff += outs[2]
            train_epoch_loss += outs[3]
            train_epoch_indices_list.extend(batch_indices_list)

            iter_num += 1

        train_epoch_dist_diff = train_epoch_dist_diff/iter_num
        train_epoch_loss = train_epoch_loss/iter_num
        
        print('train dist diff: '+str(train_epoch_dist_diff)+' train loss: '+str(train_epoch_loss))
        train_epoch_loss_list.append(train_epoch_loss)
        train_epoch_dist_diff_list.append(train_epoch_dist_diff)

    plt.plot(list(range(1, FLAGS.epochs+1)), train_epoch_loss_list)
    plt.xlabel('epochs')
    plt.ylabel('training loss')
    plt.savefig(FLAGS.output_dir+FLAGS.ds_name+'_structutal_probing_training_loss.png')
    plt.clf()

    plt.plot(list(range(1, FLAGS.epochs+1)), train_epoch_dist_diff_list)
    plt.xlabel('epochs')
    plt.ylabel('avg. disatnce difference')
    plt.savefig(FLAGS.output_dir+FLAGS.ds_name+'_structutal_probing_training_avg_dist_diff.png')
    plt.clf()

    train_epoch_per_example_dist_diff_with_indices_tuple = []
    for idx in range(len(train_epoch_per_example_dist_diff)):
        train_epoch_per_example_dist_diff_with_indices_tuple.append((train_epoch_indices_list[idx], train_epoch_per_example_dist_diff[idx])) 
    train_epoch_per_example_dist_diff.sort(reverse=True)
    train_epoch_per_example_dist_diff_with_indices_tuple.sort(reverse = True, key=lambda x: x[1])

    sorted_train_epoch_indices_list, sorted_train_epoch_per_example_dist_diff = [], []
    for idx_count in range(len(train_epoch_per_example_dist_diff_with_indices_tuple)):
        sorted_train_epoch_indices_list.append(train_epoch_per_example_dist_diff_with_indices_tuple[idx_count][0])
        sorted_train_epoch_per_example_dist_diff.append(train_epoch_per_example_dist_diff_with_indices_tuple[idx_count][1])

    plt.plot(list(range(1, len(sorted_train_epoch_per_example_dist_diff) + 1)), sorted_train_epoch_per_example_dist_diff)
    plt.xlabel('train example indices')
    plt.ylabel('sorted avg. dist. difference')
    plt.savefig(FLAGS.output_dir+FLAGS.ds_name+'_structural_probing_sorted_dist_diff.png')

    with open(FLAGS.output_dir+FLAGS.ds_name+'_structural_probing_sorted_dist_diff_with_indices.pkl', 'wb') as fp:
        pickle.dump(train_epoch_per_example_dist_diff_with_indices_tuple, fp)


def main(argv=None):
    print("Loading training data..")
    train_data = load_data(FLAGS.train_file_path)
    print("Done loading training data..")
    train(train_data)

if __name__ == '__main__':
    tf.app.run()
