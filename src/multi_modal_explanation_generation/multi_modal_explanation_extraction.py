# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import modeling_modified
import optimization
import tokenization
import tensorflow as tf
import tokenization
import numpy as np
import random
from sklearn.metrics import matthews_corrcoef, f1_score
import xml.etree.ElementTree as ET
import pickle 

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)
random.seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS

tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
#core params..
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "dataset_name", None,
    "The input dataset name.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_float("learning_rate", 3e-5, "The initial learning rate for Adam.")  

flags.DEFINE_integer('n_concept', 7, 'number of concepts.')

flags.DEFINE_integer('n_concept_word', 7, 'number of concepts for word importance.')

flags.DEFINE_integer('batch_size', 128, 'minibatch size.')

flags.DEFINE_integer('epochs', 100, 'number of epochs.')

flags.DEFINE_integer('gpu', 0, "which gpu to use.")

os.environ["CUDA_VISIBLE_DEVICES"]=str(FLAGS.gpu)

GPU_MEM_FRACTION = 0.8


stop_word_list = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", 
                  "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", 
                  "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", 
                  "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", 
                  "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", 
                  "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", 
                  "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", 
                  "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", 
                  "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor",  
                  "only", "own", "same", "so", "than", "very", "s", "t", "can", "will", "just", "don", "should", "now",
                  "one", "it's", "br", "<PAD>", "<START>", "<UNK>", "would", "could", "also", "may", "many", "go", "another",
                  "want", "two", "actually", "every", "thing", "know", "made", "get", "something", "back", "though"]

class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.

  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.

  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()

class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.is_real_example = is_real_example


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None, entity_index = None):
    """Constructs a InputExample.

    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label
    self.entity_index = entity_index



def parse_SemEval14(fn):
    polar_idx={'positive': 0, 'negative': 1, 'neutral': 2}
    sent_list = []
    asp_list = []
    label_list = []
    root=ET.parse(fn).getroot()
    corpus=[]
    opin_cnt=[0]*len(polar_idx)
    for sent in root.iter("sentence"):
        opins=set()
        for opin in sent.iter('aspectTerm'):
            if int(opin.attrib['from'] )!=int(opin.attrib['to'] ) and opin.attrib['term']!="NULL":
                if opin.attrib['polarity'] in polar_idx:
                    sent_list.append(sent.find('text').text)
                    asp_list.append(opin.attrib['term'])
                    label_list.append(opin.attrib['polarity'])

    return sent_list, asp_list, label_list

class SemEval2014AtscProcessor():
    """Processor for the Aspect-target sentiment Task of Semeval 2014 Task 4 Subtask 2"""

    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "train.xml"), os.path.join(data_dir, FLAGS.dataset_name+"_unique_entities.txt"), os.path.join(data_dir, FLAGS.dataset_name+"_train_entities.txt"), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "dev.xml"), os.path.join(data_dir, FLAGS.dataset_name+"_unique_entities.txt"), os.path.join(data_dir, FLAGS.dataset_name+"_test_entities.txt"), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
            os.path.join(data_dir, "test.xml"), os.path.join(data_dir, FLAGS.dataset_name+"_unique_entities.txt"), os.path.join(data_dir, FLAGS.dataset_name+"_test_entities.txt"), "test")

    def get_labels(self):
        """See base class."""
        return ["positive", "negative", "neutral"]

    def _create_examples(self, corpus, unique_entity_corpus, example_wise_entity_corpus, set_type):
        """Creates examples for the training, dev and test sets."""

        sentences, aspects, labels = parse_SemEval14(corpus)
        unique_entity_list, example_wise_entity_list = [], []
        with open(unique_entity_corpus, encoding='utf-8') as fp:
            for line in fp:
                unique_entity_list.append(line.strip())
        with open(example_wise_entity_corpus, encoding='utf-8') as fp:
            for line in fp:
                example_wise_entity_list.append(line.strip())

        examples = []

        for i in range(len(sentences)):

            guid = "%s-%s" % (set_type, i)
            try:
                text_a = sentences[i]
                text_b = aspects[i]
                label = labels[i]
                entity_index = int(unique_entity_list.index(example_wise_entity_list[i]))
            except IndexError:
                continue
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label, entity_index=entity_index))
        return examples



def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""

  if isinstance(example, PaddingInputExample):
    return InputFeatures(
        input_ids=[0] * max_seq_length,
        input_mask=[0] * max_seq_length,
        segment_ids=[0] * max_seq_length,
        label_id=0,
        is_real_example=False)

  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i

  tokens_a = tokenizer.tokenize(example.text_a)
  tokens_b = None
  if example.text_b:
    tokens_b = tokenizer.tokenize(example.text_b)

  if tokens_b:
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  else:
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]

  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)

  if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  label_id = label_map[example.label]
  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in tokens]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_id=label_id,
      is_real_example=True)
  entity_index = example.entity_index
  return feature, entity_index



# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
  """Convert a set of `InputExample`s to a list of `InputFeatures`."""

  features = []
  
  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature, entity_index = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)

    features.append((feature, entity_index))
    
  return features
    

def construct_placeholders(f_train, f_train_word):
    # Define placeholders
    placeholders = {}
    placeholders['f_input'] = tf.placeholder(tf.float32, shape = (None, f_train.shape[1],f_train.shape[2]), name="f_input")
    placeholders['f_input_word'] = tf.placeholder(tf.float32, shape = (None, f_train_word.shape[1],f_train_word.shape[2]), name="f_input_word")
    placeholders['gold_label'] = tf.placeholder(tf.int32, shape = (None), name="gold_label")
    return placeholders


def topic_model_nlp(weight_input,
                weight_input_word,
               bias_input,
               f_train,
               f_train_word, 
               y_train,
               f_val,
               f_val_word, 
               y_val,
               n_concept,
               n_concept_word,
               thres=0.5,
               load=False):
  """Returns main function of topic model."""
  # f_input size (None, 8,8,2048)
  #input = Input(shape=(299,299,3), name='input')
  #f_input = get_feature(input)
  
  placeholders = construct_placeholders(f_train, f_train_word)

  f_input = placeholders['f_input'] #Input(shape=(f_train.shape[1],f_train.shape[2]), name='f_input')
  f_input_word = placeholders['f_input_word']

  f_input_n = tf.math.l2_normalize(f_input, axis=2) #Lambda(lambda x:K.l2_normalize(x,axis=(2)))(f_input)
  f_input_n_word = tf.math.l2_normalize(f_input_word, axis=2)

  # topic vector size (2048,n_concept)
  topic_vector = tf.get_variable("weight_1", [f_train.shape[2], n_concept]) #Weight((f_train.shape[2], n_concept))(f_input)
  topic_vector_word = tf.get_variable("weight_1_word", [f_train_word.shape[2], n_concept_word])
  topic_vector_n = tf.math.l2_normalize(topic_vector, axis=0) #Lambda(lambda x: K.l2_normalize(x, axis=0))(topic_vector)
  topic_vector_n_word = tf.math.l2_normalize(topic_vector_word, axis=0)

  # topic prob = batchsize * 8 * 8 * n_concept
  #topic_prob = Weight_instance((n_concept))(f_input)
  topic_prob = tf.matmul(f_input, topic_vector_n) #Lambda(lambda x:K.dot(x[0],x[1]))([f_input, topic_vector_n])
  topic_prob_word = tf.matmul(f_input_word, topic_vector_n_word)

  topic_prob_n = tf.matmul(f_input_n, topic_vector_n) #Lambda(lambda x:K.dot(x[0],x[1]))([f_input_n, topic_vector_n])
  topic_prob_n_word = tf.matmul(f_input_n_word, topic_vector_n_word)

  topic_prob_mask = tf.cast(tf.greater(topic_prob_n, thres), 'float32') #Lambda(lambda x:K.cast(K.greater(x,thres),'float32'))(topic_prob_n)
  topic_prob_mask_word = tf.cast(tf.greater(topic_prob_n_word, thres), 'float32')

  topic_prob_am = tf.multiply(topic_prob,topic_prob_mask) #Lambda(lambda x:x[0]*x[1])([topic_prob,topic_prob_mask])
  topic_prob_am_word = tf.multiply(topic_prob_word,topic_prob_mask_word)

  #topic_prob_pos = Lambda(lambda x: K.maximum(x,-1000))(topic_prob)
  #print(K.sum(topic_prob, axis=3, keepdims=True))
  topic_prob_sum = tf.add(tf.reduce_sum(topic_prob_am, axis = 2, keepdims=True), 1e-3) #Lambda(lambda x: K.sum(x, axis=2, keepdims=True)+1e-3)(topic_prob_am)
  topic_prob_sum_word = tf.add(tf.reduce_sum(topic_prob_am_word, axis = 2, keepdims=True), 1e-3)

  topic_prob_nn = tf.divide(topic_prob_am, topic_prob_sum) #Lambda(lambda x: x[0]/x[1])([topic_prob_am, topic_prob_sum])
  topic_prob_nn_word = tf.divide(topic_prob_am_word, topic_prob_sum_word)


  # rec size is batchsize * 8 * 8 * 2048
  rec_vector_1 = tf.get_variable("weight_2", [n_concept, 420]) #Weight((n_concept, 500))(f_input)
  rec_vector_1_word = tf.get_variable("weight_2_word", [n_concept_word, 100])

  rec_vector_2 = tf.get_variable("weight_3", [420, f_train.shape[2]]) #Weight((500, f_train.shape[2]))(f_input)
  rec_vector_2_word = tf.get_variable("weight_3_word", [100, f_train_word.shape[2]])
  #rec_vector_2_word = tf.get_variable("weight_3_word", [n_concept_word, f_train_word.shape[2]])

  rec_layer_1 = tf.matmul(topic_prob_nn, rec_vector_1) #Lambda(lambda x:(K.dot(x[0],x[1])))([topic_prob_nn, rec_vector_1])
  rec_layer_1_word = tf.matmul(topic_prob_nn_word, rec_vector_1_word)
  #rec_layer_1_word = tf.matmul(topic_prob_nn_word, rec_vector_2_word) 

  rec_layer_2 = tf.matmul(rec_layer_1, rec_vector_2) #Lambda(lambda x:K.dot(x[0],x[1]))([rec_layer_1, rec_vector_2])
  rec_layer_2_word = tf.matmul(rec_layer_1_word, rec_vector_2_word)

  rec_layer_f2 = tf.reshape(rec_layer_2, [-1, f_train.shape[1]*f_train.shape[2]]) #Flatten()(rec_layer_2)
  rec_layer_f2_word = tf.reshape(rec_layer_2_word, [-1, f_train_word.shape[1]*f_train_word.shape[2]])
  #rec_layer_f2_word = tf.reshape(rec_layer_1_word, [-1, f_train_word.shape[1]*f_train_word.shape[2]])

  rec_vector_3_word = tf.get_variable("proj_word", [f_train_word.shape[1]*f_train_word.shape[2], f_train_word.shape[2]])

  rec_layer_f2_word_proj = tf.matmul(rec_layer_f2_word, rec_vector_3_word)

  weight_for_prediction = tf.get_variable("weight_4", initializer = weight_input)
  weight_for_prediction_word = tf.get_variable("weight_4_word", initializer = weight_input_word)

  logit_1 = tf.matmul(rec_layer_f2, tf.transpose(weight_input))
  logit_2 = tf.nn.bias_add(logit_1, bias_input)

  logit_1_word = tf.matmul(rec_layer_f2_word_proj, tf.transpose(weight_input_word))
  logit_2_word = tf.nn.bias_add(logit_1_word, bias_input)

  log_probs = tf.nn.log_softmax(logit_2, axis=-1)
  predictions = tf.argmax(logit_2, axis=-1, output_type=tf.int32)
  y_true = placeholders['gold_label']
  y_true_onehot = tf.one_hot(y_true, depth=3, dtype=tf.float32)

  log_probs_word = tf.nn.log_softmax(logit_2_word, axis=-1)
  predictions_word = tf.argmax(logit_2_word, axis=-1, output_type=tf.int32)

  #rec_layer_n =  Lambda(lambda x:K.l2_normalize(x,axis=(3)))(rec_layer)
  #pred = predict(rec_layer_f2)
  #topic_model_pr = Model(inputs=f_input, outputs=pred)
  #topic_model_pr.layers[-1].trainable = True
  #topic_model_pr.layers[1].trainable = False
  #if opt =='sgd':
  #  optimizer = SGD(lr=0.001)
  #  optimizer_state = [optimizer.iterations, optimizer.lr,
  #        optimizer.momentum, optimizer.decay]
  #  optimizer_reset = tf.compat.v1.variables_initializer(optimizer_state)
  #elif opt =='adam':
  #  # These depend on the optimizer class
  #  optimizer = Adam(lr=0.001)
  #  optimizer_state = [optimizer.iterations, optimizer.lr, optimizer.beta_1,
  #                           optimizer.beta_2, optimizer.decay]
  #  optimizer_reset = tf.compat.v1.variables_initializer(optimizer_state)

  # Later when you want to reset the optimizer
  #K.get_session().run(optimizer_reset)
  #print(metric1)
  #metric1.append(mean_sim(topic_prob_n, n_concept))
  #topic_model_pr.compile(
  #    loss=topic_loss_nlp(topic_prob_n, topic_vector_n,  n_concept, f_input, loss1=loss1),
  #    optimizer=optimizer,metrics=metric1)
  #print(topic_model_pr.summary())
  #if load:
  #  topic_model_pr.load_weights(load)
  #topic_model_pr.layers[-3].set_weights([np.zeros((2048,1000))])
  #topic_model_pr.layers[-3].trainable = False
  loss = tf.reduce_mean(-tf.reduce_sum(y_true_onehot * log_probs, axis=-1)) + tf.reduce_mean(-tf.reduce_sum(y_true_onehot * log_probs_word, axis=-1)) - 0.1*tf.reduce_mean(input_tensor=(tf.nn.top_k(tf.transpose(tf.reshape(topic_prob_n,(-1,n_concept))),k=16,sorted=True).values)) - 0.1*tf.reduce_mean(input_tensor=(tf.nn.top_k(tf.transpose(tf.reshape(topic_prob_n_word,(-1,n_concept_word))),k=16,sorted=True).values))  + 0.1 *tf.reduce_mean(input_tensor=(tf.matmul(tf.transpose(topic_vector_n), topic_vector_n) - np.eye(n_concept))) + 0.1 *tf.reduce_mean(input_tensor=(tf.matmul(tf.transpose(topic_vector_n_word), topic_vector_n_word) - np.eye(n_concept_word)))
  
  vars_list   = tf.trainable_variables()
  lossL2 = tf.add_n([ tf.nn.l2_loss(v) for v in vars_list if 'bias' not in v.name ]) * 0.001
  loss += lossL2

  mean_sim_val = 1*tf.reduce_mean(input_tensor=tf.nn.top_k(tf.transpose(tf.reshape(topic_prob_n,(-1,n_concept))),k=16,sorted=True).values)
  
  optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

  train_op = optimizer.minimize(loss)
  return train_op, loss, y_true, predictions, predictions_word, mean_sim_val, topic_vector_n, topic_vector, topic_vector_word, rec_vector_2, placeholders

def minibatch_end(batch_num, batch_size, train_features):
  return batch_num * batch_size >= len(train_features)
  
def next_minibatch_feed_dict(batch_num, batch_size, train_features):
    start_idx = batch_num * batch_size
    end_idx = min(start_idx + batch_size, len(train_features))
    batch_features = train_features[start_idx : end_idx]
    return batch_feed_dict(batch_features)
    
def batch_feed_dict(batch_features):
    batch_f_train, batch_f_train_word, batch_y_train = [], [], []
    for i in range(len(batch_features)):
        batch_f_train.append(batch_features[i][0])
        batch_f_train_word.append(batch_features[i][1])
        batch_y_train.append(batch_features[i][2])
    batch_f_train = np.array(batch_f_train, dtype = np.float32)
    batch_f_train_word = np.array(batch_f_train_word, dtype = np.float32)
    batch_y_train = np.array(batch_y_train, dtype = np.int32)
    return batch_f_train, batch_f_train_word, batch_y_train

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def acc_and_f1macro(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1_macro = f1_score(y_true=labels, y_pred=preds, average='macro')
    return {
        "acc": acc,
        "f1_macro": f1_macro,
    }


def main(_):
  n_concept = FLAGS.n_concept
  n_concept_word = FLAGS.n_concept_word
  batch_size = FLAGS.batch_size
  epochs = FLAGS.epochs
  
  # Loads data.
  #x_train = SemEval2014AtscProcessor().get_train_examples(FLAGS.data_dir)
  x_val = SemEval2014AtscProcessor().get_test_examples(FLAGS.data_dir)

  label_list = SemEval2014AtscProcessor().get_labels()
  tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
  #x_train_features = convert_examples_to_features(x_train, label_list, FLAGS.max_seq_length, tokenizer)
  x_val_features = convert_examples_to_features(x_val, label_list, FLAGS.max_seq_length, tokenizer)

  y_train = np.load('twitter_train_gold_label.npy')
  y_val = np.load('twitter_test_gold_label.npy')

  # Loads model
  prediction_weight_1 = np.load('twitter_output_weights_val.npy')
  prediction_weight_2 = np.load('twitter_output_weights_for_graphsage_val.npy')
  prediction_weight = np.concatenate((prediction_weight_1, prediction_weight_2), axis = -1)
  
  prediction_bias = np.load('twitter_output_bias_val.npy')
  
  # get feature
  f_train = np.load('twitter_train_cls_features.npy')
  f_train_word = np.load('twitter_train_sequence_features.npy')
  f_val = np.load('twitter_test_cls_features.npy')
  f_val_word = np.load('twitter_test_sequence_features.npy')
  
  dim_h = f_train.shape[-1]
  print(f_train.shape)
  f_train = f_train.reshape(-1,1,dim_h)
  f_val = f_val.reshape(-1,1,dim_h)
  rem_sp = f_train_word.shape[1] - f_val_word.shape[1]
  rem_sp_arr = np.zeros((f_val_word.shape[0], rem_sp, f_val_word.shape[2]))
  f_val_word = np.concatenate((f_val_word, rem_sp_arr), axis=1)
  print(f_train.shape, f_train_word.shape, f_val_word.shape)
  
  f_train_and_y_train = []
  for i in range(f_train.shape[0]):
      f_train_and_y_train.append((f_train[i], f_train_word[i], y_train[i]))
  f_test_and_y_test = []
  for i in range(f_val.shape[0]):
      f_test_and_y_test.append((f_val[i], f_val_word[i], y_val[i]))

  # building concept based explainer model
  trained = False
  thres_array = [0.3]
  
  if not trained:
    for count,thres in enumerate(thres_array):
      if count:
        load = FLAGS.output_dir+'latest_topic_nlp.h5'
      else:
        load = False
      #load = 'latest_topic_nlp.h5'
      train_op, loss, y_true, predictions, predictions_word, mean_sim, topic_vector_n, topic_vector, topic_vector_word, rec_vector_2, placeholders = topic_model_nlp(prediction_weight,
                                        prediction_weight_1,
                                        prediction_bias,
                                        f_train,
                                        f_train_word,
                                        y_train,
                                        f_val,
                                        f_val_word,
                                        y_val,
                                        n_concept,
                                        n_concept_word,
                                        thres=thres,
                                        load=load)
                                        
      config = tf.ConfigProto(log_device_placement=False)
      config.gpu_options.allow_growth = True
      #config.gpu_options.per_process_gpu_memory_fraction = GPU_MEM_FRACTION
      config.allow_soft_placement = True
    
      # Initialize session
      sess = tf.Session(config=config)
      merged = tf.summary.merge_all()
      summary_writer = tf.summary.FileWriter(FLAGS.output_dir, sess.graph)
     
      # Init variables
      sess.run(tf.global_variables_initializer())
   
      max_test_acc = 0.0
      max_test_acc_word = 0.0
      best_topic_vector = None
      best_topic_vector_word = None
      best_rec_vector = None
      best_test_prediction = None
      for epoch in range(epochs):
          random.shuffle(f_train_and_y_train)
          iter_num_train = 0
          iter_num_test = 0
          train_epoch_loss = 0.0
          train_predictions = []
          train_predictions_word = []
          train_labels = []
          test_epoch_loss = 0.0
          test_predictions = []
          test_predictions_word = []
          test_labels = []        
          print('Epoch: %04d' % (epoch + 1))
          while not minibatch_end(iter_num_train, batch_size, f_train_and_y_train):
              batch_f_train, batch_f_train_word, batch_y_train = next_minibatch_feed_dict(iter_num_train, batch_size, f_train_and_y_train)
            
              batch_feed_dict_train = {placeholders['f_input']:batch_f_train, placeholders['f_input_word']:batch_f_train_word, placeholders['gold_label']:batch_y_train}

              outs_train = sess.run([train_op, loss, y_true, predictions, predictions_word, mean_sim, topic_vector_n, topic_vector, rec_vector_2], feed_dict = batch_feed_dict_train)
              
              train_epoch_loss += outs_train[1]
              train_predictions.extend(outs_train[3])
              train_predictions_word.extend(outs_train[4])
              train_labels.extend(outs_train[2])
              iter_num_train += 1
              
          train_epoch_loss = train_epoch_loss/iter_num_train
          train_epoch_result = acc_and_f1macro(np.array(train_predictions, dtype=np.int32), np.array(train_labels, dtype=np.int32)) 
          train_epoch_result_word = acc_and_f1macro(np.array(train_predictions_word, dtype=np.int32), np.array(train_labels, dtype=np.int32)) 
          print('train loss: '+str(train_epoch_loss)+' train acc: '+str(train_epoch_result['acc'])+ ' train acc. wrt. word importance: '+str(train_epoch_result_word['acc']))

          while not minibatch_end(iter_num_test, batch_size, f_test_and_y_test):
              batch_f_test, batch_f_test_word, batch_y_test = next_minibatch_feed_dict(iter_num_test, batch_size, f_test_and_y_test)

              batch_feed_dict_test = {placeholders['f_input']:batch_f_test, placeholders['f_input_word']:batch_f_test_word, placeholders['gold_label']:batch_y_test}

              outs_test = sess.run([loss, y_true, predictions, predictions_word, mean_sim, topic_vector_n, topic_vector, rec_vector_2, topic_vector_word], feed_dict = batch_feed_dict_test)

              test_epoch_loss += outs_test[0]
              test_predictions.extend(outs_test[2])
              test_predictions_word.extend(outs_test[3])
              test_labels.extend(outs_test[1])
              iter_num_test += 1

          test_epoch_loss = test_epoch_loss/iter_num_test
          test_epoch_result = acc_and_f1macro(np.array(test_predictions, dtype=np.int32), np.array(test_labels, dtype=np.int32))
          test_epoch_result_word = acc_and_f1macro(np.array(test_predictions_word, dtype=np.int32), np.array(test_labels, dtype=np.int32)) 
          print('test loss: '+str(test_epoch_loss)+' test acc: '+str(test_epoch_result['acc'])+ ' test acc. wrt. word importance: '+str(test_epoch_result_word['acc']))
          if test_epoch_result['acc'] > max_test_acc:
              max_test_acc = test_epoch_result['acc']
              best_topic_vector = outs_test[6]
              best_rec_vector = outs_test[7]
              best_test_prediction = test_predictions

          if test_epoch_result_word['acc'] > max_test_acc_word:
              max_test_acc_word = test_epoch_result_word['acc']
              best_topic_vector_word = outs_test[8]

    print('Maximum test accuracy after feeding concept vector over the epochs: '+str(max_test_acc))
    print('Maximum test accuracy after feeding word based concept vector over the epochs: '+str(max_test_acc_word))
    np.save('twitter_concept_vector.npy', best_topic_vector)
    np.save('twitter_concept_vector_word.npy', best_topic_vector_word)
    np.save('twitter_concept_based_explainer_weight.npy', best_rec_vector)
  else:
    best_topic_vector = np.load('twitter_concept_vector.npy')
    best_topic_vector_word = np.load('twitter_concept_vector_word.npy')
    best_rec_vector = np.load('twitter_concept_based_explainer_weight.npy')

  f_val_n = f_val/(np.linalg.norm(f_val,axis=2,keepdims=True)+1e-9)
  f_val_word_n = f_val_word/(np.linalg.norm(f_val_word,axis=2,keepdims=True)+1e-9)

  topic_vec_n = best_topic_vector/(np.linalg.norm(best_topic_vector,axis=0,keepdims=True)+1e-9)
  topic_vector_n_word = best_topic_vector_word/(np.linalg.norm(best_topic_vector_word,axis=0,keepdims=True)+1e-9)

  topic_prob = np.matmul(f_val_n,topic_vec_n)
  sp_1, sp_2 = topic_prob.shape[1:]
  topic_prob = topic_prob.reshape(-1,sp_1*sp_2 )
  topic_prob_word = np.matmul(f_val_word_n,topic_vector_n_word)
  sp_1_word, sp_2_word = topic_prob_word.shape[1:]
  print(topic_prob.shape, topic_prob_word.shape)
  
  concept_idx_example_wise = np.argmax(topic_prob, axis = -1).tolist()
  '''
  for i in range(n_concept):
    print('concept:{}'.format(i))
    for j in range(len(concept_idx_example_wise)):
      if concept_idx_example_wise[j] == i:
        print(x_val[j].text_a+'\t'+x_val[j].text_b+'\t'+x_val[j].label+'\n')
    print('\n')
  '''

  label_idx_to_label_val = {}
  for label_idx, label_val in enumerate(["positive", "negative", "neutral"]):
    label_idx_to_label_val[label_idx] = label_val

  fp_wr = open('twitter_dataset_concept_based_explanation.txt', 'w')
  top_6_word_list_all_examples = []
  for i in range(len(x_val)):
    top_6_word_list_per_example = []
    sentence_concept_idx = np.argmax(topic_prob[i], axis = -1)
    word_concept = topic_prob_word[i]
    top_6_word_concept_idx = np.max(word_concept,axis=1).argsort()[-6:][::-1]
    example_token_ids = x_val_features[i][0].input_ids
    example_tokens = tokenizer.convert_ids_to_tokens(example_token_ids)
    example_cand_indices = []
    for tok_idx in range(len(example_tokens)):
      if example_tokens[tok_idx] != '[CLS]' and example_tokens[tok_idx] != '[SEP]' and example_tokens[tok_idx] != '[PAD]':
        if example_tokens[tok_idx].startswith("##"):
          example_cand_indices[-1].append(tok_idx)
        else:
          example_cand_indices.append([tok_idx])

    example_words = []
    for idx_list in  example_cand_indices:
        s_i = idx_list[0]
        e_i = idx_list[-1] + 1
        printable_token_list = example_tokens[s_i:e_i]
        printable_word = ''
        for printable_tok_val in printable_token_list:
            if printable_tok_val.startswith('##'):
                printable_word += printable_tok_val[2:].strip()
            else:
                printable_word += printable_tok_val.strip()
        example_words.append(printable_word)

    fp_wr.write('Sentence: '+x_val[i].text_a+'\n')
    fp_wr.write('Aspect: '+x_val[i].text_b+'\n')
    fp_wr.write('Gold Label: '+x_val[i].label+'\n')
    fp_wr.write('Predicted label: '+label_idx_to_label_val[best_test_prediction[i]]+'\n')
    fp_wr.write('Top Concept: '+str(sentence_concept_idx)+'\n')
    fp_wr.write('Top 6 words: '+'\n')
    for top_k_val in range(6):
        if top_6_word_concept_idx[top_k_val] >= len(example_words):
            continue
        fp_wr.write(example_words[top_6_word_concept_idx[top_k_val]]+'\n')
        top_6_word_list_per_example.append(example_words[top_6_word_concept_idx[top_k_val]])
    fp_wr.write('\n\n')
    top_6_word_list_all_examples.append(top_6_word_list_per_example)

  fp_wr.close()
  with open('twitter_test_top_6_word_list_all_examples.pkl', 'wb') as fp:
    pickle.dump(top_6_word_list_all_examples, fp)


if __name__ == '__main__':
  tf.app.run(main)
 
