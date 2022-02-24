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
import numpy as np
import random
from sklearn.metrics import matthews_corrcoef, f1_score
import xml.etree.ElementTree as ET

from graphsage_models_modified import SampleAndAggregate, SAGEInfo
from graphsage_minibatch import EdgeMinibatchIterator
from graphsage_neigh_samplers import UniformNeighborSampler
from graphsage_utils import load_data

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)
random.seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "cluster_embedding_fp", None,
    "The cluster embedding file path.")

flags.DEFINE_string(
    "bert_pretrained_embedding_fp", None,
    "The BERT pretrained embedding train file path.")

flags.DEFINE_string(
    "bert_pretrained_embedding_fp_test", None,
    "The BERT pretrained embedding test file path.")

flags.DEFINE_string(
    "dataset_name", None,
    "The input dataset name.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_string(
    "init_checkpoint_graphsage", None,
    "Initial checkpoint of GraphSAGE(usually from a pre-trained GraphSAGE model).")

flags.DEFINE_string(
    "init_checkpoint_probe", None,
    "Initial checkpoint of Semantic Probe(usually from a pre-trained Semantic Probe model).")


flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 32, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 32, "Total batch size for predict.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")


tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")

flags.DEFINE_string('model', 'graphsage', 'model names. See README for possible values.')
flags.DEFINE_float("learning_rate", 3e-5, "The initial learning rate for Adam.")  
flags.DEFINE_string("model_size", "small", "Can be big or small; model specific def'ns")
flags.DEFINE_string('train_prefix', '', 'name of the object file that stores the training data. must be specified.')

flags.DEFINE_float("num_train_epochs", 5.0,
                   "Total number of training epochs to perform.") 
flags.DEFINE_float('dropout', 0.0, 'dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0.0, 'weight for l2 loss on embedding matrix.')
flags.DEFINE_integer('max_degree', 107, 'maximum node degree.')
flags.DEFINE_integer('samples_1', 25, 'number of samples in layer 1')
flags.DEFINE_integer('samples_2', 10, 'number of users samples in layer 2')
flags.DEFINE_integer('dim_1', 25, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_integer('dim_2', 25, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_boolean('random_context', True, 'Whether to use random context or direct edges')
flags.DEFINE_integer('neg_sample_size', 5, 'number of negative samples')
flags.DEFINE_integer('graphsage_batch_size', 1822, 'minibatch size.')
flags.DEFINE_integer('n2v_test_epochs', 1, 'Number of new SGD epochs for n2v.')
flags.DEFINE_integer('identity_dim', 128, 'Set to positive value to use identity embedding features of that dimension. Default 0.')

#logging, saving, validation settings etc.
flags.DEFINE_boolean('save_embeddings', True, 'whether to save embeddings for all nodes after training')
flags.DEFINE_string('base_log_dir', '.', 'base directory for logging and saving embeddings')
flags.DEFINE_integer('validate_iter', 5000, "how often to run a validation minibatch.")
flags.DEFINE_integer('validate_batch_size', 1822, "how many nodes per validation sample.")
flags.DEFINE_integer('gpu', 0, "which gpu to use.")
flags.DEFINE_integer('print_every', 50, "How often to print training info.")
flags.DEFINE_integer('max_total_steps', 10**10, "Maximum total number of iterations")

flags.DEFINE_integer('probe_hidden_size', 100, 'probe weight shape.')

os.environ["CUDA_VISIBLE_DEVICES"]=str(FLAGS.gpu)

GPU_MEM_FRACTION = 0.8

def log_dir():
    log_dir = FLAGS.base_log_dir + "/unsup-" + FLAGS.train_prefix.split("/")[-2]
    log_dir += "/{model:s}_{model_size:s}_{lr:0.6f}/".format(
            model=FLAGS.model,
            model_size=FLAGS.model_size,
            lr=FLAGS.learning_rate)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    return log_dir

# Define model evaluation function
def evaluate(sess, model, minibatch_iter, size=None):
    t_test = time.time()
    feed_dict_val = minibatch_iter.val_feed_dict(size)
    outs_val = sess.run([model.loss, model.ranks, model.mrr], 
                        feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], outs_val[2], (time.time() - t_test)

def incremental_evaluate(sess, model, minibatch_iter, size):
    t_test = time.time()
    finished = False
    val_losses = []
    val_mrrs = []
    iter_num = 0
    while not finished:
        feed_dict_val, finished, _ = minibatch_iter.incremental_val_feed_dict(size, iter_num)
        iter_num += 1
        outs_val = sess.run([model.loss, model.ranks, model.mrr], 
                            feed_dict=feed_dict_val)
        val_losses.append(outs_val[0])
        val_mrrs.append(outs_val[2])
    return np.mean(val_losses), np.mean(val_mrrs), (time.time() - t_test)

def save_val_embeddings(sess, model, minibatch_iter, size, mod=""):
    val_embeddings = []
    finished = False
    seen = set([])
    nodes = []
    iter_num = 0
    name = "val"
    while not finished:
        feed_dict_val, finished, edges = minibatch_iter.incremental_embed_feed_dict(size, iter_num)
        iter_num += 1
        outs_val = sess.run([model.outputs1], 
                            feed_dict=feed_dict_val)
        #ONLY SAVE FOR embeds1 because of planetoid
        for i, edge in enumerate(edges):
            if not edge[0] in seen:
                val_embeddings.append(outs_val[-1][i,:])
                nodes.append(edge[0])
                seen.add(edge[0])
    
    val_embeddings = np.vstack(val_embeddings)
    return val_embeddings, nodes

def construct_placeholders():
    # Define placeholders
    placeholders = {}
    # bert
    placeholders['input_ids'] = tf.placeholder(tf.int32, shape = (None, FLAGS.max_seq_length), name="input_ids") # FLAGS.batch_size
    placeholders['input_mask'] = tf.placeholder(tf.int32, shape = (None, FLAGS.max_seq_length), name="input_mask")
    placeholders['segment_ids'] = tf.placeholder(tf.int32, shape = (None, FLAGS.max_seq_length), name="segment_ids")
    placeholders['label_ids'] = tf.placeholder(tf.int32, shape = (None), name="label_ids")
    placeholders['is_training'] = tf.placeholder(tf.bool, name="is_training") # FLAGS.is_training
    placeholders['graphsage_embeddings'] = tf.placeholder(tf.float32, shape = (None, 4*(FLAGS.dim_1)), name="graphsage_embeddings")
    # graphsage
    placeholders['batch1'] = tf.placeholder(tf.int32, shape=(None), name='batch1')
    placeholders['batch2'] = tf.placeholder(tf.int32, shape=(None), name='batch2')
    # negative samples for all nodes in the batch
    placeholders['neg_samples'] = tf.placeholder(tf.int32, shape=(None,), name='neg_sample_size')
    placeholders['dropout'] = tf.placeholder_with_default(0., shape=(), name='dropout')
    placeholders['batch_size'] = tf.placeholder(tf.int32, name='batch_size')
    # semantic probe
    placeholders['h_i'] = tf.placeholder(tf.float32, shape=(None, 768), name='bert_embedding_of_entity_i')
    placeholders['h_j'] = tf.placeholder(tf.float32, shape=(None, 768), name='bert_embedding_of_entity_j')
    placeholders['h_k'] = tf.placeholder(tf.float32, shape=(None, 768), name='bert_embedding_of_entity_k')
    return placeholders

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
        with open(unique_entity_corpus) as fp:
            for line in fp:
                unique_entity_list.append(line.strip())
        with open(example_wise_entity_corpus) as fp:
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


def minibatch_end(batch_num, batch_size, train_features):
  return batch_num * batch_size >= len(train_features)

def next_minibatch_feed_dict(batch_num, batch_size, train_features):
  start_idx = batch_num * batch_size
  end_idx = min(start_idx + batch_size, len(train_features))
  batch_features = train_features[start_idx : end_idx]
  return batch_feed_dict(batch_features)

def batch_feed_dict(batch_features):
  input_ids_list, input_mask_list, segment_ids_list, label_id_list, entity_indices, embedding_array = [], [], [], [], [], []
  
  for i in range(len(batch_features)):
    input_ids_list.append(batch_features[i][0][0].input_ids)
    input_mask_list.append(batch_features[i][0][0].input_mask)
    segment_ids_list.append(batch_features[i][0][0].segment_ids)
    label_id_list.append(batch_features[i][0][0].label_id)
    entity_indices.append(batch_features[i][0][1])
    embedding_array.append(batch_features[i][1])

  embedding_array = np.array(embedding_array, dtype = np.float32)
  embedding_i = embedding_array[:, 0:768]
  embedding_j = embedding_array[:, 768:1536]
  embedding_k = embedding_array[:, 1536:2304]

    

  return input_ids_list, input_mask_list, segment_ids_list, label_id_list, entity_indices, embedding_i, embedding_j, embedding_k

def simple_accuracy(preds, labels):
    return (preds == labels).mean()

def acc_and_f1macro(preds, labels):
    acc = simple_accuracy(preds, labels)
    f1_macro = f1_score(y_true=labels, y_pred=preds, average='macro')
    return {
        "acc": acc,
        "f1_macro": f1_macro,
    }


def train(bert_config, num_labels, init_checkpoint, learning_rate, bert_num_train_steps, num_warmup_steps, use_tpu, use_one_hot_embeddings, bert_train_examples, graphsage_train_data, label_list, FLAGS, tokenizer, processor):
    
    #bert
    bert_train_features = convert_examples_to_features(bert_train_examples, label_list, FLAGS.max_seq_length, tokenizer)

    # cluster kg
    cluster_embeddings = np.load(FLAGS.cluster_embedding_fp)

    # h_i, h_j, h_k bert pre-trained embeddings
    pretrained_embeddings = np.load(FLAGS.bert_pretrained_embedding_fp)
     
    bert_train_features_mod = []
    for idx in range(len(bert_train_features)):
        bert_train_features_mod.append((bert_train_features[idx], pretrained_embeddings[idx]))
      
    # graphsage
    G = graphsage_train_data[0]
    features = graphsage_train_data[1]
    id_map = graphsage_train_data[2]

    if not features is None:
        # pad with dummy zero vector
        features = np.vstack([features, np.zeros((features.shape[1],))])

    context_pairs = graphsage_train_data[3] if FLAGS.random_context else None
    placeholders = construct_placeholders()
    
    minibatch = EdgeMinibatchIterator(G, 
            id_map,
            placeholders, batch_size=FLAGS.graphsage_batch_size,
            max_degree=FLAGS.max_degree, 
            num_neg_samples=FLAGS.neg_sample_size,
            context_pairs = context_pairs)
    adj_info_ph = tf.placeholder(tf.int32, shape=minibatch.adj.shape)
    adj_info = tf.Variable(adj_info_ph, trainable=False, name="adj_info")

    if FLAGS.model == 'graphsage_mean':
        # Create model
        """Creates a bert model."""
        bert_model = modeling_modified.BertModel(
            config=bert_config,
            is_training=placeholders['is_training'],
            input_ids=placeholders['input_ids'],
            input_mask=placeholders['input_mask'],
            token_type_ids=placeholders['segment_ids'],
            use_one_hot_embeddings=use_one_hot_embeddings)

        bert_output_layer = bert_model.get_pooled_output()

        bert_hidden_size = bert_output_layer.shape[-1].value
        

        """Creates a graphsage model."""
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                            SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

        graphsage_model = SampleAndAggregate(placeholders, 
                                     features,
                                     adj_info,
                                     minibatch.deg,
                                     layer_infos=layer_infos, 
                                     model_size=FLAGS.model_size,
                                     identity_dim = FLAGS.identity_dim,
                                     logging=True)

        graphsage_output_layer = placeholders['graphsage_embeddings']

        graphsage_hidden_size = graphsage_output_layer.shape[-1].value

        """Creates semantic probe model."""
        B = tf.get_variable(
            "probe_weights", [768, FLAGS.probe_hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02),
            trainable= True)

        transformed_i_in_B_space = tf.matmul(placeholders['h_i'], B) # B(h_i) (bs*100)
        transformed_j_in_B_space = tf.matmul(placeholders['h_j'], B) # B(h_j) (bs*100)
        transformed_k_in_B_space = tf.matmul(placeholders['h_k'], B) # B(h_k) (bs*100)

        sigmoid_dot_i_j_in_B_space =  tf.sigmoid(tf.reduce_sum(tf.multiply(transformed_i_in_B_space, transformed_j_in_B_space), axis = -1)) # sigmoid(B(h_i)^TB(h_j)) shape:(bs)
        sigmoid_dot_i_k_in_B_space =  tf.sigmoid(tf.reduce_sum(tf.multiply(transformed_i_in_B_space, transformed_k_in_B_space), axis = -1)) # sigmoid(B(h_i)^TB(h_k)) shape:(bs) 

        per_example_sigmoid_dot_diff = sigmoid_dot_i_j_in_B_space - sigmoid_dot_i_k_in_B_space # (bs)

        sigmoid_dot_diff = tf.reduce_mean(per_example_sigmoid_dot_diff)
   
        probe_loss = -sigmoid_dot_diff

        score = tf.cast(tf.math.greater_equal(sigmoid_dot_i_j_in_B_space, sigmoid_dot_i_k_in_B_space), dtype = tf.float32)

        graphsage_output_layer_modified = tf.multiply(graphsage_output_layer, tf.expand_dims(score,1))
        
        """Combine bert and graphsage model."""
        output_layer = tf.concat([bert_output_layer, graphsage_output_layer_modified], -1)

        output_hidden_size = bert_hidden_size + graphsage_hidden_size

        output_weights_for_bert = tf.get_variable(
            "output_weights", [num_labels, bert_hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_weights_for_graphsage = tf.get_variable(
            "output_weights_for_graphsage", [num_labels, graphsage_hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))


        output_bias = tf.get_variable(
            "output_bias", [num_labels], initializer=tf.zeros_initializer())

        output_weights = tf.concat([output_weights_for_bert, output_weights_for_graphsage], -1)

        def apply_dropout_last_layer(output_layer):
            output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)
            return output_layer

        def not_apply_dropout(output_layer):
            return output_layer

        output_layer=tf.cond(placeholders['is_training'], lambda: apply_dropout_last_layer(output_layer), lambda:not_apply_dropout(output_layer))

        logits = tf.matmul(output_layer, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        probabilities = tf.nn.softmax(logits, axis=-1)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        labels = placeholders['label_ids']
        one_hot_labels = tf.one_hot(placeholders['label_ids'], depth=num_labels, dtype=tf.float32)

        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        bert_loss = tf.reduce_mean(per_example_loss)



        # TF graph management
        graphsage_model_loss = 0.0
        for aggregator in graphsage_model.aggregators:
            for var in aggregator.vars.values():
                graphsage_model_loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        graphsage_model_loss += graphsage_model.link_pred_layer.loss(graphsage_model.outputs1, graphsage_model.outputs2, graphsage_model.neg_outputs) 
        graphsage_model_loss = graphsage_model_loss / tf.cast(placeholders["batch_size"], tf.float32)
        total_loss = bert_loss + graphsage_model_loss + probe_loss
 
        tf.summary.scalar('loss', total_loss)

        tvars = tf.trainable_variables() # returns all trainable variable names
        
        # for initializing bert variables
        initialized_variable_names = {}

        if init_checkpoint:
            (assignment_map, initialized_variable_names
            ) = modeling_modified.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
    
            tf.train.init_from_checkpoint(init_checkpoint, assignment_map)


        # for initializing graphsage variables
        initialized_variable_names_2 = {}

        if FLAGS.init_checkpoint_graphsage:
            (assignment_map_2, initialized_variable_names_2
            ) = modeling_modified.get_assignment_map_from_checkpoint(tvars, FLAGS.init_checkpoint_graphsage)
    
            tf.train.init_from_checkpoint(FLAGS.init_checkpoint_graphsage, assignment_map_2)

        # for initializing semantic probe variables
        initialized_variable_names_3 = {}

        if FLAGS.init_checkpoint_probe:
            (assignment_map_3, initialized_variable_names_3
            ) = modeling_modified.get_assignment_map_from_checkpoint(tvars, FLAGS.init_checkpoint_probe)
    
            tf.train.init_from_checkpoint(FLAGS.init_checkpoint_probe, assignment_map_3)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names or var.name in initialized_variable_names_2:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape, init_string)
        
        train_op = optimization.create_optimizer(
                total_loss, learning_rate, bert_num_train_steps, num_warmup_steps, use_tpu)

        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)

    elif FLAGS.model == 'gcn':
        # Create model
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, 2*FLAGS.dim_1),
                            SAGEInfo("node", sampler, FLAGS.samples_2, 2*FLAGS.dim_2)]

        model = SampleAndAggregate(placeholders, 
                                     features,
                                     adj_info,
                                     minibatch.deg,
                                     layer_infos=layer_infos, 
                                     aggregator_type="gcn",
                                     model_size=FLAGS.model_size,
                                     identity_dim = FLAGS.identity_dim,
                                     concat=False,
                                     logging=True)

    elif FLAGS.model == 'graphsage_seq':
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                            SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

        model = SampleAndAggregate(placeholders, 
                                     features,
                                     adj_info,
                                     minibatch.deg,
                                     layer_infos=layer_infos, 
                                     identity_dim = FLAGS.identity_dim,
                                     aggregator_type="seq",
                                     model_size=FLAGS.model_size,
                                     logging=True)

    elif FLAGS.model == 'graphsage_maxpool':
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                            SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

        model = SampleAndAggregate(placeholders, 
                                    features,
                                    adj_info,
                                    minibatch.deg,
                                     layer_infos=layer_infos, 
                                     aggregator_type="maxpool",
                                     model_size=FLAGS.model_size,
                                     identity_dim = FLAGS.identity_dim,
                                     logging=True)
    elif FLAGS.model == 'graphsage_meanpool':
        sampler = UniformNeighborSampler(adj_info)
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                            SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

        model = SampleAndAggregate(placeholders, 
                                    features,
                                    adj_info,
                                    minibatch.deg,
                                     layer_infos=layer_infos, 
                                     aggregator_type="meanpool",
                                     model_size=FLAGS.model_size,
                                     identity_dim = FLAGS.identity_dim,
                                     logging=True)

    elif FLAGS.model == 'n2v':
        model = Node2VecModel(placeholders, features.shape[0],
                                       minibatch.deg,
                                       #2x because graphsage uses concat
                                       nodevec_dim=2*FLAGS.dim_1,
                                       lr=FLAGS.learning_rate)
    else:
        raise Exception('Error: model name unrecognized.')

    config = tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)
    config.gpu_options.allow_growth = True
    #config.gpu_options.per_process_gpu_memory_fraction = GPU_MEM_FRACTION
    config.allow_soft_placement = True
    
    # Initialize session
    sess = tf.Session(config=config)
    merged = tf.summary.merge_all()
    summary_writer = tf.summary.FileWriter(log_dir(), sess.graph)
     
    # Init variables
    sess.run(tf.global_variables_initializer(), feed_dict={adj_info_ph: minibatch.adj})
    
    # Train model
    saver = tf.train.Saver() # to save and restore training variables

    total_steps = 0
    max_acc = 0.0
    best_pred, best_prob, best_label = [], [], []

    train_adj_info = tf.assign(adj_info, minibatch.adj)
    val_adj_info = tf.assign(adj_info, minibatch.test_adj)

    for epoch in range(int(FLAGS.num_train_epochs)):
        random.shuffle(bert_train_features) 
        minibatch.shuffle() 

        iter_num = 0
        print('Epoch: %04d' % (epoch + 1))

        train_epoch_loss = 0.0
        train_predictions = []
        train_labels = []
        
        while not minibatch_end(iter_num, FLAGS.train_batch_size, bert_train_features_mod):
            # Construct feed dictionary
            input_ids_list, input_mask_list, segment_ids_list, label_id_list, entity_indices, embedding_i, embedding_j, embedding_k = next_minibatch_feed_dict(iter_num, FLAGS.train_batch_size, bert_train_features_mod)
            bert_feed_dict = {placeholders['input_ids']:input_ids_list, placeholders['input_mask']:input_mask_list, placeholders['segment_ids']:segment_ids_list, placeholders['label_ids']:label_id_list, placeholders['is_training']:True, placeholders['h_i']:embedding_i, placeholders['h_j']:embedding_j, placeholders['h_k']:embedding_k}
            node_embeddings, node_list = save_val_embeddings(sess, graphsage_model, minibatch, FLAGS.validate_batch_size)
            
            mapped_entity_indices, mapped_node_embeddings = [], []
            for idx in entity_indices:
                mapped_entity_indices.append(int(node_list.index(idx)))
            
            for idx in mapped_entity_indices:
                
                mapped_node_embeddings.append(node_embeddings[idx])
            
            mapped_node_embeddings = np.array(mapped_node_embeddings, dtype = np.float32)

            mapped_cluster_embeddings = []
            
            for idx in entity_indices:
                mapped_cluster_embeddings.append(cluster_embeddings[idx])

            mapped_cluster_embeddings = np.array(mapped_cluster_embeddings, dtype = np.float32)

            mapped_node_embeddings_concat = np.concatenate((mapped_node_embeddings, mapped_cluster_embeddings), axis=1)

            bert_feed_dict.update({placeholders['graphsage_embeddings']: mapped_node_embeddings_concat})
            
            graphsage_feed_dict = minibatch.next_minibatch_feed_dict()
            graphsage_feed_dict.update({placeholders['dropout']: FLAGS.dropout})

            feed_dict = {**bert_feed_dict, **graphsage_feed_dict}

            # Training step
            outs = sess.run([train_op, total_loss, per_example_loss, logits, probabilities, predictions, labels], feed_dict=feed_dict)
            train_epoch_loss += outs[1]
            train_predictions.extend(outs[5])
            train_labels.extend(outs[6])

            iter_num += 1
            total_steps += 1

        train_epoch_loss = train_epoch_loss/iter_num
        train_epoch_result = acc_and_f1macro(np.array(train_predictions, dtype=np.int32), np.array(train_labels, dtype=np.int32))

        print('train loss: '+str(train_epoch_loss)+' train acc: '+str(train_epoch_result['acc'])+' train macro f1:'+str(train_epoch_result['f1_macro']))
        

        if FLAGS.do_predict:
            bert_predict_examples = processor.get_test_examples(FLAGS.data_dir)
            bert_num_actual_predict_examples = len(bert_predict_examples)
      
            bert_predict_features = convert_examples_to_features(bert_predict_examples, label_list, FLAGS.max_seq_length, tokenizer)

            # h_i, h_j, h_k bert pre-trained embeddings
            pretrained_embeddings_test = np.load(FLAGS.bert_pretrained_embedding_fp_test)
     
            bert_predict_features_mod = []
            for idx in range(len(bert_predict_features)):
                bert_predict_features_mod.append((bert_predict_features[idx], pretrained_embeddings_test[idx]))

            tf.logging.info("***** Running prediction*****")
            tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                len(bert_predict_examples), bert_num_actual_predict_examples,
                len(bert_predict_examples) - bert_num_actual_predict_examples)
            tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)
            test_result = predict(bert_predict_features_mod, cluster_embeddings, FLAGS.predict_batch_size, placeholders, bert_loss, graphsage_model_loss, probe_loss, total_loss, per_example_loss, logits, probabilities, predictions, labels, sess, graphsage_model, minibatch, FLAGS)

            print('test loss: '+str(test_result[0])+' test acc: '+str(test_result[1])+' test macro f1: '+str(test_result[2]))
            if test_result[1] > max_acc:
                max_acc = test_result[1]
                saver.save(sess, FLAGS.output_dir+'bert_gs_e_probe_best_model.ckpt')
                best_prob = test_result[3]
                best_pred = test_result[4]
                best_label = test_result[5]

    np.save(FLAGS.output_dir+'laptop_prob.npy', np.array(best_prob))
    np.save(FLAGS.output_dir+'laptop_pred.npy', np.array(best_pred))
    np.save(FLAGS.output_dir+'laptop_label.npy', np.array(best_label))
    

def main(argv=None):
    tf.logging.set_verbosity(tf.logging.INFO)

    processors = {
        "semeval2014-atsc":SemEval2014AtscProcessor,
    }

    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                  FLAGS.init_checkpoint)

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
            "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    bert_config = modeling_modified.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model "
            "was only trained up to sequence length %d" %
            (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(FLAGS.output_dir)

    task_name = FLAGS.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()

    label_list = processor.get_labels()

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    bert_train_examples = None
    bert_num_train_steps = None
    num_warmup_steps = None

    if FLAGS.do_train:
        print("Loading training data..")
        # bert
        bert_train_examples = processor.get_train_examples(FLAGS.data_dir)
        bert_num_train_steps = int(
            len(bert_train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(bert_num_train_steps * FLAGS.warmup_proportion)

        # graphsage
        graphsage_train_data = load_data(FLAGS.train_prefix, load_walks=True)
        print("Done loading training data..")

    num_labels = len(label_list)
    init_checkpoint = FLAGS.init_checkpoint
    learning_rate = FLAGS.learning_rate
    use_tpu=FLAGS.use_tpu
    use_one_hot_embeddings=FLAGS.use_tpu

    if FLAGS.do_train:
        train(bert_config, num_labels, init_checkpoint, learning_rate, bert_num_train_steps, num_warmup_steps, use_tpu, use_one_hot_embeddings, bert_train_examples, graphsage_train_data, label_list, FLAGS, tokenizer, processor)


def predict(bert_predict_features, cluster_embeddings, predict_batch_size, placeholders, bert_loss, graphsage_model_loss, probe_loss, total_loss, per_example_loss, logits, probabilities, predictions, labels, sess, graphsage_model, minibatch, FLAGS):
    iter_num = 0

    test_epoch_loss = 0.0
    test_probabilities = []
    test_predictions = []
    test_labels = []

    while not minibatch_end(iter_num, FLAGS.predict_batch_size, bert_predict_features):
        # Construct feed dictionary
        input_ids_list, input_mask_list, segment_ids_list, label_id_list, entity_indices, embedding_i, embedding_j, embedding_k = next_minibatch_feed_dict(iter_num, FLAGS.predict_batch_size, bert_predict_features)
        bert_feed_dict = {placeholders['input_ids']:input_ids_list, placeholders['input_mask']:input_mask_list, placeholders['segment_ids']:segment_ids_list, placeholders['label_ids']:label_id_list, placeholders['is_training']:False, placeholders['h_i']:embedding_i, placeholders['h_j']:embedding_j, placeholders['h_k']:embedding_k}
        node_embeddings, node_list = save_val_embeddings(sess, graphsage_model, minibatch, FLAGS.validate_batch_size)
            
        mapped_entity_indices, mapped_node_embeddings = [], []
        for idx in entity_indices:
            mapped_entity_indices.append(int(node_list.index(idx)))
            
        for idx in mapped_entity_indices:
                
            mapped_node_embeddings.append(node_embeddings[idx])
            
        mapped_node_embeddings = np.array(mapped_node_embeddings, dtype = np.float32)

        mapped_cluster_embeddings = []
            
        for idx in entity_indices:
            mapped_cluster_embeddings.append(cluster_embeddings[idx])

        mapped_cluster_embeddings = np.array(mapped_cluster_embeddings, dtype = np.float32)

        mapped_node_embeddings_concat = np.concatenate((mapped_node_embeddings, mapped_cluster_embeddings), axis=1)

        bert_feed_dict.update({placeholders['graphsage_embeddings']: mapped_node_embeddings_concat})
            
        graphsage_feed_dict = minibatch.entire_embed_feed_dict()

        graphsage_feed_dict.update({placeholders['dropout']: FLAGS.dropout})

        feed_dict = {**bert_feed_dict, **graphsage_feed_dict}


        # Prediction step
        outs = sess.run([bert_loss, graphsage_model_loss, probe_loss, total_loss, per_example_loss, logits, probabilities, predictions, labels], feed_dict=feed_dict)       

        test_epoch_loss += outs[3]
        test_probabilities.extend(outs[6])
        test_predictions.extend(outs[7])
        test_labels.extend(outs[8])

        iter_num += 1

    test_epoch_loss = test_epoch_loss/iter_num
    test_epoch_result = acc_and_f1macro(np.array(test_predictions, dtype=np.int32), np.array(test_labels, dtype=np.int32))

    return (test_epoch_loss, test_epoch_result['acc'], test_epoch_result['f1_macro'], test_probabilities, test_predictions, test_labels)  



if __name__ == '__main__':
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
