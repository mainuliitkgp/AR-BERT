# -*- coding: utf-8 -*-
import os
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
from theano.tensor.signal import pool
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from nn_utils_sentence import *
import pickle


class LSTM:
    def __init__(self, bs, n_in, n_out, name):
        """

        :param bs: batch size
        :param n_in: input size
        :param n_out: hidden size
        :param name: alias of layer
        """
        self.bs = bs
        self.n_in = n_in
        self.n_out = n_out
        self.name = name
        # W shape: (n_in, 4*n_out)
        # U shape: (n_out, 4*n_out)
        # b shape: (4*n_out
        self.W, self.U, self.b = lstm_init(n_in=self.n_in, n_out=self.n_out, component=name)
        self.h0 = theano.shared(value=zeros(size=(self.bs, n_out)), name='h0')
        self.c0 = theano.shared(value=zeros(size=(self.bs, n_out)), name='c0')
        self.params = [self.W, self.U, self.b]

    def __str__(self):
        return "%s: LSTM(%s, %s)" % (self.name, self.n_in, self.n_out)

    __repr__ = __str__

    def __call__(self, x):
        """

        :param x: input tensor, shape: (bs, seq_len, n_in)
        :return: generated hidden states
        """
        h0 = T.zeros_like(self.h0)
        c0 = T.zeros_like(self.c0)
        rnn_input = x.dimshuffle(1, 0, 2)
        [H, _], _ = theano.scan(fn=self.recurrence, sequences=rnn_input, outputs_info=[h0, c0])
        return H.dimshuffle(1, 0, 2)

    def recurrence(self, xt, htm1, ctm1):
        """

        :param xt: x[t] \in (bs, n_in)
        :param htm1: h[t-1] \in (bs, n_out)
        :param ctm1: c[t-1] \in (bs, n_out)
        :return:
        """
        Wx = T.dot(xt, self.W)
        Uh = T.dot(htm1, self.U)
        Sum_item = Wx + Uh + self.b
        it = T.nnet.hard_sigmoid(Sum_item[:, :self.n_out])
        ft = T.nnet.hard_sigmoid(Sum_item[:, self.n_out:2*self.n_out])
        ct_tilde = T.tanh(Sum_item[:, 2*self.n_out:3*self.n_out])
        ot = T.nnet.hard_sigmoid(Sum_item[:, 3*self.n_out:])
        ct = ft * ctm1 + it * ct_tilde
        ht = ot * T.tanh(ct)
        return ht, ct


class Linear:
    """
    fully connected layer
    """
    def __init__(self, n_in, n_out, name, use_bias=True):
        """

        :param n_in: input size
        :param n_out: output size
        :param name: layer name
        :param use_bias: use bias or not
        """
        self.n_in = n_in
        self.n_out = n_out
        self.name = name
        self.use_bias = use_bias
        # sample weight from uniform distribution [-INIT_RANGE, INIT_RANGE]
        # initialize bias as zero vector
        self.W = theano.shared(value=uniform(lb=-INIT_RANGE, ub=INIT_RANGE, size=(n_in, n_out)), name="%s_W" % name)
        self.b = theano.shared(value=zeros(size=n_out), name="%s_b" % name)
        self.params = [self.W]
        if self.use_bias:
            self.params.append(self.b)

    def __str__(self):
        return "%s: Linear(%s, %s)" % (self.name, self.n_in, self.n_out)

    __repr__ = __str__

    def __call__(self, x, bs=None):
        """

        :param x: input tensor, shape: (bs, *, n_in)
        :return:
        """
        if bs is None:
            output = T.dot(x, self.W)
        else:
            # current shape: (bs, n_in, n_out)
            padded_W = T.tile(self.W, (bs, 1, 1))
            # output shape: (bs, seq_len, n_out)
            output = T.batched_dot(x, padded_W)
        if self.use_bias:
            output = output + self.b
        return output

class CNN:
    def __init__(self, bs, n_in, sent_len, kernel_size, n_filters, name):
        """

        :param bs: batch size
        :param n_in: input size
        :param sent_len: sentence length
        :param kernel_size: size of convolutional kernel
        :param n_filters: number of filters
        :param name: layer alias
        """
        self.bs = bs
        self.n_in = n_in
        self.sent_len = sent_len
        self.kernel_size = kernel_size
        self.n_filters = n_filters
        self.filter_shape = (self.n_filters, 1, self.kernel_size, self.n_in)
        self.image_shape = (self.bs, 1, self.sent_len, self.n_in)
        self.name = name
        self.pool_size = (self.sent_len - self.kernel_size + 1, 1)
        self.W = theano.shared(
            value=uniform(lb=-INIT_RANGE, ub=INIT_RANGE, size=(self.n_filters, 1, self.kernel_size, self.n_in)),
            name='%s_W' % self.name
        )
        self.b = theano.shared(value=zeros(size=self.n_filters), name='%s_b' % self.name)
        self.params = [self.W, self.b]

    def __str__(self):
        return "%s: CNN(%s, %s, kernel_size=%s)" % (self.name, self.n_in, self.n_filters, self.kernel_size)

    __repr__ = __str__

    def __call__(self, x):
        """

        :param x: input tensor, shape: (bs, seq_len, n_in)
        :return: (1) features after pooling; (2) generated feature maps
        """
        x = x.dimshuffle(0, 'x', 1, 2)
        conv_out = T.nnet.conv2d(input=x, filters=self.W, filter_shape=self.filter_shape, input_shape=self.image_shape)
        conv_out = T.nnet.relu(conv_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        # shape: (bs, n_filter, sent_len - kernel_size + 1)
        feature_maps = conv_out.flatten(3)
        # max pooling
        conv_out_pool = pool.pool_2d(input=conv_out, ws=self.pool_size, mode='max', ignore_border=True).flatten(2)
        return conv_out_pool, feature_maps


class Dropout:
    def __init__(self, p):
        self.p = p
        self.retain_prob = 1 - p

    def __str__(self):
        return "Dropout(%s)" % (1.0 - self.retain_prob)

    __repr__ = __str__

    def __call__(self, x):
        """

        :param x: input tensor
        :return:
        """
        rng = np.random.RandomState(1344)
        srng = RandomStreams(rng.randint(999999))
        mask = srng.binomial(size=x.shape, n=1, p=self.retain_prob, dtype='float32')
        scaling_factor = 1.0 / (1.0 - self.p)
        return x * mask


class CPT_AS:
    # Context-Preserving Transformation with Adaptive-Scaling
    def __init__(self, bs, sent_len, n_in, n_out, name):
        self.bs = bs
        self.sent_len = sent_len
        self.n_in = n_in
        self.n_out = n_out
        self.name = name
        self.fc_gate = Linear(n_in=self.n_in, n_out=self.n_out, name="Gate")
        self.fc_trans = Linear(n_in=2*self.n_in, n_out=self.n_out, name="Trans")
        # for model with highway transformation
        self.layers = [self.fc_gate, self.fc_trans]
        # for model without highway transformation
        #self.layers = [self.fc_trans]
        self.params = []
        for layer in self.layers:
            self.params.extend(layer.params)

    def __str__(self):
        des_str = 'CPT(%s, %s)' % (self.n_in, self.n_out)
        for layer in self.layers:
            des_str += ', %s' % layer
        return des_str

    __repr__ = __str__

    def __call__(self, x, xt):
        """

        :param x: input sentence, shape: (bs, sent_len, n_in)
        :param xt: input target, shape: (bs, target_len, n_in)
        :return:
        """
        trans_gate = T.nnet.hard_sigmoid(self.fc_gate(x, bs=self.bs))
        # (max_len, bs, n_in)
        x_ = x.dimshuffle(1, 0, 2)
        # (bs, n_in, target_len)
        xt_ = xt.dimshuffle(0, 2, 1)
        x_new = []
        for i in range(self.sent_len):
            # (bs, n_in)
            xi = x_[i]
            # shape: (bs, sent_len)
            alphai = T.nnet.softmax(T.batched_dot(xt, xi.dimshuffle(0, 1, 'x')).flatten(2))
            ti = T.batched_dot(xt_, alphai.dimshuffle(0, 1, 'x')).flatten(2)
            xi_new = T.tanh(self.fc_trans(x=T.concatenate([xi, ti], axis=1)))
            x_new.append(xi_new)
        x_new = T.stack(x_new, axis=0).dimshuffle(1, 0, 2)
        return trans_gate * x_new + (1.0 - trans_gate) * x


class CPT_LF:
    # Context-Preserving Transformation with Lossless-Forwarding
    def __init__(self, bs, sent_len, n_in, n_out, name):
        self.bs = bs
        self.sent_len = sent_len
        self.n_in = n_in
        self.n_out = n_out
        self.name = name
        self.fc_trans = Linear(n_in=2*self.n_in, n_out=self.n_out, name="Trans")
        self.layers = [self.fc_trans]
        self.params = []
        for layer in self.layers:
            self.params.extend(layer.params)

    def __str__(self):
        des_str = 'CPT(%s, %s)' % (self.n_in, self.n_out)
        for layer in self.layers:
            des_str += ', %s' % layer
        return des_str

    __repr__ = __str__

    def __call__(self, x, xt):
        """

        :param x: input sentence, shape: (bs, sent_len, n_in)
        :param xt: input target, shape: (bs, target_len, n_in)
        :return:
        """
        # (max_len, bs, n_in)
        x_ = x.dimshuffle(1, 0, 2)
        # (bs, n_in, target_len)
        xt_ = xt.dimshuffle(0, 2, 1)
        x_new = []
        for i in range(self.sent_len):
            # (bs, n_in)
            xi = x_[i]
            # shape: (bs, sent_len)
            alphai = T.nnet.softmax(T.batched_dot(xt, xi.dimshuffle(0, 1, 'x')).flatten(2))
            ti = T.batched_dot(xt_, alphai.dimshuffle(0, 1, 'x')).flatten(2)
            xi_new = T.nnet.relu(self.fc_trans(x=T.concatenate([xi, ti], axis=1)))
            x_new.append(xi_new)
        x_new = T.stack(x_new, axis=0).dimshuffle(1, 0, 2)
        return x_new + x


class TNet:
    """
    Transformation Networks for Target-Oriented Sentiment Analysis
    """
    def __init__(self, args):
        self.ds_name = args.ds_name # dataset name
        self.connection_type = args.connection_type # connection type AS/LF
        self.dropout_rate = args.dropout_rate # dropout rate 0.3
        self.lr = args.lr
        self.model_name = 'TNet'
        self.model_dir = './models_no_split_lstm_after_cpt_first_and_last_hidden_concat/'+self.ds_name+'/'
        self.saved_model_name = self.connection_type+'_'+str(args.n_epoch)+'_'+str(self.lr)
        if args.ds_name != '14semeval_rest':
            seed = 14890
        else:
            seed = 11456
        print("Use seed %s..." % seed)
        np.random.seed(seed)
        self.bs = args.bs
        self.n_in = args.dim_w
        self.n_rnn_out = args.dim_h
        self.n_rnn_out_after_cpt = 25
        self.kernels = args.kernels
        self.embedding_weights = args.embeddings
        self.n_filters = args.n_filter
        self.n_y = args.dim_y
        self.sent_len = args.sent_len
        self.target_len = args.target_len
        #assert len(self.kernels) == 1

        # model component for ASTN
        self.Words = theano.shared(value=np.array(self.embedding_weights, 'float32'), name="embedding")
        self.Dropout_ctx = Dropout(p=0.3)
        self.Dropout_tgt = Dropout(p=0.3)
        self.Dropout = Dropout(p=self.dropout_rate)
        self.LSTM_ctx = LSTM(bs=self.bs, n_in=self.n_in, n_out=self.n_rnn_out, name="CTX_LSTM")
        self.LSTM_tgt = LSTM(bs=self.bs, n_in=self.n_in, n_out=self.n_rnn_out, name="TGT_LSTM")
        self.LSTM_cpt = LSTM(bs=self.bs, n_in=2*self.n_rnn_out, n_out=self.n_rnn_out_after_cpt, name="CPT_LSTM")
        if self.connection_type == 'AS':
            self.CPT = CPT_AS(bs=self.bs, sent_len=self.sent_len, n_in=2*self.n_rnn_out, n_out=2*self.n_rnn_out, name="CPT")
        else:
            self.CPT = CPT_LF(bs=self.bs, sent_len=self.sent_len, n_in=2 * self.n_rnn_out, n_out=2 * self.n_rnn_out, name="CPT")

        # convolutional layers; actually, we just use one kernel size in our model
        self.Conv_layers = []
        for i in range(len(self.kernels)):
            self.Conv_layers.append(CNN(bs=self.bs, n_in=2*self.n_rnn_out, sent_len=self.sent_len,
                                        kernel_size=self.kernels[i], n_filters=self.n_filters, name='Conv2D_%s' % i))
        self.fc_gate_after_cpt = Linear(n_in=4*self.n_rnn_out_after_cpt, n_out=4*self.n_rnn_out_after_cpt, name="Gate_after_cpt")
        self.FC = Linear(n_in=self.n_filters*len(self.kernels), n_out=self.n_y, name="LAST_FC")
        # parameters for full model
        self.layers = [self.LSTM_ctx, self.LSTM_tgt, self.CPT, self.FC]
        self.layers.extend(self.Conv_layers)

        self.params = []
        for layer in self.layers:
            self.params.extend(layer.params)
        print(self.params)
        self.build_model()
        self.make_function()

    def __str__(self):
        strs = []
        for layer in self.layers:
            strs.append(str(layer))
        return ', '.join(strs)

    __repr__ = __str__

    def build_model(self):
        """
        build the computational graph of ASTN
        :return:
        """
        self.x = T.imatrix('wids')
        self.xt = T.imatrix('wids_target')
        self.y = T.ivector('label')
        self.pw = T.fmatrix("position_weight")
        self.is_train = T.iscalar("is_training")
        input = self.Words[T.cast(self.x.flatten(), 'int32')].reshape((self.bs, self.sent_len, self.n_in))
        input_target = self.Words[T.cast(self.xt.flatten(), 'int32')].reshape((self.bs, self.target_len, self.n_in))

        input = T.switch(T.eq(self.is_train, np.int32(1)), self.Dropout_ctx(input), input * (1 - self.dropout_rate))
        input_target = T.switch(T.eq(self.is_train, np.int32(1)), self.Dropout_tgt(input_target), input_target * (1 - self.dropout_rate))

        # model component for TNet
        rnn_input = input
        rnn_input_reverse = reverse_tensor(tensor=rnn_input)

        rnn_input_target = input_target
        rnn_input_target_reverse = reverse_tensor(tensor=rnn_input_target)

        H0_forward = self.LSTM_ctx(x=rnn_input)
        Ht_forward = self.LSTM_tgt(x=rnn_input_target)
        H0_backward = reverse_tensor(tensor=self.LSTM_ctx(x=rnn_input_reverse))
        Ht_backward = reverse_tensor(tensor=self.LSTM_tgt(x=rnn_input_target_reverse))
        H0 = T.concatenate([H0_forward, H0_backward], axis=2)
        Ht = T.concatenate([Ht_forward, Ht_backward], axis=2)
        H1 = self.CPT(H0, Ht)
        if self.pw is not None:
           H1 = H1 * self.pw.dimshuffle(0, 1, 'x')
        H2 = self.CPT(H1, Ht)
        if self.pw is not None:
            H2 = H2 * self.pw.dimshuffle(0, 1, 'x')
        """
        H3 = self.CPT(H2, Ht)
        if self.pw is not None:
            H3 = H3 * self.pw.dimshuffle(0, 1, 'x')
        H4 = self.CPT(H3, Ht)
        if self.pw is not None:
            H4 = H4 * self.pw.dimshuffle(0, 1, 'x')
        H5 = self.CPT(H4, Ht)
        if self.pw is not None:
            H5 = H5 * self.pw.dimshuffle(0, 1, 'x')
        """

        # lstm layer after CPT
        # batch_size*max_sequence_length*2*hidden_size_after_cpt
        H2_after_lstm_forward = self.LSTM_cpt(x=H2)
        H2_after_lstm_backward = reverse_tensor(tensor=self.LSTM_cpt(x=reverse_tensor(tensor=H2)))
        H2_after_lstm = T.concatenate([H2_after_lstm_forward, H2_after_lstm_backward], axis=2)

        #concat first bi-lstm hidden and last bi-lstm hidden
        H2_after_lstm_shape = T.shape(H2_after_lstm)
        H2_after_lstm_first_hidden = H2_after_lstm[:, 0:1, :]
        H2_after_lstm_last_hidden = H2_after_lstm[:, H2_after_lstm_shape[1]-1:H2_after_lstm_shape[1], :]
        H2_after_lstm_concat = T.concatenate([H2_after_lstm_first_hidden, H2_after_lstm_last_hidden], axis = -1)
        H2_after_lstm_concat_reshape = T.reshape(H2_after_lstm_concat, [T.shape(H2_after_lstm_concat)[0], T.shape(H2_after_lstm_concat)[1]*T.shape(H2_after_lstm_concat)[2]])
        
        feat_and_feat_maps = [conv(H2) for conv in self.Conv_layers]
        feat = [ele[0] for ele in feat_and_feat_maps]
        self.feature_maps = T.concatenate([ele[1] for ele in feat_and_feat_maps], axis=2)
        self.feature_maps_argmax = T.squeeze(T.argmax(self.feature_maps.dimshuffle(0, 2, 1), axis = 1))
        feat = T.concatenate(feat, axis=1)

        # concat cnn o/p and lstm after cpt o/p
        trans_gate_after_cpt = T.nnet.hard_sigmoid(self.fc_gate_after_cpt(H2_after_lstm_concat_reshape))
        mod_hidden = trans_gate_after_cpt * feat + (1.0 - trans_gate_after_cpt) * H2_after_lstm_concat_reshape
        #concat_hidden = T.concatenate([H2_after_lstm_concat_reshape, feat], axis = 1)        

        # we do not use the self-implemented Dropout class
        feat_dropout = T.switch(T.eq(self.is_train, np.int32(1)), self.Dropout(mod_hidden), mod_hidden * (1 - self.dropout_rate))
        # shape: (bs, n_y)
        self.p_y_x = T.nnet.softmax(self.FC(feat_dropout))
        # self.p_y_x = self.FC(feat_dropout)
        self.loss = T.nnet.categorical_crossentropy(coding_dist=self.p_y_x, true_dist=self.y).mean()
        self.pred_y = T.argmax(self.p_y_x, axis=1)

    def save_model(self):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        params_list = [self.Words.get_value()]
        for param in self.params:
            params_list.append(param.get_value())
        model_file = self.model_dir+self.model_name+'_'+self.saved_model_name
        f = open(model_file, 'wb')
        pickle.dump(params_list, f, protocol=pickle.HIGHEST_PROTOCOL)
        return model_file

    def load_model(self, model_file):
        params_list = pickle.load(model_file)
        self.Words.set_value(params_list[0])
        for param, param_value in zip(self.params, params_list[1:]):
            param.set_value(param_value)

    def make_function(self):
        """
        compile theano function
        :return:
        """

        print("Use adam...")
        self.updates = adam(cost=self.loss, params=self.params, lr=self.lr)
        model_inputs = [self.x, self.xt, self.y, self.pw, self.is_train]
        model_outputs = [self.pred_y, self.y, self.loss, self.feature_maps_argmax]
        self.train = theano.function(
            inputs=model_inputs,
            outputs=model_outputs,
            updates=self.updates,
            #mode='DebugMode'
        )
        self.test = theano.function(
            inputs=model_inputs,
            outputs=model_outputs
        )


