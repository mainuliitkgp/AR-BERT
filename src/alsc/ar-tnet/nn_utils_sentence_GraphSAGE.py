# -*- coding: utf-8 -*-
import theano
import theano.tensor as T
import numpy as np
import pandas as pd
from collections import OrderedDict

INIT_RANGE = 0.01


def uniform(lb, ub, size):
    """

    :param lb: lower bound
    :param ub: upper bound
    :param size: tensor shape
    :return:
    """
    return np.array(np.random.uniform(low=lb, high=ub, size=size), dtype='float32')


def glorot_uniform(size):
    """
    glorot uniform initializer
    """
    if len(size) == 1:
        values = np.zeros_like(size)
    else:
        scale = np.sqrt(6.0 / np.sum(size))
        values = np.random.uniform(low=-scale, high=scale, size=size)
    return values.astype('float32')


def orthogonal(size):
    """
    equivalent to orthogonal_init but return numpy array
    """
    if len(size) == 1:
        values = np.zeros_like(size)
    else:
        a = np.random.normal(loc=0.0, scale=1.0, size=size)
        # reconstruction based on reduced SVD
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == size else v
        q = q.reshape(size)
        values = q
    return values.astype("float32")


def zeros(size):
    """
    generate zero vector / tensor
    :param size: size
    :return:
    """
    return np.array(np.zeros(shape=size), dtype='float32')


def ones(size):
    """
    generate one vector / tensor
    :param size:
    :return:
    """
    return np.array(np.zeros(shape=size), dtype='float32')


def lstm_init(n_in, n_out, component="LSTM"):
    """

    :param n_in: input size
    :param n_out: hidden size
    :param component: component name
    :return:
    """
    if True:
        print("Initialize LSTM weights from uniform distribution...")
        W_values = np.concatenate([uniform(lb=-INIT_RANGE, ub=INIT_RANGE, size=(n_in, n_out)),
                                   uniform(lb=-INIT_RANGE, ub=INIT_RANGE, size=(n_in, n_out)),
                                   uniform(lb=-INIT_RANGE, ub=INIT_RANGE, size=(n_in, n_out)),
                                   uniform(lb=-INIT_RANGE, ub=INIT_RANGE, size=(n_in, n_out))], axis=1)

        U_values = np.concatenate([uniform(lb=-INIT_RANGE, ub=INIT_RANGE, size=(n_out, n_out)),
                                   uniform(lb=-INIT_RANGE, ub=INIT_RANGE, size=(n_out, n_out)),
                                   uniform(lb=-INIT_RANGE, ub=INIT_RANGE, size=(n_out, n_out)),
                                   uniform(lb=-INIT_RANGE, ub=INIT_RANGE, size=(n_out, n_out))], axis=1)
        b_values = np.concatenate([zeros(n_out),
                                   zeros(n_out),
                                   zeros(n_out),
                                   zeros(n_out)])
    else:
        print("Initialize LSTM weights from glorot uniform + glorot uniform...")
        W_values = np.concatenate([glorot_uniform(size=(n_in, n_out)),
                                  glorot_uniform(size=(n_in, n_out)),
                                  glorot_uniform(size=(n_in, n_out)),
                                  glorot_uniform(size=(n_in, n_out))], axis=1)

        U_values = np.concatenate([glorot_uniform(size=(n_out, n_out)),
                                   glorot_uniform(size=(n_out, n_out)),
                                   glorot_uniform(size=(n_out, n_out)),
                                   glorot_uniform(size=(n_out, n_out))], axis=1)
        b_values = np.concatenate([zeros(n_out),
                                   ones(n_out),  # set forget gate to 1.0
                                   zeros(n_out),
                                   zeros(n_out)])
    W = theano.shared(value=W_values, name='%s_W' % component)
    U = theano.shared(value=U_values, name='%s_U' % component)
    b = theano.shared(value=b_values, name='%s_b' % component)
    return W, U, b


def reverse_tensor(tensor):
    """

    :param tensor: input tensor, shape: (bs, seq_len, n_in)
    :return:
    """
    new_tensor = tensor.dimshuffle(1, 0, 2)
    return new_tensor[::-1].dimshuffle(1, 0, 2)


def adam(cost, params, max_norm=3.0, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    adam optimizer, default learning rate is 0.001
    """

    grads = T.grad(cost, params)
    t_prev = theano.shared(value=np.float32(0.0))
    updates = OrderedDict()

    t = t_prev + 1
    one = T.constant(1, dtype='float32')

    a_t = lr * T.sqrt(one - beta2 ** t) / (one - beta1 ** t)

    for p, g in zip(params, grads):
        value = p.get_value(borrow=True)
        m_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                               broadcastable=p.broadcastable)
        v_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                               broadcastable=p.broadcastable)

        m_t = beta1 * m_prev + (one - beta1) * g
        v_t = beta2 * v_prev + (one - beta2) * g ** 2

        step = a_t * m_t / (T.sqrt(v_t) + epsilon)

        stepped_p = p - step
        updates[m_prev] = m_t
        updates[v_prev] = v_t
        # add max norm constraint on the W_hidden
        # if p.name == 'W_hidden':
        #	col_norms = T.sqrt(T.sum(T.sqr(stepped_p), axis=0))
        #	desired_norms = T.clip(col_norms, 0, max_norm)
        #	scale = desired_norms / (1e-7 + col_norms)
        #	updates[p] = stepped_p * scale
        # else:
        #	updates[p] = stepped_p
        updates[p] = stepped_p
    updates[t_prev] = t
    return updates


def sgd_momentum(cost, params, lr=0.01, momentum=0.9):
    """
    sgd with momentum
    :param cost: training loss
    :param params: parameters
    :param lr: learning rate
    :param momentum: amount of momentum to apply
    :return:
    """
    grads = T.grad(cost, params)
    updates = OrderedDict()
    for p, g in zip(params, grads):
        updates[p] = p - lr * g
    for p in params:
        value = p.get_value(borrow=True)
        velocity = theano.shared(np.zeros(value.shape, dtype=value.dtype), broadcastable=p.broadcastable)
        x = momentum * velocity + updates[p]
        updates[velocity] = x - p
        updates[p] = x
    return updates

def get_batch_input(max_sequence_length, dataset, bs, idx):
    """

    :param dataset: dataset
    :param bs: batch size
    :param idx: batch index
    :return:
    """
    batch_input = dataset[idx*bs:(idx+1)*bs]
    #print(batch_input)
    batch_data = pd.DataFrame.from_dict(batch_input)
    #print(batch_data)
    #print(batch_data)
    target_fields = ['wids', 'tids', 'y', 'pw']
    batch_input_var = []
    for key in target_fields:
        #print("for %s data" % key)
        data = list(batch_data[key].values)
        #print(data[0])
        #print(data[5])
        if key == 'pw':
            # for position weights
            #batch_input_var.append(theano.shared(value=np.array(data, dtype='float32')))
            batch_input_var.append(np.array(data, dtype='float32'))
        else:
            #batch_input_var.append(theano.shared(value=np.array(data, dtype='int32')))
            batch_input_var.append(np.array(data, dtype='int32'))
    batch_input_var.append(list(batch_data['sent'].values))

    graph_data = list(batch_data['graphsage_embedding'].values)
    graph_data = np.array(graph_data, dtype='float32')
    graphsage_embedding = []

    for s in range(len(batch_input_var[0])):
        graph_embedding = []
        for i in range(max_sequence_length):
            graph_embedding.append(graph_data[s]) 
        graphsage_embedding.append(graph_embedding)

    graphsage_embedding = np.array(graphsage_embedding, dtype=np.float32)
    batch_input_var.append(graphsage_embedding)

    return batch_input_var
