# -*- coding: utf-8 -*-
import argparse
import math
import time
import os
from layer_no_split_lstm_after_cpt_first_and_last_hidden_reshape_concat_GraphSAGE import TNet
from utils_no_split_GraphSAGE import *
from nn_utils_sentence_GraphSAGE import *
from evals import *
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='TNet settings')
    parser.add_argument("-ds_name", type=str, default="14semeval_rest", help="dataset name")
    parser.add_argument("-n_filter", type=int, default=50, help="number of convolutional filters")
    parser.add_argument("-bs", type=int, default=64, help="batch size")
    parser.add_argument("-dim_w", type=int, default=300, help="dimension of word embeddings")
    parser.add_argument("-dim_e", type=int, default=25, help="dimension of episode")
    parser.add_argument("-dim_func", type=int, default=10, help="dimension of functional embeddings")
    parser.add_argument("-dim_p", type=int, default=30, help="dimension of position embeddings")
    parser.add_argument("-dropout_rate", type=float, default=0.3, help="dropout rate for sentimental features")
    parser.add_argument("-dim_h", type=int, default=50, help="dimension of hidden state")
    parser.add_argument("-rnn_type", type=str, default='LSTM', help="type of recurrent unit")
    parser.add_argument("-n_epoch", type=int, default=100, help="number of training epoch")
    parser.add_argument("-dim_y", type=int, default=3, help="dimension of label space")
    parser.add_argument("-lr", type=float, default=0.002, help="learning rate")
    parser.add_argument("-lambda", type=float, default=1e-4, help="L2 coefficient")
    parser.add_argument("-did", type=int, default=2, help="gpu device id")
    parser.add_argument("-connection_type", type=str, default="AS", help="connection type, only AS and LF are valid")

    args = parser.parse_args()

    args.kernels = [3]
    
    dataset, embeddings, embeddings_func, n_train, n_test = build_dataset(args.connection_type, ds_name=args.ds_name, bs=args.bs,
                                                                          dim_w=args.dim_w, dim_func=args.dim_func)
    
    
    # update the size of the used word embeddings
    args.dim_w = len(embeddings[1])
    print(args)
    args.embeddings = embeddings
    # args.embeddings_func = embeddings_func
    args.sent_len = len(dataset[0][0]['wids'])
    args.target_len = len(dataset[0][0]['tids'])
    print("Maximum sent length:", args.sent_len)
    print("Maximum target length:", args.target_len)
    print("length of padded training set:", len(dataset[0]))

    n_train_batches = math.ceil(n_train / args.bs)
    # print("n batches of training set:", n_train_batches)
    n_test_batches = math.ceil(n_test / args.bs)
    train_set, test_set = dataset

    cur_model_name = 'TNet-%s' % args.connection_type
    print("Current model name:", cur_model_name)

    model = TNet(args=args)
    print(model)
 
    result_strings = []

    train_loss_all_epoch = []
    test_loss_all_epoch = []
    train_acc_all_epoch = []
    test_acc_all_epoch = []

    max_acc_test = 0.0
    max_f1_test = 0.0

    acc_test_corr_max_f1_test = 0.0
    f1_test_corr_max_acc_test = 0.0

    train_context_word_sequence_best_epoch = []
    train_sentences_best_epoch = []
    train_y_pred_best_epoch = []
    train_y_gold_best_epoch = []

    test_context_word_sequence_best_epoch = []
    test_sentences_best_epoch = []
    test_y_pred_best_epoch = []
    test_y_gold_best_epoch = []

    if args.ds_name != '14semeval_rest':
        seed = 14890
    else:
        seed = 11456

    np.random.seed(seed)


    for i in range(1, args.n_epoch + 1):
        # ---------------training----------------
        print("In epoch %s/%s:" % (i, args.n_epoch))
        np.random.shuffle(train_set)

        train_y_pred, train_y_gold = [], []
        train_losses = []
        train_sentences = []
       
        train_context_word_sequence_epoch = []

        for j in range(n_train_batches):

            train_x, train_xt, train_y, train_pw, sentence, train_graph_embedding = get_batch_input(max_sequence_length=args.sent_len, dataset=train_set, bs=args.bs, idx=j)

            y_pred, y_gold, loss, feature_maps_argmax = model.train(train_x, train_xt, train_y, train_pw, train_graph_embedding, np.int32(1))

            train_losses.append(loss)
            train_y_pred.extend(y_pred)
            train_y_gold.extend(y_gold)
            train_sentences.extend(sentence)

            train_context_word_sequence_iteration = []

            for index in range(len(feature_maps_argmax)):
                train_context_word_sequence = []
                words = []
                words = sentence[index].strip().split()
                for item in feature_maps_argmax[index]:
                    if item<len(words):
                        train_context_word_sequence.append(words[item]+' ')
                        if item+1<len(words):
                            train_context_word_sequence.append(words[item+1]+' ')
                        else: 
                            train_context_word_sequence.append('zero_padded_word')
                        if item+2<len(words):
                            train_context_word_sequence.append(words[item+2]+' ')
                        else: 
                            train_context_word_sequence.append('zero_padded_word')
                    else:
                        train_context_word_sequence.append('zero_padded_word')
                        train_context_word_sequence.append('zero_padded_word')
                        train_context_word_sequence.append('zero_padded_word')

                train_context_word_sequence_iteration.append(train_context_word_sequence)

            train_context_word_sequence_epoch.extend(train_context_word_sequence_iteration)

        acc_train, f_train, _, _ = evaluate(pred=train_y_pred, gold=train_y_gold)
        print("\tnormalized train loss: %.4f, train acc: %.4f, train f1: %.4f" % (sum(train_losses)/n_train_batches, acc_train, f_train))
        train_loss_all_epoch.append(sum(train_losses)/n_train_batches)
        train_acc_all_epoch.append(acc_train)
        result_strings.append("In Epoch %s: train accuracy: %.2f, train macro-f1: %.2f\n" % (i, acc_train * 100, f_train * 100))

        # ---------------prediction----------------

        test_y_pred, test_y_gold = [], []
        test_losses = []
        test_sentences = []
       
        test_context_word_sequence_epoch = []

        for j in range(n_test_batches):

            test_x, test_xt, test_y, test_pw, sentence, test_graph_embedding = get_batch_input(max_sequence_length=args.sent_len, dataset=test_set, bs=args.bs, idx=j)

            y_pred, y_gold, loss, feature_maps_argmax = model.test(test_x, test_xt, test_y, test_pw, test_graph_embedding, np.int32(0))

            test_losses.append(loss)
            test_y_pred.extend(y_pred)
            test_y_gold.extend(y_gold)
            test_sentences.extend(sentence)

            test_context_word_sequence_iteration = []

            for index in range(len(feature_maps_argmax)):
                test_context_word_sequence = []
                words = []
                words = sentence[index].strip().split()
                for item in feature_maps_argmax[index]:
                    if item<len(words):
                        test_context_word_sequence.append(words[item]+' ')
                        if item+1<len(words):
                            test_context_word_sequence.append(words[item+1]+' ')
                        else: 
                            test_context_word_sequence.append('zero_padded_word')
                        if item+2<len(words):
                            test_context_word_sequence.append(words[item+2]+' ')
                        else: 
                            test_context_word_sequence.append('zero_padded_word')
                    else:
                        test_context_word_sequence.append('zero_padded_word')
                        test_context_word_sequence.append('zero_padded_word')
                        test_context_word_sequence.append('zero_padded_word')

                test_context_word_sequence_iteration.append(test_context_word_sequence)

            test_context_word_sequence_epoch.extend(test_context_word_sequence_iteration)

        acc_test, f_test, _, _ = evaluate(pred=test_y_pred, gold=test_y_gold)
        test_loss_all_epoch.append(sum(test_losses)/n_test_batches)
        test_acc_all_epoch.append(acc_test)

        print("\tnormalized test loss: %.4f, test acc: %.4f, test f1: %.4f" % (sum(test_losses)/n_test_batches, acc_test, f_test))
        result_strings.append("In Epoch %s: test accuracy: %.2f, test macro-f1: %.2f\n" % (i, acc_test * 100, f_test * 100))

        if acc_test>max_acc_test:
            max_acc_test = acc_test
            train_context_word_sequence_best_epoch = []
            train_sentences_best_epoch = []
            train_y_pred_best_epoch =[]
            train_y_gold_best_epoch = []
            train_context_word_sequence_best_epoch = train_context_word_sequence_epoch
            train_sentences_best_epoch = train_sentences
            train_y_pred_best_epoch = train_y_pred
            train_y_gold_best_epoch = train_y_gold

            test_context_word_sequence_best_epoch = []
            test_sentences_best_epoch = []
            test_y_pred_best_epoch =[]
            test_y_gold_best_epoch = []
            test_context_word_sequence_best_epoch = test_context_word_sequence_epoch
            test_sentences_best_epoch = test_sentences
            test_y_pred_best_epoch = test_y_pred
            test_y_gold_best_epoch = test_y_gold

            f1_test_corr_max_acc_test = f_test

        if f_test>max_f1_test:
            max_f1_test = f_test
            acc_test_corr_max_f1_test = acc_test
            

    result_strings.append("Maximum test accuracy: %.2f, corresponding test macro-f1: %.2f\n" % (max_acc_test * 100, f1_test_corr_max_acc_test * 100))
    result_strings.append("Maximum test macro-f1: %.2f, corresponding test test macro-f1: %.2f\n" % (max_f1_test * 100, acc_test_corr_max_f1_test * 100))

    plt.plot(range(1, args.n_epoch + 1), train_loss_all_epoch, label='train loss')
    plt.plot(range(1, args.n_epoch + 1), test_loss_all_epoch, label='test loss')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig('loss_trend_'+cur_model_name+'_'+args.ds_name+'_'+str(args.n_epoch)+'_'+str(args.lr)+'_no_split_lstm_after_cpt_first_and_last_hidden_reshape_concat_GraphSAGE.png')
    plt.clf()

    plt.plot(range(1, args.n_epoch + 1), train_acc_all_epoch, label='train_acc')
    plt.plot(range(1, args.n_epoch + 1), test_acc_all_epoch, label='test_acc')
    plt.xlabel('epochs')
    plt.ylabel('acc')
    plt.legend()
    plt.savefig('acc_trend_'+cur_model_name+'_'+args.ds_name+'_'+str(args.n_epoch)+'_'+str(args.lr)+'_no_split_lstm_after_cpt_first_and_last_hidden_reshape_concat_GraphSAGE.png')
    plt.clf()

    result_logs = ['-------------------------------------------------------\n']
    params_string = str(args)
    result_logs.append("Running model: %s\n" % cur_model_name)
    result_logs.append(params_string + "\n")
    result_logs.extend(result_strings)
    result_logs.append('-------------------------------------------------------\n')
    if not os.path.exists('./log_no_split_lstm_after_cpt_first_and_last_hidden_reshape_concat_GraphSAGE'):
        os.makedirs('log_no_split_lstm_after_cpt_first_and_last_hidden_reshape_concat_GraphSAGE')
    with open("./log_no_split_lstm_after_cpt_first_and_last_hidden_reshape_concat_GraphSAGE/%s_%s_%s_%s_%s.txt" % (cur_model_name, args.ds_name, str(args.n_epoch), str(args.lr), str(args.dim_h)), 'w') as fp:
        fp.writelines(result_logs)

    with open("./log_no_split_lstm_after_cpt_first_and_last_hidden_reshape_concat_GraphSAGE/%s_%s_%s_%s_%s_visualize_train.txt" % (cur_model_name, args.ds_name, str(args.n_epoch), str(args.lr), str(args.dim_h)), 'w') as fp_train:
        
        for i in range(len(train_sentences_best_epoch)):
            fp_train.write('Sentence: '+train_sentences_best_epoch[i]+'\n')
            fp_train.write('Correct label: '+str(train_y_gold_best_epoch[i])+'\n')
            fp_train.write('Predicted label: '+str(train_y_pred_best_epoch[i])+'\n')
            fp_train.write('Maxpooled context word sequences(3 consecutive words): ')
            j = 0
            while j<len(train_context_word_sequence_best_epoch[i]):
                fp_train.write(train_context_word_sequence_best_epoch[i][j]+' '+train_context_word_sequence_best_epoch[i][j+1]+' '+train_context_word_sequence_best_epoch[i][j+2]+'  ;')
                j += 3

            fp_train.write('\n\n')

    with open("./log_no_split_lstm_after_cpt_first_and_last_hidden_reshape_concat_GraphSAGE/%s_%s_%s_%s_%s_visualize_train_wrong_prediction_only.txt" % (cur_model_name, args.ds_name, str(args.n_epoch), str(args.lr), str(args.dim_h)), 'w') as fp_train:
        
        for i in range(len(train_sentences_best_epoch)):
            if train_y_gold_best_epoch[i] != train_y_pred_best_epoch[i]:
                fp_train.write('Sentence: '+train_sentences_best_epoch[i]+'\n')
                fp_train.write('Correct label: '+str(train_y_gold_best_epoch[i])+'\n')
                fp_train.write('Predicted label: '+str(train_y_pred_best_epoch[i])+'\n')
                fp_train.write('Maxpooled context word sequences(3 consecutive words): ')
                j = 0
                while j<len(train_context_word_sequence_best_epoch[i]):
                    fp_train.write(train_context_word_sequence_best_epoch[i][j]+' '+train_context_word_sequence_best_epoch[i][j+1]+' '+train_context_word_sequence_best_epoch[i][j+2]+'  ;')
                    j += 3

                fp_train.write('\n\n')



    with open("./log_no_split_lstm_after_cpt_first_and_last_hidden_reshape_concat_GraphSAGE/%s_%s_%s_%s_%s_visualize_test.txt" % (cur_model_name, args.ds_name, str(args.n_epoch), str(args.lr), str(args.dim_h)), 'w') as fp_test:
        
        for i in range(len(test_sentences)):
            fp_test.write('Sentence: '+test_sentences[i]+'\n')
            fp_test.write('Correct label: '+str(test_y_gold[i])+'\n')
            fp_test.write('Predicted label: '+str(test_y_pred[i])+'\n')
            fp_test.write('Maxpooled context word sequences(3 consecutive words): ')
            j = 0
            while j<len(test_context_word_sequence_epoch[i]):
                fp_test.write(test_context_word_sequence_epoch[i][j]+' '+test_context_word_sequence_epoch[i][j+1]+' '+test_context_word_sequence_epoch[i][j+2]+'  ;')
                
                j += 3

            fp_test.write('\n\n')

    with open("./log_no_split_lstm_after_cpt_first_and_last_hidden_reshape_concat_GraphSAGE/%s_%s_%s_%s_%s_visualize_test_wrong_prediction_only.txt" % (cur_model_name, args.ds_name, str(args.n_epoch), str(args.lr), str(args.dim_h)), 'w') as fp_test:
        
        for i in range(len(test_sentences)):
            if test_y_gold[i] != test_y_pred[i]:
                fp_test.write('Sentence: '+test_sentences[i]+'\n')
                fp_test.write('Correct label: '+str(test_y_gold[i])+'\n')
                fp_test.write('Predicted label: '+str(test_y_pred[i])+'\n')
                fp_test.write('Maxpooled context word sequences(3 consecutive words): ')
                j = 0
                while j<len(test_context_word_sequence_epoch[i]):
                    fp_test.write(test_context_word_sequence_epoch[i][j]+' '+test_context_word_sequence_epoch[i][j+1]+' '+test_context_word_sequence_epoch[i][j+2]+'  ;')
                
                    j += 3

                fp_test.write('\n\n')

    correct_label = np.array(test_y_gold).astype(int)
    predicted_label = np.array(test_y_pred).astype(int)
    np.save("./log_no_split_lstm_after_cpt_first_and_last_hidden_reshape_concat_GraphSAGE/%s_%s_%s_%s_%s_correct_test_label.npy" % (cur_model_name, args.ds_name, str(args.n_epoch), str(args.lr), str(args.dim_h)), correct_label)
    np.save("./log_no_split_lstm_after_cpt_first_and_last_hidden_reshape_concat_GraphSAGE/%s_%s_%s_%s_%s_predicted_test_label_original.npy" % (cur_model_name, args.ds_name, str(args.n_epoch), str(args.lr), str(args.dim_h)), predicted_label)
                
                   

    

        
      

        

        
