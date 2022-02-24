import numpy as np
import pickle
import os
#from torch.autograd import Variable
#import torch
import string
import pandas as pd
from nltk import ngrams

def transform_twitter(mode='train'):
	"""
	"""
	file = './dataset/Twitter/original/%s.txt' % mode
	lines = ['dump']
	with open(file) as fp:
		lines.extend(fp.readlines())
	outputs = []
	for i in range(len(lines)):
		if not i:
			continue
		if i % 3 == 0:
			y = int(lines[i].strip())
			target_words = lines[i - 1].strip().split()
			label = 'xxx'
			if y == 1:
				label = '/p'
			elif y == 0:
				label = '/0'
			elif y == -1:
				label = '/n'
			else:
				raise Exception("Invalid y!!")
			new_target_words = []
			for w in target_words:
				new_target_words.append(w + label)
			sent = lines[i - 2].strip()
			target = ' '.join(new_target_words)
			target_no_label = ' '.join(target_words)
			#assert '$T$' in sent
			n_target = sent.count('$T$')
			for i in range(n_target):
				#new_sent = sent.replace('$T$', target, 1)
				sent_copy = sent
				for j in range(i):
					sent_copy = sent_copy.replace('$T$', target_no_label, 1)
				sent_copy = sent_copy.replace('$T$', target, 1)
				# replace the rest
				sent_copy = sent_copy.replace('$T$', target_no_label)
				outputs.append('%s\n' % sent_copy)
				break
	with open('./dataset/Twitter/%s.txt' % mode, 'w+') as fp:
		fp.writelines(outputs)

def pad_dataset(dataset, bs):
	"""
	"""
	n_records = len(dataset)
	n_padded = bs - n_records % bs
	new_dataset = [t for t in dataset]
	new_dataset.extend(dataset[:n_padded])
	return new_dataset

def pad_seq(dataset, field, max_len, symbol):
	"""
	pad sequence to max_len with symbol
	"""
	n_records = len(dataset)
	for i in range(n_records):
		assert isinstance(dataset[i][field], list)
		while len(dataset[i][field]) < max_len:
			dataset[i][field].append(symbol)
	return dataset

def shuffle_dataset(dataset):
	"""
	"""
	np.random.shuffle(dataset)
	return dataset

def read(path):
	"""
	"""
	dataset = []
	sid = 0
	with open(path) as fp:
		for line in fp:
			record = {}
			tokens = line.strip().split()
			words, target_words = [], []
			d = []
			find_label = False
			for t in tokens:
				if '/p' in t or '/n' in t or '/0' in t:
					# negative: 0, positive: 1, neutral: 2
					# note, this part should be consistent with evals part
					end = 'xx'
					y = 0
					if '/p' in t:
						end = '/p'
						y = 1
					elif '/n' in t:
						end = '/n'
						y = 0
					elif '/0' in t:
						end = '/0'
						y = 2
					words.append(t.strip(end))
					target_words.append(t.strip(end))
					if not find_label:
						find_label = True
						#ys.append(y)
						record['y'] = y
						left_most = right_most = tokens.index(t)
					else:
						right_most += 1
				else:
					words.append(t)
			for pos in range(len(tokens)):
				if pos < left_most:
					d.append(right_most - pos)
				else:
					d.append(pos - left_most)
			record['sent'] = line.strip()
			record['words'] = words.copy()
			record['twords'] = target_words.copy()  # target words
			#record['twords'] = [target_words[-1]]   # treat the last word as head word
			record['wc'] = len(words)  # word count
			record['wct'] = len(record['twords'])  # target word count
			record['dist'] = d.copy()  # relative distance
			record['sid'] = sid
			record['beg'] = left_most
			record['end'] = right_most + 1
			# note: if aspect is single word, then aspect
			sid += 1
			dataset.append(record)
	return dataset

def load_data(connection_type, ds_name):
        """
        """
        train_file = './dataset/%s/train.txt' % ds_name
        test_file = './dataset/%s/test.txt' % ds_name
        train_set = read(path=train_file)
        test_set = read(path=test_file)

        train_wc = [t['wc'] for t in train_set]
        test_wc = [t['wc'] for t in test_set]
        max_len = max(train_wc) if max(train_wc) > max(test_wc) else max(test_wc)

        train_t_wc = [t['wct'] for t in train_set]
        test_t_wc = [t['wct'] for t in test_set]
        max_len_target = max(train_t_wc) if max(train_t_wc) > max(test_t_wc) else max(test_t_wc)

        #print("maximum length of target:", max_len_target)

        train_set = pad_seq(dataset=train_set, field='dist', max_len=max_len, symbol=-1)
        test_set = pad_seq(dataset=test_set, field='dist', max_len=max_len, symbol=-1)

        # calculate position weight
        train_set = calculate_position_weight(connection_type, dataset=train_set)
        test_set = calculate_position_weight(connection_type, dataset=test_set)

        vocab = build_vocab(dataset=train_set+test_set)

        vocab_func = get_func_words()

        train_set = set_fid(dataset=train_set, vocab=vocab, vocab_func=vocab_func, max_len=max_len)
        test_set = set_fid(dataset=test_set, vocab=vocab, vocab_func=vocab_func, max_len=max_len)

        train_set = set_wid(dataset=train_set, vocab=vocab, max_len=max_len)
        test_set = set_wid(dataset=test_set, vocab=vocab, max_len=max_len)

        train_set = set_tid(dataset=train_set, vocab=vocab, max_len=max_len_target)
        test_set = set_tid(dataset=test_set, vocab=vocab, max_len=max_len_target)

        dataset = [train_set, test_set]

        return dataset, vocab

def process_target(target_words):
	"""
	"""
	
	target_words_filtered = []
	punct = string.punctuation
	for w in target_words:
		if w in punct or w == '':
			continue
		target_words_filtered.append(w)
	return '-'.join(target_words_filtered)

def read_multi(path):
	"""
	read data and place multiple aspects in single instances 
	each aspect phrase is regarded as a single term
	"""
	dataset = []
	
	sent2aspects = {}
	with open(path) as fp:
		for line in fp:
			tokens = line.strip().split()
			raw_sent, target_words = [], []
			words = []
			find_target = False
			for t in tokens:
				if '/p' in t or '/n' in t or '/0' in t: 
					# negative: 0, positive: 1, neutral: 2
					if '/p' in t:
						w = t.strip('/p')
						target_words.append(w)
						y = 1
					elif '/n' in t:
						w = t.strip('/n')
						target_words.append(w)
						y = 0
					elif '/0' in t:
						w = t.strip('/0')
						target_words.append(w)
						y = 2
					else:
						raise Exception("Invalid label")
				else:
					w = t
				raw_sent.append(w)
			sent = ' '.join(raw_sent)
			target_phrase = ' '.join(target_words)
			if sent not in sent2aspects:
				sent2aspects[sent] = [(target_phrase, y)]
			else:
				sent2aspects[sent].append((target_phrase, y))
	place_holder = '#ASP#'
	dataset = []
	for sent in sent2aspects:
		record = {}
		aspects = sent2aspects[sent]
		norm_sent = sent
		record['aspects'] = []
		record['ys'] = []
		for (target_phrase, y) in aspects:
			target_words = target_phrase.split()
			norm_target_phrase = process_target(target_words)
			norm_sent = norm_sent.replace(target_phrase, norm_target_phrase, 1)
			record['aspects'].append(norm_target_phrase)
			record['ys'].append(y)
		record['words'] = norm_sent.split()
		record['wc'] = len(record['words'])
		pass
	#with open(path) as fp:pass

def build_vocab(dataset):
	"""
	"""
	vocab = {}
	idx = 1  # start from 1
	n_records = len(dataset)

	for i in range(n_records):
		for w in dataset[i]['words']:
			if w not in vocab:
				vocab[w] = idx
				idx += 1

		for w in dataset[i]['twords']:
			if w not in vocab:
				vocab[w] = idx
				idx += 1
	return vocab

def set_wid(dataset, vocab, max_len):
	"""
	word to id
	"""
	n_records = len(dataset)
	for i in range(n_records):
		sent = dataset[i]['words']
		dataset[i]['wids'] = word2id(vocab, sent, max_len)
	return dataset

def set_fid(dataset, vocab, vocab_func, max_len):
	"""
	word to functional id
	"""
	n_records = len(dataset)
	for i in range(n_records):
		sent = dataset[i]['words']
		dataset[i]['fids'] = word2fid(vocab, vocab_func, sent, max_len)
	return dataset

def set_tid(dataset, vocab, max_len):
	"""
	target word to id
	"""
	n_records = len(dataset)
	for i in range(n_records):
		sent = dataset[i]['twords']
		dataset[i]['tids'] = word2id(vocab, sent, max_len)
	return dataset

def word2id(vocab, sent, max_len):
	"""
	mapping word to word id together with sequence padding
	"""
	wids = [vocab[w] for w in sent]
	while len(wids) < max_len:
		wids.append(0)
	return wids

def word2fid(vocab, vocab_func, sent, max_len):
	"""
	mapping word to function word id
	"""
	fids = []
	for w in sent:
		if w in vocab_func:
			fids.append(0)
		else:
			fids.append(vocab[w])
	while len(fids) < max_len:
		fids.append(0)
	return fids

def get_func_words():
	vocab_func = {}
	# auxiliary verbs, conjunctions, determiners, prepositions, pronouns, quantifiers
	with open('func_words.txt') as fp:
		for line in fp:
			w = line.strip()
			if w not in vocab_func:
				vocab_func[w] = 1
	# add punctuations
	for symbol in string.punctuation:
		vocab_func[symbol] = 1
	return vocab_func

def get_embedding(vocab, ds_name, dim_w):
        """
        """
        emb_file = './embeddings/glove_840B_300d.txt'   # path of the pre-trained word embeddings
        pkl = './embeddings/%s_840B.pkl' % ds_name    # word embedding file of the current dataset
        print("Load embeddings from %s or %s..." % (emb_file, pkl))
        n_emb = 0
        if not os.path.exists(pkl):
            embeddings = np.zeros((len(vocab)+1, dim_w), dtype='float32')
            with open(emb_file) as fp:
                for line in fp:
                    eles = line.strip().split()
                    w = eles[0]
                    #if embeddings.shape[1] != len(eles[1:]):
                    #	embeddings = np.zeros((len(vocab) + 1, len(eles[1:])), dtype='float32')
                    n_emb += 1
                    if w in vocab:
                        try:
                            embeddings[vocab[w]] = [float(v) for v in eles[1:]]
                        except ValueError:
                            #print(embeddings[vocab[w]])
                            pass
            print("Find %s word embeddings!!" % n_emb)
            pickle.dump(embeddings, open(pkl, 'wb'))
        else:
            embeddings = pickle.load(open(pkl, 'rb'))
        return embeddings

def build_dataset(connection_type, ds_name, bs, dim_w, dim_func):
        """
        """
        dataset, vocab = load_data(connection_type, ds_name=ds_name)
        n_train = len(dataset[0])
        n_test = len(dataset[1])
        embeddings = get_embedding(vocab, ds_name, dim_w)
        if ds_name != '14semeval_rest':
            seed = 14890
        else:
            seed = 11456

        np.random.seed(seed)

        for i in range(len(embeddings)):
            if i and np.count_nonzero(embeddings[i]) == 0:
                embeddings[i] = np.random.uniform(-0.25, 0.25, embeddings.shape[1])
        embeddings = np.array(embeddings, dtype='float32')
        train_set = pad_dataset(dataset=dataset[0], bs=bs)
        test_set = pad_dataset(dataset=dataset[1], bs=bs)
        embeddings_func = np.random.uniform(-0.25, 0.25, (len(vocab)+1, dim_func))
        embeddings_func[0] = np.zeros(dim_func)
        embeddings_func = np.array(embeddings_func, 'float32')
        return [train_set, test_set], embeddings, embeddings_func, n_train, n_test

def to_tensor(dataset):
	"""
	"""
	return [torch.IntTensor(data) for data in dataset]

def reverse_tensor_(tensor, bs, wc=None):
	"""
	tensor type: torch tensor or torch variable
	tensor shape: (bs, max_len, dim_w)
	wc shape: (bs)
	"""
	new_tensor = []
	for i in range(bs):
		#print(wc[i])
		if wc is not None:
			main = tensor[i, :wc.data[i]].clone()
			# flip the main tensor / variable
			idx = Variable(torch.LongTensor([i for i in range(main.size(0)-1, -1, -1)]).cuda())
			try:
				padded = tensor[i, wc.data[i]:].clone()
				new_tensor.append(torch.cat([main.index_select(0, idx), padded], dim=0))
			except ValueError:
				# result of slicing is an empty tensor
				new_tensor.append(main.index_select(0, idx))
		else:
			main = tensor[i].clone()
			idx = Variable(torch.LongTensor([i for i in range(main.size(0)-1, -1, -1)]).cuda())
			new_tensor.append(main.index_select(0, idx))
	return torch.stack(new_tensor)

def collect_errors(errors, ds_name):
	"""
	collect error cases and write back to the disk 
	"""
	file_path = './dataset/%s/test.txt' % ds_name
	sentences = []
	y_to_label = {0: 'neg', 1: 'pos', 2: 'neutral'}
	with open(file_path, 'r') as fp:
		for line in fp:
			sentences.append(line.strip())
	error_cases = []
	for sid in errors:
		pred, gold = errors[sid]
		sent = sentences[sid]
		error_cases.append('%s\tGOLD:%s\tPRED:%s\n' % (sent, y_to_label[gold], y_to_label[pred]))

	with open('./errors/%s.txt' % ds_name, 'w+') as fp:
		fp.writelines(error_cases)

def calculate_position_weight(connection_type, dataset):
        """
        calculate position weight
        """
        if connection_type == 'AS':
            tmax = 30
        elif connection_type == 'LF':
            tmax = 40
        ps = []
        n_tuples = len(dataset)
        for i in range(n_tuples):
            dataset[i]['pw'] = []
            weights = []
            for w in dataset[i]['dist']:
                if w == -1:
                    weights.append(0.0)
                elif w > tmax:
                    weights.append(0.0)
                else:
                    weights.append(1.0 - float(w) / tmax)
            #print(weights)
            #ps.append(weights)
            dataset[i]['pw'].extend(weights)
        return dataset

def output(predictions, ds_name, res=None):
	"""
	"""
	file_path = './dataset/%s/test.txt' % ds_name
	data = []
	y_to_label = {0: 'neg', 1: 'pos', 2: 'neutral'}
	with open(file_path, 'r') as fp:
		for line in fp:
			sent = line.strip()
			record = [sent]
			if '/p' in sent:
				label = 'pos'
			elif '/n' in sent:
				label = 'neg'
			else:
				label = 'neutral'
			record.append(label)
			data.append(record)
	outputs = []
	for (y_pred, r) in zip(predictions, data):
		label_pred = y_to_label[y_pred]
		sent, label_gold = r
		outputs.append('%s\tGOLD:%s\tPRED:%s\n' % (sent, label_gold, label_pred))
	if res is None:
		output_path = './outputs/%s.txt' % ds_name
	else:
		acc_number, f1_number, model_name = res
		output_path = './outputs/%s_%s_%s_%s.txt' % (ds_name, model_name, acc_number, f1_number)
	with open(output_path, 'w+') as fp:
		fp.writelines(outputs)


def output_active_ngram(feat_maps, ds_name, model_name, n_filter, pool_size, ks, acc, f1):
	"""
	"""
	file_path = './dataset/%s/test.txt' % ds_name
	test_sents = []
	with open(file_path, 'r') as fp:
		for line in fp:
			sent = line.strip()
			test_sents.append(sent)
	assert len(test_sents) == len(feat_maps)
	output_lines = []
	for (sent, fm) in zip(test_sents, feat_maps):
		# shape of fm: (n_filter, poolsize)
		assert len(fm) == n_filter
		assert len(fm[0]) == pool_size
		count = np.zeros(pool_size)
		for feat in fm:
			feat_np = np.array(feat)
			max_idx = feat_np.argmax()
			count[max_idx] += 1
		top5_ngram = count.argsort()[::-1][:5]
		#ngram_id = count.argmax()
		sent_ngrams = list(ngrams(sent.split(), ks))
		active_ngrams = []
		for i in range(len(top5_ngram)):
			ngram_id = top5_ngram[i]
			rank = i + 1
			frequency = count[ngram_id]
			try:
				#active_ngrams.append('%s:{id:%s, n_grams:%s, frequency:%s}' % (rank, ngram_id, ' '.join(sent_ngrams[ngram_id]), count))
				ng = sent_ngrams[ngram_id]
			except IndexError:
				ng = ['PADDED']
			active_ngrams.append('%s:{id:%s, n_gram:%s, frequency:%s}' % (rank, ngram_id, ' '.join(ng), frequency))
		line = '%s\t%s\n' % (sent, '\t'.join(active_ngrams))
		output_lines.append(line)
	with open('./feat_maps/%s/%s_%s_%s.txt' % (ds_name, model_name, acc, f1), 'w+') as fp:
		fp.writelines(output_lines)

 
