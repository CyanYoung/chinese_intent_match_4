import pickle as pk

import numpy as np

from gensim.corpora import Dictionary

from util import flat_read


embed_len = 200
min_freq = 1
max_vocab = 5000
seq_len = 30

pad_ind, oov_ind = 0, 1

path_word_vec = 'feat/word_vec.pkl'
path_word_ind = 'feat/word_ind.pkl'
path_embed = 'feat/embed.pkl'
path_label_ind = 'feat/label_ind.pkl'


def tran_dict(word_inds, off):
    off_word_inds = dict()
    for word, ind in word_inds.items():
        off_word_inds[word] = ind + off
    return off_word_inds


def embed(sent_words, path_word_ind, path_word_vec, path_embed):
    model = Dictionary(sent_words)
    model.filter_extremes(no_below=min_freq, no_above=1.0, keep_n=max_vocab)
    word_inds = model.token2id
    word_inds = tran_dict(word_inds, off=2)
    with open(path_word_ind, 'wb') as f:
        pk.dump(word_inds, f)
    with open(path_word_vec, 'rb') as f:
        word_vecs = pk.load(f)
    vocab = word_vecs.vocab
    vocab_num = min(max_vocab + 2, len(word_inds) + 2)
    embed_mat = np.zeros((vocab_num, embed_len))
    for word, ind in word_inds.items():
        if word in vocab:
            if ind < max_vocab:
                embed_mat[ind] = word_vecs[word]
    with open(path_embed, 'wb') as f:
        pk.dump(embed_mat, f)


def label2ind(labels, path_label_ind):
    labels = sorted(list(set(labels)))
    label_inds = dict()
    for i in range(len(labels)):
        label_inds[labels[i]] = i
    with open(path_label_ind, 'wb') as f:
        pk.dump(label_inds, f)


def sent2ind(words, word_inds, seq_len, keep_oov):
    seq = list()
    for word in words:
        if word in word_inds:
            seq.append(word_inds[word])
        elif keep_oov:
            seq.append(oov_ind)
    if len(seq) < seq_len:
        return [pad_ind] * (seq_len - len(seq)) + seq
    else:
        return seq[-seq_len:]


def align(sent_words):
    with open(path_word_ind, 'rb') as f:
        word_inds = pk.load(f)
    pad_seqs = list()
    for words in sent_words:
        pad_seq = sent2ind(words, word_inds, seq_len, keep_oov=True)
        pad_seqs.append(pad_seq)
    return np.array(pad_seqs)


def vectorize(path_data, path_sent, path_label, mode):
    sents = flat_read(path_data, 'text')
    sent_words = [list(sent) for sent in sents]
    labels = flat_read(path_data, 'label')
    if mode == 'train':
        embed(sent_words, path_word_ind, path_word_vec, path_embed)
        label2ind(labels, path_label_ind)
    pad_seqs = align(sent_words)
    with open(path_label_ind, 'rb') as f:
        label_inds = pk.load(f)
    inds = list()
    for label in labels:
        inds.append(label_inds[label])
    inds = np.array(inds)
    with open(path_sent, 'wb') as f:
        pk.dump(pad_seqs, f)
    with open(path_label, 'wb') as f:
        pk.dump(inds, f)


def vectorize_pair(path_data, path_pair, path_flag):
    sent1s = flat_read(path_data, 'text1')
    sent2s = flat_read(path_data, 'text2')
    flags = flat_read(path_data, 'flag')
    pad_seq1s, pad_seq2s = align(sent1s), align(sent2s)
    pairs = (pad_seq1s, pad_seq2s)
    flags = np.array(flags)
    with open(path_pair, 'wb') as f:
        pk.dump(pairs, f)
    with open(path_flag, 'wb') as f:
        pk.dump(flags, f)


if __name__ == '__main__':
    path_data = 'data/train.csv'
    path_sent = 'feat/sent_train.pkl'
    path_label = 'feat/label_train.pkl'
    vectorize(path_data, path_sent, path_label, 'train')
    path_data = 'data/test.csv'
    path_sent = 'feat/sent_test.pkl'
    path_label = 'feat/label_test.pkl'
    vectorize(path_data, path_sent, path_label, 'test')
    path_data = 'data/train_pair.csv'
    path_pair = 'feat/pair_train.pkl'
    path_flag = 'feat/flag_train.pkl'
    vectorize_pair(path_data, path_pair, path_flag)
    path_data = 'data/dev_pair.csv'
    path_pair = 'feat/pair_dev.pkl'
    path_flag = 'feat/flag_dev.pkl'
    vectorize_pair(path_data, path_pair, path_flag)
    path_data = 'data/test_pair.csv'
    path_pair = 'feat/pair_test.pkl'
    path_flag = 'feat/flag_test.pkl'
    vectorize_pair(path_data, path_pair, path_flag)
