import pickle as pk

import numpy as np

from collections import Counter

import torch

from preprocess import clean

from represent import sent2ind

from nn_arch import Match

from encode import load_encode

from util import map_item


def ind2label(label_inds):
    ind_labels = dict()
    for label, ind in label_inds.items():
        ind_labels[ind] = label
    return ind_labels


def load_match(name, device):
    model = torch.load(map_item(name, paths), map_location=device)
    full_dict = model.state_dict()
    part = Match(name).to(device)
    part_dict = part.state_dict()
    for part_key in part_dict.keys():
        full_key = 'match.' + part_key
        if full_key in full_dict:
            part_dict[part_key] = full_dict[full_key]
    part.load_state_dict(part_dict)
    return part


def load_cache(path_cache):
    with open(path_cache, 'rb') as f:
        cache_sents = pk.load(f)
    return cache_sents


device = torch.device('cpu')

seq_len = 30
encode_len = 200

path_word_ind = 'feat/word_ind.pkl'
path_embed = 'feat/embed.pkl'
path_label_ind = 'feat/label_ind.pkl'
path_label = 'feat/label_train.pkl'
with open(path_word_ind, 'rb') as f:
    word_inds = pk.load(f)
with open(path_embed, 'rb') as f:
    embed_mat = pk.load(f)
with open(path_label_ind, 'rb') as f:
    label_inds = pk.load(f)
with open(path_label, 'rb') as f:
    labels = pk.load(f)

ind_labels = ind2label(label_inds)

paths = {'dnn': 'model/dnn.pkl',
         'cnn': 'model/cnn.pkl',
         'rnn': 'model/rnn.pkl',
         'dnn_cache': 'cache/dnn.pkl',
         'cnn_cache': 'cache/cnn.pkl',
         'rnn_cache': 'cache/rnn.pkl'}

caches = {'dnn': load_cache(map_item('dnn_cache', paths)),
          'cnn': load_cache(map_item('cnn_cache', paths)),
          'rnn': load_cache(map_item('rnn_cache', paths))}

models = {'dnn_encode': load_encode('dnn', embed_mat, device),
          'cnn_encode': load_encode('cnn', embed_mat, device),
          'rnn_encode': load_encode('rnn', embed_mat, device),
          'dnn_match': load_match('dnn', device),
          'cnn_match': load_match('cnn', device),
          'rnn_match': load_match('rnn', device)}


def predict(text, name, vote):
    text = clean(text)
    cache_sents = map_item(name, caches)
    cache_sents = torch.Tensor(cache_sents).to(device)
    pad_seq = sent2ind(text, word_inds, seq_len, keep_oov=True)
    sent = torch.LongTensor([pad_seq]).to(device)
    encode = map_item(name + '_encode', models)
    with torch.no_grad():
        encode_seq = encode(sent)
    encode_mat = encode_seq.repeat(len(cache_sents), 1)
    model = map_item(name + '_match', models)
    with torch.no_grad():
        model.eval()
        probs = torch.sigmoid(model(encode_mat, cache_sents))
    probs = probs.numpy()
    probs = np.squeeze(probs, axis=-1)
    max_probs = sorted(probs, reverse=True)[:vote]
    max_inds = np.argsort(-probs)[:vote]
    max_preds = [labels[ind] for ind in max_inds]
    if __name__ == '__main__':
        formats = list()
        for pred, prob in zip(max_preds, max_probs):
            formats.append('{} {:.3f}'.format(ind_labels[pred], prob))
        return ', '.join(formats)
    else:
        pairs = Counter(max_preds)
        return pairs.most_common()[0][0]


if __name__ == '__main__':
    while True:
        text = input('text: ')
        print('dnn: %s' % predict(text, 'dnn', vote=5))
        print('cnn: %s' % predict(text, 'cnn', vote=5))
        print('rnn: %s' % predict(text, 'rnn', vote=5))
