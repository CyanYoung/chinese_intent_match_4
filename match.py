import pickle as pk

import re

import numpy as np

from collections import Counter

import torch

from represent import sent2ind

from nn_arch import Match

from encode import load_encode

from util import load_word_re, load_type_re, load_pair, word_replace, map_item


def load_match(name, device):
    model = torch.load(map_item(name, paths), map_location=device)
    full_dict = model.state_dict()
    match = Match()
    match_dict = match.state_dict()
    part_dict = {key: val for key, val in full_dict.items() if key in match_dict}
    match_dict.update(part_dict)
    match.load_state_dict(match_dict)
    return match


def load_cache(path_cache):
    with open(path_cache, 'rb') as f:
        core_sents = pk.load(f)
    return core_sents


device = torch.device('cpu')

seq_len = 30
encode_len = 200

path_stop_word = 'dict/stop_word.txt'
path_type_dir = 'dict/word_type'
path_homo = 'dict/homo.csv'
path_syno = 'dict/syno.csv'
stop_word_re = load_word_re(path_stop_word)
word_type_re = load_type_re(path_type_dir)
homo_dict = load_pair(path_homo)
syno_dict = load_pair(path_syno)

path_word_ind = 'feat/word_ind.pkl'
path_embed = 'feat/embed.pkl'
path_label = 'cache/label.pkl'
with open(path_word_ind, 'rb') as f:
    word_inds = pk.load(f)
with open(path_embed, 'rb') as f:
    embed_mat = pk.load(f)
with open(path_label, 'rb') as f:
    core_labels = pk.load(f)

oov_ind = len(embed_mat) - 1

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
    text = re.sub(stop_word_re, '', text.strip())
    for word_type, word_re in word_type_re.items():
        text = re.sub(word_re, word_type, text)
    text = word_replace(text, homo_dict)
    text = word_replace(text, syno_dict)
    core_sents = map_item(name, caches)
    core_sents = torch.Tensor(core_sents).to(device)
    pad_seq = sent2ind(text, word_inds, seq_len, oov_ind, keep_oov=True)
    sent = torch.LongTensor([pad_seq]).to(device)
    encode = map_item(name + '_encode', models)
    with torch.no_grad():
        encode_seq = encode(sent)
    encode_mat = encode_seq.repeat(len(core_sents), 1)
    model = map_item(name + '_match', models)
    with torch.no_grad():
        model.eval()
        probs = torch.sigmoid(model(encode_mat, core_sents))
    probs = probs.numpy()
    probs = np.squeeze(probs, axis=-1)
    max_probs = sorted(probs, reverse=True)[:vote]
    max_inds = np.argsort(-probs)[:vote]
    max_preds = [core_labels[ind] for ind in max_inds]
    if __name__ == '__main__':
        formats = list()
        for pred, prob in zip(max_preds, max_probs):
            formats.append('{} {:.3f}'.format(pred, prob))
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
