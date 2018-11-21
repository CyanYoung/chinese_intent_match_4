import pickle as pk

import numpy as np

import torch

from sklearn.cluster import KMeans

from nn_arch import DnnEncode, CnnEncode, RnnEncode

from util import flat_read, map_item


def load_encode(name, embed_mat, seq_len):
    embed_mat = torch.Tensor(embed_mat)
    model = torch.load(map_item(name, paths), map_location='cpu')
    full_dict = model.state_dict()
    arch = map_item(name, archs)
    encode = arch(embed_mat, seq_len)
    encode_dict = encode.state_dict()
    part_dict = {key: val for key, val in full_dict.items() if key in encode_dict}
    encode_dict.update(part_dict)
    encode.load_state_dict(encode_dict)
    return encode


seq_len = 30
max_core = 5

path_embed = 'feat/embed.pkl'
with open(path_embed, 'rb') as f:
    embed_mat = pk.load(f)

archs = {'dnn': DnnEncode,
         'cnn': CnnEncode,
         'rnn': RnnEncode}

paths = {'dnn': 'model/dnn.pkl',
         'cnn': 'model/cnn.pkl',
         'rnn': 'model/rnn.pkl',
         'dnn_cache': 'cache/dnn.pkl',
         'cnn_cache': 'cache/cnn.pkl',
         'rnn_cache': 'cache/rnn.pkl'}

models = {'dnn': load_encode('dnn', embed_mat, seq_len),
          'cnn': load_encode('cnn', embed_mat, seq_len),
          'rnn': load_encode('rnn', embed_mat, seq_len)}


def split(sents, labels, path_label):
    label_set = sorted(list(set(labels)))
    labels = np.array(labels)
    sent_mat = list()
    core_labels = list()
    core_nums = list()
    for match_label in label_set:
        match_inds = np.where(labels == match_label)
        match_sents = sents[match_inds]
        sent_mat.append(match_sents)
        core_num = min(len(match_sents), max_core)
        core_nums.append(core_num)
        core_labels.extend([match_label] * core_num)
    with open(path_label, 'wb') as f:
        pk.dump(core_labels, f)
    return sent_mat, core_nums


def cluster(encode_mat, core_nums):
    core_sents = list()
    for sents, core_num in zip(encode_mat, core_nums):
        model = KMeans(n_clusters=core_num, n_init=10, max_iter=100)
        model.fit(sents.numpy())
        core_sents.extend(model.cluster_centers_.tolist())
    return np.array(core_sents)


def cache(path_sent, path_train, path_label):
    with open(path_sent, 'rb') as f:
        sents = pk.load(f)
    labels = flat_read(path_train, 'label')
    sent_mat, core_nums = split(sents, labels, path_label)
    for name, model in models.items():
        with torch.no_grad():
            model.eval()
            encode_mat = list()
            for sents in sent_mat:
                sents = torch.LongTensor(sents)
                encode_mat.append(model(sents))
        core_sents = cluster(encode_mat, core_nums)
        path_cache = map_item(name + '_cache', paths)
        with open(path_cache, 'wb') as f:
            pk.dump(core_sents, f)


if __name__ == '__main__':
    path_train = 'data/train.csv'
    path_sent = 'feat/sent_train.pkl'
    path_label = 'cache/label.pkl'
    cache(path_sent, path_train, path_label)
