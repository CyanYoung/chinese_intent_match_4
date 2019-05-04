import pickle as pk

import numpy as np

import torch

from sklearn.ensemble import IsolationForest

from sklearn.cluster import KMeans

from nn_arch import DnnEncode, CnnEncode, RnnEncode

from util import map_item


def load_encode(name, embed_mat, device):
    embed_mat = torch.Tensor(embed_mat)
    model = torch.load(map_item(name, paths), map_location=device)
    full_dict = model.state_dict()
    arch = map_item(name, archs)
    part = arch(embed_mat).to(device)
    part_dict = part.state_dict()
    for part_key in part_dict.keys():
        full_key = 'encode.' + part_key
        if full_key in full_dict:
            part_dict[part_key] = full_dict[full_key]
    part.load_state_dict(part_dict)
    return part


device = torch.device('cpu')

max_core = 5

path_embed = 'feat/embed.pkl'
with open(path_embed, 'rb') as f:
    embed_mat = pk.load(f)

path_sent = 'feat/sent_train.pkl'
path_label = 'feat/label_train.pkl'
with open(path_sent, 'rb') as f:
    sents = pk.load(f)
with open(path_label, 'rb') as f:
    labels = pk.load(f)

archs = {'dnn': DnnEncode,
         'cnn': CnnEncode,
         'rnn': RnnEncode}

paths = {'dnn': 'model/dnn.pkl',
         'cnn': 'model/cnn.pkl',
         'rnn': 'model/rnn.pkl',
         'dnn_cache': 'cache/dnn.pkl',
         'cnn_cache': 'cache/cnn.pkl',
         'rnn_cache': 'cache/rnn.pkl'}

models = {'dnn': load_encode('dnn', embed_mat, device),
          'cnn': load_encode('cnn', embed_mat, device),
          'rnn': load_encode('rnn', embed_mat, device)}


def split(sents, labels):
    label_set = sorted(list(set(labels)))
    labels = np.array(labels)
    sent_mat, label_mat = list(), list()
    for match_label in label_set:
        match_inds = np.where(labels == match_label)
        match_sents = sents[match_inds]
        sent_mat.append(match_sents)
        match_labels = [match_label] * len(match_sents)
        label_mat.append(match_labels)
    return sent_mat, label_mat


def clean(encode_mat, label_mat):
    for i in range(len(encode_mat)):
        model = IsolationForest(n_estimators=100, contamination=0.1)
        model.fit(encode_mat[i])
        flags = model.predict(encode_mat[i])
        count = np.sum(flags > 0)
        if count > max_core:
            inds = np.where(flags < 0)
            encode_mat[i] = np.delete(encode_mat[i], inds, axis=0)
    return encode_mat, label_mat


def merge(encode_mat, label_mat):
    core_sents, core_labels = list(), list()
    for sents, labels in zip(encode_mat, label_mat):
        core_num = min(len(sents), max_core)
        model = KMeans(n_clusters=core_num, n_init=10, max_iter=100)
        model.fit(sents)
        core_sents.extend(model.cluster_centers_.tolist())
        core_labels.extend([labels[0]] * core_num)
    return np.array(core_sents), np.array(core_labels)


def cache(sents, labels):
    sent_mat, label_mat = split(sents, labels)
    for name, model in models.items():
        encode_mat = list()
        with torch.no_grad():
            model.eval()
            for sents in sent_mat:
                sents = torch.LongTensor(sents).to(device)
                encode_mat.append(model(sents).numpy())
        encode_mat, label_mat = clean(encode_mat, label_mat)
        core_sents, core_labels = merge(encode_mat, label_mat)
        path_cache = map_item(name + '_cache', paths)
        with open(path_cache, 'wb') as f:
            pk.dump((core_sents, core_labels), f)


if __name__ == '__main__':
    cache(sents, labels)
