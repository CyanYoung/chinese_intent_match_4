import pickle as pk

import torch

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

path_embed = 'feat/embed.pkl'
path_sent = 'feat/sent_train.pkl'
with open(path_embed, 'rb') as f:
    embed_mat = pk.load(f)
with open(path_sent, 'rb') as f:
    sents = pk.load(f)

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


def cache(sents):
    sents = torch.LongTensor(sents).to(device)
    for name, model in models.items():
        with torch.no_grad():
            model.eval()
            encode_sents = model(sents).numpy()
        path_cache = map_item(name + '_cache', paths)
        with open(path_cache, 'wb') as f:
            pk.dump(encode_sents, f)


if __name__ == '__main__':
    cache(sents)
