import time

import pickle as pk

import torch
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader

from nn_arch import Dnn, Cnn, Rnn

from util import map_item


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

detail = False if torch.cuda.is_available() else True

batch_size = 128

path_embed = 'feat/embed.pkl'
with open(path_embed, 'rb') as f:
    embed_mat = pk.load(f)

archs = {'dnn': Dnn,
         'cnn': Cnn,
         'rnn': Rnn}

paths = {'dnn': 'model/dnn.pkl',
         'cnn': 'model/cnn.pkl',
         'rnn': 'model/rnn.pkl'}


def load_feat(path_feats):
    with open(path_feats['pair_train'], 'rb') as f:
        train_pairs = pk.load(f)
    with open(path_feats['flag_train'], 'rb') as f:
        train_flags = pk.load(f)
    with open(path_feats['pair_dev'], 'rb') as f:
        dev_pairs = pk.load(f)
    with open(path_feats['flag_dev'], 'rb') as f:
        dev_flags = pk.load(f)
    return train_pairs, train_flags, dev_pairs, dev_flags


def tensorize(feats, device):
    tensors = list()
    for feat in feats:
        tensors.append(torch.LongTensor(feat).to(device))
    return tensors


def get_loader(pairs, flags):
    sent1s, sent2s = pairs
    triples = TensorDataset(sent1s, sent2s, flags)
    return DataLoader(triples, batch_size=batch_size, shuffle=True)


def get_metric(model, loss_func, pairs, flags, thre):
    sent1s, sent2s = pairs
    probs = model(sent1s, sent2s)
    probs = torch.squeeze(probs, dim=-1)
    preds = probs > thre
    loss = loss_func(probs, flags.float())
    acc = (preds == flags.byte()).sum().item() / len(preds)
    return loss, acc


def step_print(step, batch_loss, batch_acc):
    print('\n{} {} - loss: {:.3f} - acc: {:.3f}'.format('step', step, batch_loss, batch_acc))


def epoch_print(epoch, delta, train_loss, train_acc, dev_loss, dev_acc, extra):
    print('\n{} {} - {:.2f}s - loss: {:.3f} - acc: {:.3f} - val_loss: {:.3f} - val_acc: {:.3f}'.format(
          'epoch', epoch, delta, train_loss, train_acc, dev_loss, dev_acc) + extra)


def fit(name, max_epoch, embed_mat, path_feats, detail):
    feats = load_feat(path_feats)
    train_pairs, train_flags, dev_pairs, dev_flags = tensorize(feats, device)
    train_loader = get_loader(train_pairs, train_flags)
    embed_mat = torch.Tensor(embed_mat)
    seq_len = len(train_pairs[0][0])
    arch = map_item(name, archs)
    model = arch(embed_mat, seq_len).to(device)
    loss_func = BCEWithLogitsLoss()
    learn_rate, min_rate = 1e-3, 1e-5
    min_dev_loss = float('inf')
    trap_count, max_count = 0, 5
    print('\n{}'.format(model))
    train, epoch = True, 0
    while train and epoch < max_epoch:
        epoch = epoch + 1
        model.train()
        optimizer = Adam(model.parameters(), lr=learn_rate)
        start = time.time()
        for step, (sent1s, sent2s, flags) in enumerate(train_loader):
            batch_loss, batch_acc = get_metric(model, loss_func, [sent1s, sent2s], flags, thre=0)
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            if detail:
                step_print(step + 1, batch_loss, batch_acc)
        delta = time.time() - start
        with torch.no_grad():
            model.eval()
            train_loss, train_acc = get_metric(model, loss_func, train_pairs, train_flags, thre=0)
            dev_loss, dev_acc = get_metric(model, loss_func, dev_pairs, dev_flags, thre=0)
        extra = ''
        if dev_loss < min_dev_loss:
            extra = ', val_loss reduce by {:.3f}'.format(min_dev_loss - dev_loss)
            min_dev_loss = dev_loss
            trap_count = 0
            torch.save(model, map_item(name, paths))
        else:
            trap_count = trap_count + 1
            if trap_count > max_count:
                learn_rate = learn_rate / 10
                if learn_rate < min_rate:
                    extra = ', early stop'
                    train = False
                else:
                    extra = ', learn_rate divide by 10'
                    trap_count = 0
        epoch_print(epoch, delta, train_loss, train_acc, dev_loss, dev_acc, extra)


if __name__ == '__main__':
    path_feats = dict()
    path_feats['pair_train'] = 'feat/pair_train.pkl'
    path_feats['flag_train'] = 'feat/flag_train.pkl'
    path_feats['pair_dev'] = 'feat/pair_dev.pkl'
    path_feats['flag_dev'] = 'feat/flag_dev.pkl'
    fit('dnn', 50, embed_mat, path_feats, detail)
    fit('cnn', 50, embed_mat, path_feats, detail)
    fit('rnn', 50, embed_mat, path_feats, detail)
