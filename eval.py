import pickle as pk

import numpy as np

import torch

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from build import tensorize

from match import ind_labels, predict

from util import flat_read, map_item


device = torch.device('cpu')

path_test = 'data/test.csv'
path_label = 'feat/label_test.pkl'
texts = flat_read(path_test, 'text')
with open(path_label, 'rb') as f:
    labels = pk.load(f)

path_test_pair = 'data/test_pair.csv'
path_pair = 'feat/pair_test.pkl'
path_flag = 'feat/flag_test.pkl'
text1s = flat_read(path_test_pair, 'text1')
text2s = flat_read(path_test_pair, 'text2')
with open(path_pair, 'rb') as f:
    pairs = pk.load(f)
with open(path_flag, 'rb') as f:
    flags = pk.load(f)

class_num = len(ind_labels)

paths = {'dnn': 'model/dnn.pkl',
         'cnn': 'model/cnn.pkl',
         'rnn': 'model/rnn.pkl',
         'dnn_metric': 'metric/dnn.csv',
         'cnn_metric': 'metric/cnn.csv',
         'rnn_metric': 'metric/rnn.csv'}

models = {'dnn': torch.load(map_item('dnn', paths), map_location=device),
          'cnn': torch.load(map_item('cnn', paths), map_location=device),
          'rnn': torch.load(map_item('rnn', paths), map_location=device)}


def test_pair(name, pairs, flags, thre):
    sent1s, sent2s = tensorize(pairs, device)
    model = map_item(name, models)
    with torch.no_grad():
        model.eval()
        probs = torch.sigmoid(model(sent1s, sent2s))
    probs = probs.numpy()
    probs = np.squeeze(probs, axis=-1)
    preds = probs > thre
    f1 = f1_score(flags, preds)
    print('\n%s f1: %.2f - acc: %.2f\n' % (name, f1, accuracy_score(flags, preds)))
    for flag, prob, text1, text2, pred in zip(flags, probs, text1s, text2s, preds):
        if flag != pred:
            print('{} {:.3f} {} | {}'.format(flag, prob, text1, text2))


def test(name, texts, labels, vote):
    preds = list()
    for text in texts:
        preds.append(predict(text, name, vote))
    precs = precision_score(labels, preds, average=None)
    recs = recall_score(labels, preds, average=None)
    with open(map_item(name + '_metric', paths), 'w') as f:
        f.write('label,prec,rec' + '\n')
        for i in range(class_num):
            f.write('%s,%.2f,%.2f\n' % (ind_labels[i], precs[i], recs[i]))
    f1 = f1_score(labels, preds, average='weighted')
    print('\n%s f1: %.2f - acc: %.2f\n' % (name, f1, accuracy_score(labels, preds)))
    for text, label, pred in zip(texts, labels, preds):
        if label != pred:
            print('{}: {} -> {}'.format(text, ind_labels[label], ind_labels[pred]))


if __name__ == '__main__':
    test_pair('dnn', pairs, flags, thre=0.5)
    test_pair('cnn', pairs, flags, thre=0.5)
    test_pair('rnn', pairs, flags, thre=0.5)
    test('dnn', texts, labels, vote=5)
    test('cnn', texts, labels, vote=5)
    test('rnn', texts, labels, vote=5)
