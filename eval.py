import pickle as pk

import numpy as np

import torch

from sklearn.metrics import accuracy_score

from match import predict

from util import flat_read, map_item


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

paths = {'dnn': 'model/dnn.pkl',
         'cnn': 'model/cnn.pkl',
         'rnn': 'model/rnn.pkl'}

models = {'dnn': torch.load(map_item('dnn', paths), map_location='cpu'),
          'cnn': torch.load(map_item('cnn', paths), map_location='cpu'),
          'rnn': torch.load(map_item('rnn', paths), map_location='cpu')}


def test_pair(name, pairs, flags, thre):
    sent1s, sent2s = pairs
    sent1s, sent2s = torch.LongTensor(sent1s), torch.LongTensor(sent2s)
    model = map_item(name, models)
    with torch.no_grad():
        model.eval()
        probs = torch.sigmoid(model(sent1s, sent2s))
    probs = probs.numpy()
    probs = np.squeeze(probs, axis=-1)
    preds = probs > thre
    print('\n%s %s %.2f\n' % (name, 'acc:', accuracy_score(flags, preds)))
    for flag, prob, text1, text2, pred in zip(flags, probs, text1s, text2s, preds):
        if flag != pred:
            print('{} {:.3f} {} | {}'.format(flag, prob, text1, text2))


def test(name, texts, labels, vote):
    preds = list()
    for text in texts:
        preds.append(predict(text, name, vote))
    print('\n%s %s %.2f\n' % (name, 'acc:', accuracy_score(labels, preds)))
    for text, label, pred in zip(texts, labels, preds):
        if label != pred:
            print('{}: {} -> {}'.format(text, label, pred))


if __name__ == '__main__':
    test_pair('dnn', pairs, flags, thre=0.5)
    test_pair('cnn', pairs, flags, thre=0.5)
    test_pair('rnn', pairs, flags, thre=0.5)
    test('dnn', texts, labels, vote=5)
    test('cnn', texts, labels, vote=5)
    test('rnn', texts, labels, vote=5)
