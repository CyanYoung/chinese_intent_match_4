import os

import re

import pandas as pd

from random import shuffle, sample, randint

from util import load_word_re, load_type_re, load_pair, word_replace


def drop(words, bound):
    ind = randint(0, bound)
    words.pop(ind)
    return ''.join(words)


def swap(words, bound):
    ind1, ind2 = randint(0, bound), randint(0, bound)
    words[ind1], words[ind2] = words[ind2], words[ind1]
    return ''.join(words)


def copy(words, bound):
    ind1, ind2 = randint(0, bound), randint(0, bound)
    words.insert(ind1, words[ind2])
    return ''.join(words)


path_stop_word = 'dict/stop_word.txt'
path_type_dir = 'dict/word_type'
path_homo = 'dict/homo.csv'
path_syno = 'dict/syno.csv'
stop_word_re = load_word_re(path_stop_word)
word_type_re = load_type_re(path_type_dir)
homo_dict = load_pair(path_homo)
syno_dict = load_pair(path_syno)

aug_rate, pos_rate, neg_rate = 2, 4, 8

funcs = [drop, swap, copy]


def save_pair(path, pairs):
    head = 'text1,text2,flag'
    with open(path, 'w') as f:
        f.write(head + '\n')
        for text1, text2, flag in pairs:
            f.write(text1 + ',' + text2 + ',' + str(flag) + '\n')


def expand(pairs, path_extra_pair):
    extra_pairs = list()
    for text1, text2, flag in pd.read_csv(path_extra_pair).values:
        extra_pairs.append((text1, text2, flag))
    shuffle(extra_pairs)
    return extra_pairs + pairs


def make_pair(path_aug_dir, paths):
    labels = list()
    label_texts = dict()
    files = os.listdir(path_aug_dir)
    for file in files:
        label = os.path.splitext(file)[0]
        labels.append(label)
        label_texts[label] = list()
        with open(os.path.join(path_aug_dir, file), 'r') as f:
            for line in f:
                label_texts[label].append(line.strip())
    pairs = list()
    for i in range(len(labels)):
        texts = label_texts[labels[i]]
        res_texts = list()
        for j in range(len(labels)):
            if j != i:
                res_texts.extend(label_texts[labels[j]])
        for text in texts:
            pos_texts = sample(texts, pos_rate)
            for pos_text in pos_texts:
                pairs.append((text, pos_text, 1))
            neg_texts = sample(res_texts, neg_rate)
            for neg_text in neg_texts:
                pairs.append((text, neg_text, 0))
    shuffle(pairs)
    bound1 = int(len(pairs) * 0.7)
    bound2 = int(len(pairs) * 0.9)
    train_pairs = expand(pairs[:bound1], paths['extra'])
    save_pair(paths['train'], train_pairs)
    save_pair(paths['dev'], pairs[bound1:bound2])
    save_pair(paths['test'], pairs[bound2:])


def save(path, texts, labels):
    head = 'text,label'
    with open(path, 'w') as f:
        f.write(head + '\n')
        for text, label in zip(texts, labels):
            f.write(text + ',' + label + '\n')


def gather(path_aug_dir, path_train, path_test):
    texts, labels = list(), list()
    files = os.listdir(path_aug_dir)
    for file in files:
        label = os.path.splitext(file)[0]
        with open(os.path.join(path_aug_dir, file), 'r') as f:
            for line in f:
                texts.append(line.strip())
                labels.append(label)
    texts_labels = list(zip(texts, labels))
    shuffle(texts_labels)
    texts, labels = zip(*texts_labels)
    bound = int(len(texts) * 0.9)
    save(path_train, texts[:bound], labels[:bound])
    save(path_test, texts[bound:], labels[bound:])


def clean(text):
    text = re.sub(stop_word_re, '', text)
    for word_type, word_re in word_type_re.items():
        text = re.sub(word_re, word_type, text)
    text = word_replace(text, homo_dict)
    return word_replace(text, syno_dict)


def augment(text):
    aug_texts = list()
    bound = len(text) - 1
    if bound > 0:
        for func in funcs:
            for _ in range(aug_rate):
                words = list(text)
                aug_texts.append(func(words, bound))
    return aug_texts


def prepare(path_univ_dir, path_aug_dir):
    files = os.listdir(path_univ_dir)
    for file in files:
        text_set = set()
        texts = list()
        with open(os.path.join(path_univ_dir, file), 'r') as f:
            for line in f:
                text = line.strip().lower()
                text = clean(text)
                if text and text not in text_set:
                    text_set.add(text)
                    texts.append(text)
                    texts.extend(augment(text))
        with open(os.path.join(path_aug_dir, file), 'w') as f:
            for text in texts:
                f.write(text + '\n')


if __name__ == '__main__':
    path_univ_dir = 'data/univ'
    path_aug_dir = 'data/aug'
    prepare(path_univ_dir, path_aug_dir)
    path_train = 'data/train.csv'
    path_test = 'data/test.csv'
    gather(path_aug_dir, path_train, path_test)
    paths = dict()
    paths['train'] = 'data/train_pair.csv'
    paths['dev'] = 'data/dev_pair.csv'
    paths['test'] = 'data/test_pair.csv'
    paths['extra'] = 'data/extra_pair.csv'
    make_pair(path_univ_dir, paths)
