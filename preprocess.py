import os

import re

import pandas as pd

from random import shuffle, sample

from util import load_word_re, load_type_re, load_pair, word_replace


path_stop_word = 'dict/stop_word.txt'
path_type_dir = 'dict/word_type'
path_homo = 'dict/homonym.csv'
path_syno = 'dict/synonym.csv'
stop_word_re = load_word_re(path_stop_word)
word_type_re = load_type_re(path_type_dir)
homo_dict = load_pair(path_homo)
syno_dict = load_pair(path_syno)


def save_pair(path, pairs):
    head = 'text1,text2,flag'  # prox
    with open(path, 'w') as f:
        f.write(head + '\n')
        for text1, text2, flag in pairs:
            f.write(text1 + ',' + text2 + ',' + str(flag) + '\n')


def extend(pairs, path_extra_pair):
    extra_pairs = list()
    for text1, text2, flag in pd.read_csv(path_extra_pair).values:
        extra_pairs.append((text1, text2, flag))
    shuffle(extra_pairs)
    return extra_pairs + pairs


def insert(pairs, text, neg_texts, neg_fold):
    sub_texts = sample(neg_texts, neg_fold)
    for neg_text in sub_texts:
        pairs.append((text, neg_text, 0))


def make_pair(path_univ_dir, path_pairs):
    labels = list()
    label_texts = dict()
    files = os.listdir(path_univ_dir)
    for file in files:
        label = os.path.splitext(file)[0]
        labels.append(label)
        label_texts[label] = list()
        with open(os.path.join(path_univ_dir, file), 'r') as f:
            for line in f:
                label_texts[label].append(line.strip())
    neg_fold = 2
    pairs = list()
    for i in range(len(labels)):
        texts = label_texts[labels[i]]
        neg_texts = list()
        for j in range(len(labels)):
            if j != i:
                neg_texts.extend(label_texts[labels[j]])
        for j in range(len(texts) - 1):
            for k in range(j + 1, len(texts)):
                pairs.append((texts[j], texts[k], 1))
                sub_texts = sample(neg_texts, neg_fold)
                for neg_text in sub_texts:
                    pairs.append((texts[j], neg_text, 0))
    shuffle(pairs)
    bound1 = int(len(pairs) * 0.7)
    bound2 = int(len(pairs) * 0.9)
    train_pairs = extend(pairs[:bound1], path_pairs['extra'])
    save_pair(path_pairs['train'], train_pairs)
    save_pair(path_pairs['dev'], pairs[bound1:bound2])
    save_pair(path_pairs['test'], pairs[bound2:])


def save(path, texts, labels):
    head = 'text,label'
    with open(path, 'w') as f:
        f.write(head + '\n')
        for text, label in zip(texts, labels):
            f.write(text + ',' + label + '\n')


def gather(path_univ_dir, path_train, path_test):
    texts = list()
    labels = list()
    files = os.listdir(path_univ_dir)
    for file in files:
        label = os.path.splitext(file)[0]
        with open(os.path.join(path_univ_dir, file), 'r') as f:
            for line in f:
                texts.append(line.strip())
                labels.append(label)
    texts_labels = list(zip(texts, labels))
    shuffle(texts_labels)
    texts, labels = zip(*texts_labels)
    bound = int(len(texts) * 0.9)
    save(path_train, texts[:bound], labels[:bound])
    save(path_test, texts[bound:], labels[bound:])


def prepare(path_univ_dir):
    files = os.listdir(path_univ_dir)
    for file in files:
        text_set = set()
        texts = list()
        with open(os.path.join(path_univ_dir, file), 'r') as f:
            for line in f:
                text = line.strip().lower()
                text = re.sub(stop_word_re, '', text)
                for word_type, word_re in word_type_re.items():
                    text = re.sub(word_re, word_type, text)
                text = word_replace(text, homo_dict)
                text = word_replace(text, syno_dict)
                if text and text not in text_set:
                    text_set.add(text)
                    texts.append(text)
        with open(os.path.join(path_univ_dir, file), 'w') as f:
            for text in texts:
                f.write(text + '\n')


if __name__ == '__main__':
    path_univ_dir = 'data/univ'
    prepare(path_univ_dir)
    path_train = 'data/train.csv'
    path_test = 'data/test.csv'
    gather(path_univ_dir, path_train, path_test)
    path_pairs = dict()
    path_pairs['train'] = 'data/train_pair.csv'
    path_pairs['dev'] = 'data/dev_pair.csv'
    path_pairs['test'] = 'data/test_pair.csv'
    path_pairs['extra'] = 'data/extra_pair.csv'
    make_pair(path_univ_dir, path_pairs)
