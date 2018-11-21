import os

import re

import pandas as pd


def load_word(path):
    words = list()
    with open(path, 'r') as f:
        for line in f:
            words.append(line.strip())
    return words


def load_word_re(path):
    words = load_word(path)
    return '(' + ')|('.join(words) + ')'


def load_type_re(path_dir):
    word_type_re = dict()
    files = os.listdir(path_dir)
    for file in files:
        word_type = os.path.splitext(file)[0]
        word_type_re[word_type] = load_word_re(os.path.join(path_dir, file))
    return word_type_re


def load_pair(path):
    vocab = dict()
    for std, nstd in pd.read_csv(path).values:
        if nstd not in vocab:
            vocab[nstd] = std
    return vocab


def word_replace(text, pair):
    for nstd, std in pair.items():
        text = re.sub(nstd, std, text)
    return text


def flat_read(path, field):
    nest_items = pd.read_csv(path, usecols=[field], keep_default_na=False).values
    items = list()
    for nest_item in nest_items:
        items.append(nest_item[0])
    return items


def map_item(name, items):
    if name in items:
        return items[name]
    else:
        raise KeyError
