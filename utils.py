import requests
import os
import zipfile
import tensorflow as tf
if tf.__version__.split('.')[0] == '2':
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
    tf.disable_eager_execution()
from collections import Counter
import numpy as np
import pyhash
import math
import random
import time

DATASET_URL = 'http://mattmahoney.net/dc/text8.zip'
MYDIR = os.path.dirname(os.path.realpath(__file__))
REJ_THRESHOLD = 1e-4
HASHING_UB = 500000
VOCAB_SIZE = HASHING_UB
EMBEDDING_SIZE = 300
BATCH_SIZE = 400
EPOCHS = 10
WINDOW_SIZE = 5
TRAIN_PERCENT = 0.2
hasher = pyhash.fnv1a_32()
myhasher = lambda x : hasher(x) % HASHING_UB
LB_NGRAM = 3
UB_NGRAM = 6
random.seed(time.clock())

def download_dataset_tokenize(url):
    print('Beginning file download from {}'.format(url))
    filename = url.split('/')[-1]
    if filename in os.listdir(MYDIR):
        print('Requested dataset already exists in current directory')
        if zipfile.is_zipfile(filename):
            with zipfile.ZipFile(os.path.join(MYDIR, filename)) as fp:
                data = tf.compat.as_str(fp.read(fp.namelist()[0])).split()
            train_data = data[:int(len(data) * TRAIN_PERCENT)]
        return train_data
    try:
        r = requests.get(url)
        with open(os.path.join(MYDIR, filename), 'wb') as fp:
            fp.write(r.content)
    except Exception as e:
        raise e
    print('File downloaded')
    if zipfile.is_zipfile(filename):
        with zipfile.ZipFile(os.path.join(MYDIR, filename)) as fp:
            data = tf.compat.as_str(fp.read(fp.namelist()[0])).split()
    train_data = data[:int(len(data) * TRAIN_PERCENT)]
    return train_data

def text_preprocessing(words):
    trained_text = list()
    freq = Counter(words)
    total_count = len(words)
    for _, word in enumerate(words):
        #Only include words that occurs more than 4 times
        if freq[word] >= 5:
            #Calculate subsampling probability
            discard_prob = 1 - np.sqrt(REJ_THRESHOLD / (freq[word] / total_count))
            if np.random.random() < 1 - discard_prob:
                trained_text.append(word)
    new_freq = Counter(trained_text)
    new_freq = new_freq.most_common(len(new_freq))
    words_to_int = {word[0]: index for index, word in enumerate(new_freq)}
    int_to_words = {index: word for word, index in words_to_int.items()}
    trained_text = [words_to_int[word] for word in trained_text]
    return words_to_int, int_to_words, trained_text

def generate_subwords():
    while True:
        word = yield
        subwords = set()
        word = '<{}>'.format(word)
        if len(word) <= LB_NGRAM:
            yield [myhasher(word)]
        else:
            subwords.add(myhasher(word))
            for window in range(LB_NGRAM, UB_NGRAM + 1):
                for index in range(len(word) - window + 1):
                    subwords.add(myhasher(word[index: index+window]))
            yield list(subwords)

def generate_batches(words_int, int_to_words, batch_size, window_size):
    subword_generator = generate_subwords()
    next(subword_generator)
    for index in range(math.ceil(len(words_int) / batch_size)):
        batch = words_int[index * batch_size: min((index + 1) * batch_size, len(words_int))]
        input = list()
        target = list()
        for curr_index in batch:
            curr_window_size = random.randint(1, window_size)
            neighbors = set(words_int[max(0, curr_index - curr_window_size) : curr_index] 
                    + words_int[curr_index + 1 : min(len(words_int), curr_index + curr_window_size + 1)])
            neighbors = list(neighbors)
            subwords = subword_generator.send(int_to_words[curr_index])
            next(subword_generator)
            for subword in subwords:
                input.extend([subword for _ in range(len(neighbors))])
            for _ in range(len(subwords)):
                target.extend(neighbors)
        yield input, target