import pickle as pkl
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utils import *
import csv
data = download_dataset_tokenize(DATASET_URL)
words_to_int, int_to_words, train_set = text_preprocessing(data)
gen = generate_subwords()
next(gen)   
ws353_pairs = list()
human_similarity = list()
with open('ws353.csv', 'r') as fp:
    csvReader = csv.reader(fp)
    row_idx = 0
    for row in csvReader:
        if row_idx == 0:
            row_idx += 1
            continue
        ws353_pairs.append([row[0], row[1]])
        human_similarity.append(float(row[2]))
model = pkl.load(open('embeddings.pkl','rb'))
fastText_similarity = list()
for x1, x2 in ws353_pairs:
    x1_subwords = gen.send(x1) 
    next(gen)
    x2_subwords = gen.send(x2) 
    next(gen)
    if x1 not in words_to_int:
        x1_vector = np.sum(model[x1_subwords, :], axis=0).reshape(1,300) / len(x1_subwords)
    else:
        x1_vector = np.sum(model[x1_subwords, :], axis=0).reshape(1,300)
    if x2 not in words_to_int:
        x2_vector = np.sum(model[x2_subwords, :], axis=0).reshape(1,300) / len(x2_subwords)
    else:
        x2_vector = np.sum(model[x2_subwords, :], axis=0).reshape(1,300)
    similarity = cosine_similarity(x1_vector, x2_vector)
    fastText_similarity.append(similarity[0][0])
print(spearman(torch.FloatTensor(human_similarity), torch.FloatTensor(fastText_similarity)))
