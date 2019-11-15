import pickle as pkl
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utils import *
data = download_dataset_tokenize(DATASET_URL)
words_to_int, int_to_words, train_set = text_preprocessing(data)
#Initialize subword generator
gen = generate_subwords()
next(gen)   
rw_pairs = list()
human_similarity = list()
#Load word pairs and human similarities into corresponding lists
with open('rw.txt', 'r') as fp:
    for line in fp:
        line = line.split()
        rw_pairs.append([line[0], line[1]])
        human_similarity.append(float(line[2]))
model = pkl.load(open('embeddings.pkl','rb'))
fastText_similarity = list()
#For each word in the pair, generate subwords, sum up the subword vectors to perform cosine similarities
for x1, x2 in rw_pairs:
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
#Find spearman rank coeff between human similarity and fastText cosine similarity
print(spearman(torch.FloatTensor(fastText_similarity), torch.FloatTensor(human_similarity)))
