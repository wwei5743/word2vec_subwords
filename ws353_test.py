import pickle as pkl
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utils import generate_subwords
from scipy import stats
import csv
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
   x1_vector = np.sum(model[x1_subwords, :], axis=0).reshape(1,300)
   x2_vector = np.sum(model[x2_subwords, :], axis=0).reshape(1,300)
   similarity = cosine_similarity(x1_vector, x2_vector)
   fastText_similarity.append(similarity[0][0])
print(stats.spearmanr(human_similarity, fastText_similarity))
