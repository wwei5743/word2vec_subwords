import pickle as pkl
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from utils import generate_subwords
from scipy import stats
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
        mean = np.mean([float(elem) for elem in line[2:]])
        human_similarity.append(mean)
model = pkl.load(open('embeddings.pkl','rb'))
fastText_similarity = list()
#For each word in the pair, generate subwords, sum up the subword vectors to perform cosine similarities
for x1, x2 in rw_pairs:
   x1_subwords = gen.send(x1) 
   print(x1_subwords)
   next(gen)
   x2_subwords = gen.send(x2) 
   next(gen)
   x1_vector = np.sum(model[x1_subwords, :], axis=0).reshape(1,300)
   x2_vector = np.sum(model[x2_subwords, :], axis=0).reshape(1,300)
   similarity = cosine_similarity(x1_vector, x2_vector)
   fastText_similarity.append(similarity[0][0])
#Find spearman rank coeff between human similarity and fastText cosine similarity
print(stats.spearmanr(human_similarity, fastText_similarity))
