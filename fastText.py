from utils import *
import time 

data = download_dataset_tokenize(DATASET_URL)
words_to_int, int_to_words, train_set = text_preprocessing(data)
train_graph = tf.Graph()

with train_graph.as_default():
    #Placeholder for input and labels
    train_inputs = tf.placeholder(tf.int32, shape=[None])
    train_labels = tf.placeholder(tf.int32, shape=[None, None])
    valid_dataset = tf.constant(VALID_EXAMPLES)
    #Create embeddings for hidden layer
    embeddings = tf.Variable(tf.random_uniform([VOCAB_SIZE, EMBEDDING_SIZE], -1.0, 1.0))
    #Only care about input word vectors in the embedding
    embed = tf.nn.embedding_lookup(embeddings, train_inputs)
    #Calculate loss using negative sampling
    softmax_weights = tf.Variable(tf.truncated_normal(shape = (VOCAB_SIZE, EMBEDDING_SIZE), stddev=0.1))
    softmax_biases = tf.Variable(tf.zeros([VOCAB_SIZE]))
    loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=softmax_weights,
                                      biases=softmax_biases,
                                      labels=train_labels,
                                      inputs=embed,
                                      num_sampled=100,
                                #  num_sampled=5*train_inputs.shape[0],
                                      num_classes=VOCAB_SIZE))
    #Backprop
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    # optimizer = tf.train.GradientDescentOptimizer(0.025).minimizer(loss)
    saver = tf.train.Saver()
    #Calculate final word embedding
    word_vectors = tf.Variable(initial_value=tf.zeros([len(words_to_int), EMBEDDING_SIZE]))
    word_index = tf.placeholder(tf.int32)
    summed_vectors = tf.placeholder(tf.float32, shape=[None])
    update_vector = tf.scatter_update(word_vectors, word_index, summed_vectors)
      # Compute the cosine similarity between minibatch examples and all embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(word_vectors), 1, keep_dims=True))
    normalized_embeddings = word_vectors / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

with tf.Session(graph=train_graph) as sess:
    print('Start training')
    sess.run(tf.global_variables_initializer())
    for epoch in range(EPOCHS):
        iteration = 1
        total_loss = 0
        batch_gen = generate_batches(words_to_int, int_to_words, BATCH_SIZE, WINDOW_SIZE)
        start_time = time.time()
        for input, target in batch_gen:
            feed_dict = {train_inputs: input, train_labels: np.array([target]).T}
            loss, _ = sess.run([loss, optimizer], feed_dict=feed_dict)
            total_loss += loss
            if iteration % 2000 == 0:
                end_time = time.time()
                print('Epoch: {}'.format(epoch + 1))
                print('Iteration: {}'.format(iteration))
                print('Average loss: {}'.format(total_loss / 2000))
                print('{} sec/batch'.format((end_time - start_time) / 2000))
                total_loss = 0
                start_time = time.time()
            iteration += 1
    #Generate subword for each word and calculate the summed word vectors of all subwords
    subword_gen = generate_subwords()
    next(subword_gen)
    for word in words_to_int:
        subwords = subword_gen.send(word)
        next(subword_gen)
        #Sum up word vectors of all subwords
        summed_vectors = tf.reduce_sum(tf.nn.embedding_lookup(embeddings, subwords), 0)
        #Update the word embeddings
        sess.run(update_vector, feed_dict={word_index:[words_to_int[word]], summed_vectors:summed_vectors})
    final_embeddings = normalized_embeddings.eval()

def plot_with_labels(low_dim_embs, labels, filename='tsne.png'):
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
    plt.scatter(x, y)
    plt.annotate(label,
                    xy=(x, y),
                    xytext=(5, 2),
                    textcoords='offset points',
                    ha='right',
                    va='bottom')

    plt.savefig(filename)

try:
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
    plot_only = 500
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    labels = [int_to_words[i] for i in range(plot_only)]
    plot_with_labels(low_dim_embs, labels)
except ImportError:
    print('Please install sklearn, matplotlib, and scipy to show embeddings.')

        








