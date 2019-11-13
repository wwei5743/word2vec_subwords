from utils import *
import time 
import pickle as pkl

data = download_dataset_tokenize(DATASET_URL)
words_to_int, int_to_words, train_set = text_preprocessing(data)
train_graph = tf.Graph()

with train_graph.as_default():
    #Placeholder for input and labels
    train_inputs = tf.placeholder(tf.int32, shape=[None])
    train_labels = tf.placeholder(tf.int32, shape=[None, None])
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
    #Backpropagation with Adam Optimizer
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    #Backpropagation with SGD with step size = 
    # optimizer = tf.train.GradientDescentOptimizer(0.05).minimizer(loss)

with tf.Session(graph=train_graph) as sess:
    print('Start training')
    sess.run(tf.global_variables_initializer())
    for epoch in range(EPOCHS):
        iteration = 0
        total_loss = 0
        batch_gen = generate_batches(train_set, int_to_words, BATCH_SIZE, WINDOW_SIZE)
        start_time = time.time()
        for input, target in batch_gen:
            feed_dict = {train_inputs: input, train_labels: np.array([target]).T}
            curr_loss, _ = sess.run([loss, optimizer], feed_dict=feed_dict)
            total_loss += curr_loss
            if iteration != 0 and iteration % 1000 == 0:
                end_time = time.time()
                print('Epoch: {}'.format(epoch + 1))
                print('Iteration: {}'.format(iteration))
                print('Average loss: {}'.format(total_loss / 1000))
                print('{} sec/batch'.format((end_time - start_time) / 1000))
                total_loss = 0
                start_time = time.time()
            iteration += 1
    pkl.dump(embeddings.eval(), open('embeddings.pkl', 'wb'))