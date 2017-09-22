from tensorflow.examples.tutorials.mnist import input_data
from nn import NN
import numpy as np

mnist = input_data.read_data_sets('./data', one_hot=True)

MAX_ITERATORS = 3000
nn = NN([784, 10], learning_rate=5e-1)
nn.restore()
for epoch in range(1, MAX_ITERATORS + 1):
    x, y = mnist.train.next_batch(100)
    nn.fit(x, y)
    prediction = np.argmax(nn.predict(mnist.test.images), 1)
    correct = np.equal(prediction, np.argmax(mnist.test.labels, 1)).astype(float)
    accuracy = np.mean(correct)
    print 'Epoch %s, Accuracy is %s' % (epoch, accuracy)
    if epoch % 50 == 0 or epoch == MAX_ITERATORS:
        nn.save()
