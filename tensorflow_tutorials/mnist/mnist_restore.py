import tensorflow as tf
from tensorflow_tutorials.mnist import mnistdata

mnist = mnistdata.read_data_sets("data", one_hot=True, reshape=False)

batch_size = 100
n_batch = mnist.train.images.shape[0] // batch_size

X = tf.placeholder(tf.float32, [None, 28, 28, 1])
Y_ = tf.placeholder(tf.float32, [None, 10])

# weight
W = tf.Variable(tf.zeros([784, 10]))
# bias
b = tf.Variable(tf.zeros([10]))

XX = tf.reshape(X, [-1, 784])

# The model
Y = tf.nn.softmax(tf.matmul(XX, W) + b)
cross_entropy = -tf.reduce_mean(Y_ * tf.log(Y)) * 1000.0
correct_prediction = tf.equal(tf.argmax(Y, 1), tf.argmax(Y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

optimizer = tf.train.GradientDescentOptimizer(0.003)
train_step = optimizer.minimize(cross_entropy)

saver = tf.train.Saver()

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run(accuracy, feed_dict={X: mnist.test.images, Y_: mnist.test.labels}))
    saver.restore(sess, "./data/model/my_net.ckpt")
    print(sess.run(accuracy, feed_dict={X: mnist.test.images, Y_: mnist.test.labels}))
