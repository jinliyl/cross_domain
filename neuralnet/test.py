import tensorflow as tf
import numpy

input1 = tf.placeholder(tf.float32, [2, 2])
input2 = tf.placeholder(tf.float32, [2, 2])
output = tf.div(input1, input2)

with tf.Session() as sess:
  print(sess.run([output], feed_dict={input1:[[1, 2], [3, 4]], input2:[[5, 6], [7, 8]]}))
