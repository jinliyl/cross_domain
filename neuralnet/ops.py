import tensorflow as tf

def highway(input_, size, layer_size=1, bias=-2, f=tf.nn.relu):
    """Highway Network (cf. http://arxiv.org/abs/1505.00387).
  
    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y

    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate.
    """
    output = input_
    for idx in range(layer_size):

        output = f(tf.nn.rnn_cell._linear(output, size, 0, scope='output_lin_%d' % idx))
        transform_gate = tf.sigmoid(tf.nn.rnn_cell._linear(input_, size, 0, scope='transform_lin_%d' % idx) + bias)

        carry_gate = 1. - transform_gate
        output = transform_gate * output + carry_gate * input_

    return output

def last_relevant(output, length):
    batch_size = tf.shape(output)[0]
    max_length = tf.shape(output)[1]
    out_size = int(output.get_shape()[2])
    index = tf.range(0, batch_size) * max_length + (length - 1)
    flat = tf.reshape(output, [-1, out_size])
    relevant = tf.nn.embedding_lookup(flat, index)
    return relevant

