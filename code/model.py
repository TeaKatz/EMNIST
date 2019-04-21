import tensorflow as tf


class Dense:
    def __init__(self, units,
                 kernel_initializer=tf.truncated_normal_initializer(),
                 bias_initializer=tf.zeros_initializer(),
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 name='Dense'):
        self.units = units
        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer
        self.kernel_regularizer = kernel_regularizer
        self.bias_regularizer = bias_regularizer
        self.name = name
        self.kernel = None
        self.bias = None
        self.inputs = None
        self.builded = False

    def __call__(self, inputs):
        self.inputs = inputs
        with tf.variable_scope(self.name):
            if not self.builded:
                self.build()
                self.builded = True

            outputs = tf.add(tf.matmul(self.inputs, self.kernel), self.bias)
        return outputs


    def build(self):
        self.kernel = tf.get_variable(name='kernel', shape=[self.inputs.shape[1], self.units],
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer)

        self.bias = tf.get_variable(name='bias', shape=[1, self.units],
                                      initializer=self.bias_initializer,
                                      regularizer=self.bias_regularizer)


if __name__ == "__main__":
    import numpy as np

    x = tf.placeholder(dtype=tf.float32, shape=[None, 784])

    output = Dense(units=32, name='Dense1')(x)
    output = tf.nn.relu(output)
    output = Dense(units=32, name='Dense2')(output)

    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=np.random.rand(10, 10),
                                                      logits=output)

    optimizer = tf.train.AdamOptimizer()
    train = optimizer.minimize(loss)

    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        writer = tf.summary.FileWriter('../graphs', sess.graph)

        sess.run(init)
        value = sess.run(output, feed_dict={x: np.random.rand(10, 784)})
        print(value.shape)

