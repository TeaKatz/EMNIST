import tensorflow as tf
import numpy as np
import pandas as pd
from model import Dense


def load_csv(csv_dir):
    csv_read = pd.read_csv(csv_dir, sep=',', dtype='uint8', header=None)
    inputs = np.float32(csv_read.values[:,1:])
    labels = np.uint8(csv_read.values[:,0])
    return inputs, labels

if __name__ == "__main__":
    # Load data
    x_train, y_train = load_csv('../dataset/emnist-mnist-train.csv')
    x_train = np.float32(x_train / 255.0)
    y_train = np.uint8(pd.get_dummies(y_train).values)

    x_dev, y_dev = load_csv('../dataset/emnist-mnist-test.csv')
    x_dev = np.float32(x_dev / 255.0)
    y_dev = np.uint8(pd.get_dummies(y_dev).values)
    '''
    x_test1, y_test1 = load_csv('../dataset/emnist-digits-test.csv')
    x_test1 = np.float32(x_test1 / 255.0)
    y_test1 = np.uint8(pd.get_dummies(y_test1).values)

    x_test2, y_test2 = load_csv('../dataset/emnist-digits-train.csv')
    x_test2 = np.float32(x_test2 / 255.0)
    y_test2 = np.uint8(pd.get_dummies(y_test2).values)
    '''
    # Get dataset information
    print("training set")
    print(x_train.shape)
    print(y_train.shape)
    for i in range(10):
        print("{}: {}".format(i + 1, len(np.where(y_train[:,i]==1)[0])))

    print("testing set")
    print(x_dev.shape)
    print(y_dev.shape)
    for i in range(10):
        print("{}: {}".format(i + 1, len(np.where(y_dev[:, i] == 1)[0])))

    # Build model
    inputs_shape = x_train.shape
    outputs_shape = y_train.shape

    x = tf.placeholder(dtype=tf.float32, shape=[None, inputs_shape[1]])
    y = tf.placeholder(dtype=tf.float32, shape=[None, outputs_shape[1]])

    output = Dense(units=10, name='Dense', kernel_initializer=tf.zeros_initializer(),
                   kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=0.01))(x)

    _, acc_op = tf.metrics.accuracy(labels=tf.arg_max(y, 1), predictions=tf.arg_max(output, 1))

    loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y,
                                                      logits=output)
    #reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    #loss = loss + tf.add_n(reg_losses)

    optimizer = tf.train.AdamOptimizer(0.001)
    train = optimizer.minimize(loss)

    # Make summary
    x_image = tf.reshape(x, [tf.shape(x)[0], 28, 28, 1])
    x_image = tf.image.rot90(x_image, k=3)
    x_image = tf.image.flip_left_right(x_image)
    input_summary = tf.summary.image(name='input_image', tensor=x_image, max_outputs=10)

    with tf.variable_scope("Dense", reuse=True):
        kernel = tf.get_variable(name="kernel")
        kernel_hist = tf.transpose(kernel)
        kernel_hist_summary = tf.summary.histogram(name='kernel_hist', values=kernel_hist)
        kernel_image = tf.reshape(kernel_hist, [tf.shape(kernel_hist)[0], 28, 28, 1])
        kernel_image = tf.image.rot90(kernel_image, k=3)
        kernel_image = tf.image.flip_left_right(kernel_image)
        kernel_image_summary = tf.summary.image(name='kernel_image', tensor=kernel_image, max_outputs=10)

        bias = tf.get_variable(name="bias")
        bias_hist = tf.transpose(bias)
        bias_hist_summary = tf.summary.histogram(name="bias_hist", values=bias_hist)


    loss_avg = tf.reduce_mean(loss)
    loss_summary = tf.summary.scalar(name="loss_avg", tensor=loss_avg)

    acc_summary = tf.summary.scalar(name="acc", tensor=acc_op)

    merged = tf.summary.merge_all()

    # Train moddel
    init = tf.global_variables_initializer()
    init_local = tf.local_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        sess.run(init_local)
        writer_train = tf.summary.FileWriter('../graphs/train/Adam0.001', sess.graph)
        writer_test = tf.summary.FileWriter('../graphs/test/Adam0.001')

        for epoch in range(10000):
            # Run train
            _ = sess.run(train, feed_dict={x: x_train, y: y_train})

            # Monitoring
            if epoch % 10 == 0:
                loss_train, acc_train = sess.run([loss, acc_op], feed_dict={x: x_train, y: y_train})
                loss_test, acc_test = sess.run([loss, acc_op], feed_dict={x: x_dev, y: y_dev})
                print("epoch: {}, loss_train: {:.4f}, loss_test: {:.4f}, acc_train: {:.4f}, acc_test: {:.4f}"
                      .format(epoch, np.average(loss_train), np.average(loss_test), acc_train, acc_test))

            # Run summary
            summary = sess.run(merged, feed_dict={x: x_train, y: y_train})
            writer_train.add_summary(summary, epoch)
            summary = sess.run(loss_summary, feed_dict={x: x_dev, y: y_dev})
            writer_test.add_summary(summary, epoch)


