import tensorflow.contrib.slim as slim
import tensorflow as tf
import numpy as np


# 网络在实现过程发现，网络的损失有三部分，auxiliary，depthConcat_3，depthConcat_6以及最后输出层。其中auxiliary设置0.3的权重。
# 由于在网络结构在无法获取隐藏层神经元个数，就不在网络中体现。

def depthConcat(inputs, branch_1, branch_2, branch_3, branch_4):
    # 这个BUG折腾了一天！！！
    # with slim.arg_scope([slim.conv2d], padding='SAME', weights_initializer=tf.truncated_normal(stddev=0.1),weights_regulizer=slim.l2_regularizer(0.0005)):
    with slim.arg_scope([slim.conv2d], padding='SAME', weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
                        weights_regularizer=slim.l2_regularizer(0.0005)):
        branch_1 = slim.conv2d(inputs, branch_1[0], [1, 1], scope="branch_1")
        branch_2 = slim.stack(inputs, slim.conv2d, [(branch_2[0], [1, 1]), (branch_2[1], [3, 3])], scope="branch_2")
        branch_3 = slim.stack(inputs, slim.conv2d, [(branch_3[0], [1, 1]), (branch_3[1], [5, 5])], scope="branch_3")
        branch_4_pool = slim.max_pool2d(inputs, [3, 3], stride=1, padding="SAME", scope="branch_4_pool")
        branch_4 = slim.conv2d(branch_4_pool, branch_4[0], [1, 1], scope="branch_4")
        depthConcat_output = tf.concat(values=[branch_1, branch_2, branch_3, branch_4], axis=3)
        return depthConcat_output


def inception_v1(inputs):
    with tf.variable_scope("InceptionV1", ):
        net = slim.conv2d(inputs, 64, [7, 7], 2, scope='conv1')
        net = slim.max_pool2d(net, [3, 3], 2, padding="SAME", scope="pool1")
        net = tf.nn.local_response_normalization(net)
        net = slim.conv2d(net, 64, [1, 1], scope="conv2")
        net = slim.conv2d(net, 192, [3, 3], scope="conv3")
        net = tf.nn.local_response_normalization(net)
        net = slim.max_pool2d(net, [3, 3], 2, padding="SAME", scope="pool2")
        with tf.variable_scope("depthConcat_1"):
            depthConcat_1 = depthConcat(net, [64], [96, 128], [16, 32], [32])
        with tf.variable_scope("dethConcat_2"):
            depthConcat_2 = depthConcat(depthConcat_1, [128], [128, 192], [32, 96], [64])

        net = slim.max_pool2d(depthConcat_2, [3, 3], scope="pool3")

        with tf.variable_scope("depthConcat_3"):
            depthConcat_3 = depthConcat(net, [192], [96, 208], [16, 48], [64])

        with tf.variable_scope("softmax0"):
            average_pool = slim.avg_pool2d(depthConcat_3, [5, 5, ], 3, scope="average_pool")

            conv1 = slim.conv2d(average_pool, 512, [1, 1], padding="SAME")
            # fc1=tf.reshape(conv1,[-1,])

        with tf.variable_scope("depthConcat_4"):
            depthConcat_4 = depthConcat(depthConcat_3, [160], [112, 224], [24, 64], [64])

        with tf.variable_scope("depthConcat_5"):
            depthConcat_5 = depthConcat(depthConcat_4, [128], [128, 256], [24, 64], [64])

        with tf.variable_scope("depthConcat_6"):
            depthConcat_6 = depthConcat(depthConcat_5, [112], [144, 288], [32, 64], [64])

        with tf.variable_scope("depthConcat_7"):
            depthConcat_7 = depthConcat(depthConcat_6, [256], [160, 320], [32, 128], [128])

        net = slim.max_pool2d(depthConcat_7, [3, 3], padding="SAME", scope="pool_4")

        with tf.variable_scope("depthConcat_8"):
            depthConcat_8 = depthConcat(net, [256], [160, 320], [32, 128], [128])
        with tf.variable_scope("depthConcat_9"):
            depthConcat_9 = depthConcat(depthConcat_8, [384], [192, 384], [48, 128], [128])
        net = slim.avg_pool2d(depthConcat_9, kernel_size=[7, 7], stride=1, scope="pool5")
        net = tf.reshape(net, [-1, 1024])
        net = slim.dropout(net, 0.4)
        net = slim.fully_connected(net, 1000, activation_fn=tf.nn.relu, scope="fc1")
        net = slim.fully_connected(net, 1000, activation_fn=tf.nn.softmax, scope="output")
        return net


if __name__ == '__main__':
    x = tf.placeholder(shape=(1, 224, 224, 3), dtype=tf.float32)
    inputs = np.ones(shape=(1, 224, 224, 3), dtype=np.float32)

    net = inception_v1(x)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        print(sess.run(net, feed_dict={x: inputs}))
