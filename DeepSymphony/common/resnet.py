import tensorflow.contrib.slim as slim
import tensorflow as tf


class ResNetBuilder(object):
    def __init__(self, is_train, bn_scopes, bn_scope):
        self.is_train = is_train
        self.bn_scope = bn_scope
        self.bn_scopes = bn_scopes

    def resnet(self, input,
               structure=[2, 2, 2, 2],
               filters=64, nb_class=10):
        with tf.variable_scope('conv_pool_1'):
            conv1 = self.conv_bn_relu(input, 64, (7, 7), stride=(2, 2))
            pool1 = slim.max_pool2d(conv1, [3, 3],
                                    stride=2, padding='SAME', scope='pool2')

        block = pool1
        filters = filters
        with tf.variable_scope('residual'):
            for i, r in enumerate(structure):
                with tf.variable_scope('residual_block{}'.format(i)):
                    block = self.residual_block(
                        block, filters=filters, repetitions=r,
                        is_first_layer=(i == 0)
                    )
                filters *= 2
            block = self.bn_relu(block)

        block_shape = block.shape
        with tf.variable_scope('avr_pool'):
            pool2 = slim.avg_pool2d(block,
                                    (block_shape[1], block_shape[2]),
                                    stride=1)

        with tf.variable_scope('flatten'):
            flat = slim.flatten(pool2)
        with tf.variable_scope('classify_layer'):
            dense = slim.fully_connected(flat, nb_class, activation_fn=None)
        return dense

    def bn_relu(self, input):
        # if self.bn_scope == self.bn_scopes[0]:
        #     # the first one is responsible to create all tensors
        #     for s in self.bn_scopes:
        #         with tf.variable_scope(s):
        #             slim.batch_norm(input,
        #                             center=True,
        #                             scale=True,
        #                             is_training=self.is_train)

        # with tf.variable_scope(self.bn_scope, reuse=True):
        #     bn = slim.batch_norm(input,
        #                          center=True,
        #                          scale=True,
        #                          is_training=self.is_train)
        relu = tf.nn.relu(input)
        return relu

    def bn_relu_conv(self, input, filters, kernel, stride=(1, 1),
                     padding='SAME', reg=None):
        act = self.bn_relu(input)
        init = tf.contrib.layers.xavier_initializer(uniform=False)
        conv = slim.conv2d(act, filters, kernel, stride=stride,
                           padding=padding,
                           weights_initializer=init,
                           weights_regularizer=slim.l2_regularizer(1e-4)
                           if reg is None else reg)
        return conv

    def conv_bn_relu(self, input, filters, kernel, stride=(1, 1),
                     padding='SAME', reg=None):
        init = tf.contrib.layers.xavier_initializer(uniform=False)
        reg = slim.l2_regularizer(1e-4) if reg is None else reg
        conv = slim.conv2d(input, filters, kernel,
                           stride=stride, padding=padding,
                           weights_initializer=init,
                           weights_regularizer=reg)
        act = self.bn_relu(conv)
        return act

    def shortcut(self, input, residual):
        input_shape = input.shape
        residual_shape = residual.shape
        stride_width = int(input_shape[1] / residual_shape[1])
        stride_height = int(input_shape[2] / residual_shape[2])
        equal_channels = input_shape[3] == residual_shape[3]

        shortcut = input
        if stride_width > 1 or stride_height > 1 or not equal_channels:
            with tf.variable_scope('conv2d'):
                shortcut = slim.conv2d(input,
                                       residual_shape[3], (1, 1),
                                       stride=(stride_width, stride_height),
                                       padding="VALID",
                                       weights_regularizer=slim.
                                       l2_regularizer(1e-4))
        with tf.variable_scope('residual_sum'):
            return shortcut + residual

    def basic_block(self, input, filters, init_strides=(1, 1),
                    is_first_block_of_first_layer=False):
        with tf.variable_scope('conv1'):
            if is_first_block_of_first_layer:
                reg = slim.l2_regularizer(1e-4)
                conv1 = slim.conv2d(input,
                                    filters, (3, 3),
                                    stride=init_strides, padding="SAME",
                                    weights_regularizer=reg)
            else:
                conv1 = self.bn_relu_conv(input,
                                          filters=filters,
                                          kernel=(3, 3),
                                          stride=init_strides)
        with tf.variable_scope('conv2'):
            residual = self.bn_relu_conv(conv1, filters, (3, 3))
        with tf.variable_scope('shortcut'):
            return self.shortcut(input, residual)

    def residual_block(self, input, filters, repetitions,
                       is_first_layer=False):
        for ind, r in enumerate(range(repetitions)):
            init_strides = (1, 1)
            if r == 0 and not is_first_layer:
                init_strides = (2, 2)
            with tf.variable_scope('basic_block_{}'.format(ind)):
                flag = (is_first_layer and r == 0)
                input = self.basic_block(
                    input,
                    filters,
                    init_strides,
                    is_first_block_of_first_layer=flag
                )
        return input
