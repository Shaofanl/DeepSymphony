# internal imports
import tensorflow as tf
import tensorflow.contrib.rnn as rnn
from DeepSymphony.common.HParam import HParam


class StackedLSTMHParam(HParam):
    cells = [256, 256, 256]
    basic_cell = rnn.BasicLSTMCell
    lr = 1e-3
    clip_norm = 3.
    batch_size = 32
    iterations = 300

    def __init__(self, **kwargs):
        super(StackedLSTMHParam, self).__init__(**kwargs)

        assert(hasattr(self, 'timesteps'))
        assert(hasattr(self, 'input_dim'))
        assert(hasattr(self, 'output_dim'))


class StackedLSTM(object):
    def __init__(self, hparam=None):
        self.built = False
        if hparam is None:
            hparam = StackedLSTMHParam()
        self.hparam = hparam

    def build(self):
        hparam = self.hparam
        x = tf.placeholder("float", [hparam.batch_size,
                                     hparam.timesteps,
                                     hparam.input_dim])
        y = tf.placeholder("int32", [hparam.batch_size,
                                     hparam.timesteps])

        with tf.variable_scope("LSTM"):
            cells = [hparam.basic_cell(c) for c in hparam.cells]
            cells = rnn.MultiRNNCell(cells)
            init_state = cells.zero_state(hparam.batch_size, tf.float32)

            outputs, final_state = tf.nn.dynamic_rnn(
                cells, x,
                initial_state=init_state,
                swap_memory=True)

        with tf.variable_scope("top"):
            scores = tf.contrib.layers.linear(outputs,
                                              hparam.output_dim)

            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=y, logits=scores)
            loss = tf.reduce_mean(tf.reduce_sum(loss, axis=1))

        optimizer = tf.train.AdamOptimizer(learning_rate=hparam.lr)
        train_op = tf.contrib.slim.learning.create_train_op(
            loss, optimizer, clip_gradient_norm=hparam.clip_norm)
        tf.add_to_collection('train_op', train_op)
        tf.summary.scalar('loss', loss)

        self.train_op = train_op
        self.loss = loss
        self.x, self.y = x, y
        self.built = True

    def train(self, fetch_data):
        if self.built is False:
            self.build()
        hparam = self.hparam
        # train_writer = tf.summary.FileWriter(FLAGS.summaries_dir + '/train',
        # sess.graph)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            i = 0
            while i < hparam.iterations:
                xi, yi = fetch_data(hparam.batch_size)

                _, loss = sess.run([self.train_op, self.loss],
                                   feed_dict={self.x: xi,
                                              self.y: yi})
                print('Step %d: loss = %.2f' % (i, loss))
                i += 1

            saver = tf.train.Saver()
            saver.save(sess, './temp/StackedLSTM.ckpt')
