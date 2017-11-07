import tensorflow as tf
import tensorflow.contrib.rnn as rnn


def rnn_wrapper(inputs,
                cells,
                init_state=None,
                batch_size=None,
                basic_cell=rnn.BasicLSTMCell):
    if batch_size is None:
        batch_size = tf.shape(inputs)[0]

    cells = rnn.MultiRNNCell(
        [basic_cell(c) for c in cells]
    )
    if init_state is None:
        init_state = cells.zero_state(batch_size, tf.float32)

    outputs, final_state = tf.nn.dynamic_rnn(
        cells, inputs,
        initial_state=init_state)
    return cells, init_state, outputs, final_state
