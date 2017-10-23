from tensorflow.contrib.seq2seq import GreedyEmbeddingHelper
from tensorflow.contrib import layers
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops


class GreedyEmbeddingDecisionHelper(GreedyEmbeddingHelper):
    """A helper for use during inference.
    Similar to GreedyEmbeddingHelper but add an extra decision layer
    after rnn_output
    """

    def __init__(self, decision_scope, output_dim, reuse, **kwargs):
        super(GreedyEmbeddingDecisionHelper, self).__init__(**kwargs)
        self._decision_scope = decision_scope
        self._output_dim = output_dim
        self._reuse = reuse

    def sample(self, time, outputs, state, name=None):
        """sample for GreedyEmbeddingHelper."""
        del time, state  # unused by sample_fn
        # Outputs are logits, use argmax to get the most probable id
        if not isinstance(outputs, ops.Tensor):
            raise TypeError("Expected outputs to be a single Tensor, got: %s" %
                            type(outputs))
        # decision layer
        scores = layers.linear(outputs,
                               self._output_dim,
                               scope=self._decision_scope,
                               reuse=self._reuse)
        sample_ids = math_ops.cast(math_ops.argmax(scores, axis=-1),
                                   dtypes.int32)
        return sample_ids
