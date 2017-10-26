from tensorflow.contrib.seq2seq import GreedyEmbeddingHelper
from tensorflow.contrib import layers
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import control_flow_ops


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



class GreedyEmbeddingDecisionMultiseqHelper(GreedyEmbeddingHelper):
    """A helper for use during inference.
    Similar to GreedyEmbeddingHelper but add an extra decision layer
    after rnn_output

    this one is used in multiple predictions
    """
    def __init__(self, decision_scopes, output_dims, reuse, **kwargs):
        super(GreedyEmbeddingDecisionMultiseqHelper, self).__init__(**kwargs)
        self._decision_scopes = decision_scopes
        self._output_dims = output_dims
        self._reuse = reuse

    def sample(self, time, outputs, state, name=None):
        """sample for GreedyEmbeddingHelper."""
        del time, state  # unused by sample_fn
        # Outputs are logits, use argmax to get the most probable id
        if not isinstance(outputs, ops.Tensor):
            raise TypeError("Expected outputs to be a single Tensor, got: %s" %
                            type(outputs))
        # TODO: 
        #   decode and generate multiple prediction 

        # decision layer
        scores = layers.linear(outputs,
                               self._output_dim,
                               scope=self._decision_scope,
                               reuse=self._reuse)
        sample_ids = math_ops.cast(math_ops.argmax(scores, axis=-1),
                                   dtypes.int32)
        return sample_ids

    def next_inputs(self, time, outputs, state, sample_ids, name=None):
        """next_inputs_fn for GreedyEmbeddingHelper."""
        del time, outputs  # unused by next_inputs_fn
        finished = math_ops.equal(sample_ids, self._end_token)
        all_finished = math_ops.reduce_all(finished)
        next_inputs = control_flow_ops.cond(
            all_finished,
            # If we're finished, the next_inputs value doesn't matter
            lambda: self._start_inputs,
            lambda: self._embedding_fn(sample_ids))
        return (finished, next_inputs, state)
