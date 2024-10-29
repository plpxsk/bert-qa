import tensorflow as tf
import numpy as np


def compute_official_bert_squad_loss(start_logits, end_logits, start_positions, end_positions,
                                     seq_length):
    """
    Squad Loss from official BERT implementation

    With:
    seq_length = modeling.get_shape_list(input_ids)[1]
    start_positions = features["start_positions"]
    end_positions = features["end_positions"]

    Source:
    https://github.com/google-research/bert/blob/eedf5716ce1268e56f0a50264a88cafad334ac61/run_squad.py#L646C11-L646C18
    """

    def compute_loss(logits, positions):
        one_hot_positions = tf.one_hot(
            positions, depth=seq_length, dtype=tf.float32)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        loss = -tf.reduce_mean(
            tf.reduce_sum(one_hot_positions * log_probs, axis=-1))
        return loss

    start_positions = start_positions
    end_positions = end_positions

    start_loss = compute_loss(start_logits, start_positions)
    end_loss = compute_loss(end_logits, end_positions)

    total_loss = (start_loss + end_loss) / 2.0

    return total_loss


def mx_to_tf(x):
    """https://ml-explore.github.io/mlx/build/html/usage/numpy.html#tensorflow"""
    if len(x.shape) > 1:
        return tf.convert_to_tensor(np.array(x))
    else:
        return tf.constant(memoryview(x))
