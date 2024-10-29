import unittest
import mlx.core as mx
import numpy as np

from qa import compute_loss
from deps.loss import compute_official_bert_squad_loss, mx_to_tf


class TestUtils(unittest.TestCase):

    def test_loss(self):
        start_logits = mx.array([[-6.085708, -1.045586, 0.4683273, 4.34371, -3.8366804]])
        end_logits = mx.array([[-7.80379, -5.8751764, -1.3431073, -2.0616195, 6.35468]])
        start_position = mx.array([2])
        end_position = mx.array([4])

        loss = compute_loss(start_logits, end_logits, start_position, end_position,
                            reduction="none")

        # check official loss in tensorflow
        start_logits, end_logits, start_position, end_position = map(
            mx_to_tf,
            (start_logits, end_logits, start_position, end_position)
        )
        seq_length = start_logits.shape[1]
        official_loss = compute_official_bert_squad_loss(start_logits, end_logits,
                                                         start_position, end_position, seq_length)

        self.assertTrue(
            np.allclose(np.array(loss), np.array(official_loss))
        )
