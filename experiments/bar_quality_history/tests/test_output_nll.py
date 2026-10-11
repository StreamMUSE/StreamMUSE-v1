import math
import unittest

from experiments.bar_quality_history.evaluate_output_nll import (
    aggregate, make_batch, prompt_for_target, token_losses,
)


class OutputNLLTests(unittest.TestCase):
    def test_acc_first_no_current_or_future_melody(self):
        acc, mel = [[90 + i, 170] for i in range(4)], [[120 + i, 171] for i in range(4)]
        prefix = prompt_for_target([257, 263, 264], acc, mel, 2)
        self.assertEqual(prefix, [257, 263, 264, 255, 172, 90, 170, 120, 171,
                                  172, 91, 170, 121, 171, 172])
        self.assertNotIn(122, prefix)
        self.assertNotIn(92, prefix)

    def test_window_start_and_bar_markers(self):
        acc, mel = [[i, 170] for i in range(50)], [[200 + i, 171] for i in range(50)]
        prefix = prompt_for_target([257, 263, 264], acc, mel, 36)
        self.assertEqual(prefix[:8], [257, 263, 264, 255, 172, 4, 170, 204])
        self.assertEqual(prefix[-2:], [255, 172])
        self.assertEqual(prefix.count(172), 33)

    def test_labels_only_target_acc(self):
        ids, mask, labels = make_batch([
            {'prompt': [1, 2], 'target': [3, 4]},
            {'prompt': [1], 'target': [169, 170]},
        ], 'cpu')
        self.assertEqual(ids.tolist(), [[1, 2, 3, 4], [1, 169, 170, 258]])
        self.assertEqual(mask.tolist(), [[1, 1, 1, 1], [1, 1, 1, 0]])
        self.assertEqual(labels.tolist(), [[-100, -100, 3, 4], [-100, 169, 170, -100]])

    def test_shift_and_mask_cross_entropy(self):
        import torch
        logits = torch.zeros((1, 4, 5))
        logits[0, 1, 3] = 2
        logits[0, 2, 4] = 2
        labels = torch.tensor([[-100, -100, 3, 4]])
        losses = token_losses(logits, labels)[0].tolist()
        expected = math.log(math.exp(2) + 4) - 2
        self.assertEqual(losses[0], 0)
        self.assertAlmostEqual(losses[1], expected, places=6)
        self.assertAlmostEqual(losses[2], expected, places=6)

    def test_padding_does_not_change_denominator(self):
        import torch
        logits = torch.zeros((2, 4, 200))
        labels = torch.tensor([[-100, -100, 3, 4], [-100, 169, 170, -100]])
        losses = token_losses(logits, labels)
        self.assertAlmostEqual(float(losses.sum()), 4 * math.log(200), places=5)

    def test_aggregate_by_tokens_not_mean_of_beats(self):
        rows = []
        for condition in ('baseline', 'repair'):
            for tick, count, total in ((84, 1, 2.), (88, 9, 0.)):
                rows.append({'piece_id': '01', 'title': 'test', 'seed': 0, 'condition': condition,
                             'tick': tick, 'token_count': count, 'nll_sum': total})
        trials, summary, paired = aggregate(rows)
        self.assertEqual(trials[0]['tokens'], 10)
        self.assertAlmostEqual(summary[0]['token_weighted_NLL'], .2)
        self.assertEqual(paired[0]['mean_NLL_delta'], 0)


if __name__ == '__main__':
    unittest.main()
