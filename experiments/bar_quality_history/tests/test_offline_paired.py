import unittest

from experiments.bar_quality_history.run_offline_paired import compare, increment, FIELDS


def rows():
    return [{'tick': t, 'metadata': {f: f + str(t) for f in FIELDS}} for t in (32, 80, 84, 88)]


class OfflinePairTests(unittest.TestCase):
    def test_half_open_causal_increment(self):
        events = [{'tick': t} for t in (31, 32, 35, 36)]
        self.assertEqual(increment(events, 32, 36), [{'tick': 32}, {'tick': 35}])
        increment(events, 32, 36)[0]['tick'] = -1
        self.assertEqual(events[1]['tick'], 32)

    def test_equal(self):
        self.assertTrue(all(r['equal'] == 4 for r in compare(rows(), rows()).values()))

    def test_later_acc_difference_allowed(self):
        b = rows()
        b[2]['metadata']['prompt_token_digest'] = 'changed'
        self.assertEqual(compare(rows(), b)['prompt_token_digest']['different_ticks'], [84])

    def test_early_difference_rejected(self):
        b = rows()
        b[1]['metadata']['raw_token_digest'] = 'changed'
        with self.assertRaisesRegex(AssertionError, 'before intervention'):
            compare(rows(), b)

    def test_melody_difference_rejected(self):
        b = rows()
        b[-1]['metadata']['part0_token_digest'] = 'changed'
        with self.assertRaisesRegex(AssertionError, 'Melody changed'):
            compare(rows(), b)

    def test_missing_trace_rejected(self):
        b = rows()
        b[0]['metadata']['raw_token_digest'] = None
        with self.assertRaisesRegex(AssertionError, 'Missing model trace'):
            compare(rows(), b)


if __name__ == '__main__':
    unittest.main()
