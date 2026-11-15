import copy
import unittest

from cloud.aiven_evidence import records


class EvidenceRecordsTest(unittest.TestCase):
    def test_verifier_controls_without_a_model_are_retained(self):
        row = records({'trials': [{'job': 'control', 'trial': 'forged', 'model': None}]})[0]
        self.assertEqual(row[3], 'unattributed/control')
        self.assertIn('"model":null', row[-1])

    def test_errors_are_kept_and_repeat_import_identity_is_stable(self):
        d = {'trials': [{'job': 'job', 'trial': 'failed', 'model': 'm',
                         'exception_type': 'TimeoutError', 'rewards': None,
                         'api_key': 'must-not-export'}]}
        first = records(d)
        self.assertEqual(first, records(d))
        self.assertEqual(len(first), 1)
        self.assertNotIn('must-not-export', first[0][-1])
        changed = copy.deepcopy(d)
        changed['trials'][0]['exception_type'] = 'ConnectionError'
        second = records(changed)
        self.assertEqual(first[0][1], second[0][1])
        self.assertNotEqual(first[0][2], second[0][2])

    def test_image_scores_remain_separate_from_mission_rewards(self):
        d = {'held_out_perception': [{'model': 'm', 'micro_f2': 0.63, 'examples': 182}],
             'frontier_perception': [{'model': 'm', 'micro_f2': 0.99}]}
        result = records(d)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0][0], 'held_out_image')
        self.assertIn('0.63', result[0][-1])


if __name__ == '__main__':
    unittest.main()
