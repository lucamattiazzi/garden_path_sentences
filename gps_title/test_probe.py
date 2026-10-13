import unittest

from gps_title.probe import word_scores


class WordScoreTests(unittest.TestCase):
    def test_subtokens_are_summed_and_special_tokens_ignored(self):
        result = word_scores(
            "old models", [(0, 0), (0, 3), (3, 6), (6, 10)], [None, -1.0, -2.0, -3.0]
        )
        self.assertEqual([r["word"] for r in result], ["old", "models"])
        self.assertEqual([r["surprisal_nats"] for r in result], [1.0, 5.0])
        self.assertEqual(result[1]["token_indices"], [2, 3])

    def test_rejects_token_spanning_multiple_words(self):
        with self.assertRaises(ValueError):
            word_scores("old man", [(0, 0), (0, 7)], [None, -2.0])

    def test_rejects_unscored_word(self):
        with self.assertRaises(ValueError):
            word_scores("old man", [(0, 0), (0, 3)], [None, -2.0])


if __name__ == "__main__":
    unittest.main()
