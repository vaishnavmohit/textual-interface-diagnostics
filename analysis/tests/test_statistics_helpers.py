import numpy as np
import pandas as pd
import unittest

from analysis.statistics_helpers import (
    cluster_randomization_pvalue,
    holm,
    paired_cluster_test,
)


class StatisticsHelpersTest(unittest.TestCase):
    def test_cluster_randomization_null_is_one(self):
        differences = np.array([1.0, -1.0, 1.0, -1.0])
        clusters = np.array(["a", "a", "b", "b"])
        self.assertEqual(cluster_randomization_pvalue(differences, clusters, 1_000), 1.0)

    def test_paired_test_flips_whole_support_sets(self):
        a = pd.DataFrame(
            {"case": ["a+", "a-", "b+", "b-"], "uid": ["a", "a", "b", "b"], "ok": [0, 0, 0, 0]}
        )
        b = pd.DataFrame(
            {"case": ["a+", "a-", "b+", "b-"], "uid": ["a", "a", "b", "b"], "ok": [1, 1, 1, 1]}
        )
        result = paired_cluster_test(a, b, "ok", "case", "uid", n_resamples=10_000)
        self.assertEqual(result["n_pairs"], 4)
        self.assertEqual(result["n_clusters"], 2)
        self.assertEqual(result["b01"], 4)
        self.assertTrue(0.4 < result["pvalue"] < 0.6)

    def test_holm_is_monotone_in_sorted_order(self):
        p = np.array([0.01, 0.04, 0.03])
        adjusted = holm(p)
        order = np.argsort(p)
        self.assertTrue(np.all(np.diff(adjusted[order]) >= 0))
        self.assertTrue(np.all(adjusted >= p))


if __name__ == "__main__":
    unittest.main()
