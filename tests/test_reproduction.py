"""Simple checks for the reproduced equations and optional rating summary."""
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from mappings import all_mappings, adv, inv_adv
from extended_sensitivity_experiment import EPS_TABLE, debiased_count
from dataset_ratings_summary import run as run_dataset

class TestReproduction(unittest.TestCase):
    def test_adv_inverse(self):
        for eps in (0.1, 0.5, 1.0, 2.0, 5.0):
            self.assertAlmostEqual(inv_adv(adv(eps)), eps, places=10)

    def test_endpoints_and_monotone(self):
        for method in ('linear', 'exponential', 'sequential', 'sigmoid'):
            vals = [all_mappings(i)[method] for i in range(1, 8)]
            self.assertAlmostEqual(vals[0], 0.1)
            self.assertAlmostEqual(vals[-1], 5.0)
            self.assertTrue(np.all(np.diff(vals) >= -1e-12))
            np.testing.assert_allclose(vals, EPS_TABLE[method], rtol=0, atol=1e-12)

    def test_unbiased_expectation_without_sampling(self):
        eps=np.array([0.1, 1.0, 3.0, 5.0])
        bits=np.array([0.,1.,1.,0.])
        p=1/(1+np.exp(-eps));q=1-p
        # Expected report: q + (p-q)*bit, fed into the debiasing formula.
        expected_reports=q+(p-q)*bits
        self.assertAlmostEqual(debiased_count(expected_reports, eps),bits.sum(),places=10)

    def test_optional_real_dataset_input_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            inp=Path(tmp)/'sample.csv';out=Path(tmp)/'out'
            pd.DataFrame({'rating':[1,7,2,None,8,3.5], 'category':['a']*6}).to_csv(inp,index=False)
            run_dataset(inp,'rating','category',out)
            r=pd.read_csv(out/'real_dataset_rating_counts.csv')
            self.assertEqual(int(r['count'].sum()),3)
            self.assertTrue((out/'real_dataset_group_summary.csv').exists())

if __name__=='__main__':unittest.main()
