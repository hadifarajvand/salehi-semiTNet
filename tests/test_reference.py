import json, unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
class ReferenceTests(unittest.TestCase):
    def test_targets(self):
        d=json.loads((ROOT/'paper/reference/paper_targets.json').read_text())
        self.assertEqual(d['dataset']['total'],16317)
        self.assertEqual(d['training']['iterations'],26250)
        self.assertAlmostEqual(d['overall']['dice'],95.45)
    def test_patch_config(self):
        cfg=(ROOT/'configs/paper.yaml').read_text()
        self.assertIn('num_classes: 32',cfg)
        self.assertIn('max_iter: 26250',cfg)
if __name__=='__main__': unittest.main()
