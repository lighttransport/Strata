"""Report gates must retain failures and distinguish incomplete timing matrices."""
import json
from pathlib import Path
import tempfile
import unittest
from glm_tr16_redesign_report import summarize, shape_validation


class ReportTests(unittest.TestCase):
    def row(self, case='hip', size='8k', prompt='train-code'):
        trial=dict(tokens=512,milliseconds=16000,tokens_per_second=32)
        return dict(case=case,prompt=f'prompts/{size}/{prompt}.ids',repetition=0,
                    returncode=0,elapsed_seconds=100,measurement=dict(complete=True,clean=True,
                    measurements=dict(prefill=[trial]*3,decode=[trial]*3,decode_warmup=[trial],
                                      median_decode_tok_s=32,median_prefill_tok_s=32)))

    def report(self, rows):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'results.json'
            path.write_text(json.dumps(dict(results=rows,executables={},config={})))
            return summarize(path)

    def test_full_matrix(self):
        rows=[self.row(c,s,p) for c in ('cuda','hip','hip-cuda')
              for s in ('2k','8k') for p in ('train-code','holdout-chat')]
        result=self.report(rows)
        self.assertTrue(result['complete_matrix'])
        self.assertTrue(shape_validation(result)['calibrated'])
        self.assertTrue(all(not x['quality_qualified'] for x in result['rows']))

    def test_partial_not_calibrated(self):
        result=self.report([self.row(),self.row(prompt='holdout-chat')])
        self.assertFalse(result['complete_matrix'])
        self.assertFalse(shape_validation(result)['calibrated'])

    def test_rejected_rows(self):
        rows=[self.row() for _ in range(4)]
        rows[0]['returncode']=1
        rows[1]['instrumented']=True
        rows[2]['measurement']['clean']=False
        rows[3]['measurement']['complete']=False
        result=self.report(rows)
        self.assertEqual(result['rows'],[])
        self.assertEqual(len(result['excluded']),4)


if __name__=='__main__':
    unittest.main()
