import json
from pathlib import Path
import tempfile
import unittest
from glm_current_sim_records import import_run


class ImportCurrentRun(unittest.TestCase):
    def record(self):
        return dict(clean=True, complete=True, exit_code=0, rejected=None,
                    config=dict(env={'STRATA_GLM_BF16_WMMA':'1','STRATA_GLM_F16_PREFILL_MIN_TOKENS':'4096'},
                                prefill_experts='bf16-batched',threads=12,context=8192,prefill_batch=8192,
                                decode_cache_mib=12288,gpu_budget_mib=15360),
                    measurements=dict(prefill=[dict(tokens=7936)],decode=[dict(tokens=127,kind='single')],
                                      median_decode_tok_s=35.,median_prefill_tok_s=465.))

    def read(self, row, **kwargs):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'run.json';p.write_text(json.dumps(row))
            return import_run(p,'hardware.json',**kwargs)

    def test_backend_and_forecast_are_explicit(self):
        rows=self.read(self.record(),fit=True)
        self.assertEqual(rows[0]['config']['prefill_backend'],'bf16-wmma')
        self.assertEqual(rows[0]['config']['pack'],'reap50_q22')
        self.assertNotIn('measured_hit_share',rows[0]['config'])
        self.assertNotIn('acceptance_probabilities',rows[0]['config'])

    def test_short_prompt_falls_back_to_mmq(self):
        row=self.record();row['measurements']['prefill'][0]['tokens']=2048
        self.assertEqual(self.read(row)[0]['config']['prefill_backend'],'mmq')

    def test_instrumented_data_cannot_train(self):
        row=self.record();row['config']['env']['STRATA_GLM_STEP_TRACE']='1'
        with self.assertRaises(ValueError):self.read(row,fit=True)
        self.assertFalse(self.read(row,components=True)[0]['fit'])
        with self.assertRaises(ValueError):self.read(row,components=True,fit=True)

    def test_heldout_routing_cannot_leak_into_fit(self):
        with self.assertRaises(ValueError):self.read(self.record(),observed=True,fit=True)

    def test_fault_traces_cannot_be_used_as_performance_training(self):
        for name,value in [('STRATA_GLM_TASK_TRACE','/tmp/engine.jsonl'),
                           ('STRATA_GLM_TASK_TRACE_SYNC','1'),
                           ('STRATA_GLM_STATE_TRACE_DIR','/tmp/layers')]:
            row=self.record();row['config']['env'][name]=value
            with self.assertRaisesRegex(ValueError,'instrumented'):
                self.read(row,fit=True)
            self.assertFalse(self.read(row,components=True)[0]['fit'])

    def test_incomplete_or_dirty_runs_are_rejected(self):
        for key,value in [('complete',False),('clean',False),('exit_code',1),('rejected','GPU probe failure')]:
            row=self.record();row[key]=value
            with self.assertRaises(ValueError):self.read(row)

    def test_library_backend_not_silently_modelled_as_wmma(self):
        row=self.record();row['config']['env'].pop('STRATA_GLM_BF16_WMMA')
        with self.assertRaises(ValueError):self.read(row)

    def test_historical_acceptance_cannot_train_current_speculation(self):
        row=self.record();row['config']['speculative']='mtp'
        with self.assertRaises(ValueError):self.read(row,fit=True)
        with self.assertRaises(ValueError):self.read(row,observed=True)



if __name__=='__main__':unittest.main()
