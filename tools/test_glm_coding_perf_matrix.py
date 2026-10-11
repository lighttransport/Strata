import unittest
from glm_coding_perf_matrix import FAULT, aggregate, length_summary, validate_configs


class CodingMatrixTest(unittest.TestCase):
    def test_kernel_command_line_is_not_a_vm_fault(self):
        self.assertIsNone(FAULT.search('BOOT_IMAGE=/boot/vmlinuz-7.0 default_hugepagesz=1G'))
        self.assertIsNotNone(FAULT.search('amdgpu: VM fault at address 0x1234'))
        self.assertIsNotNone(FAULT.search('amdgpu: GPU reset begin!'))

    def test_frozen_routing_and_weights_are_enforced(self):
        control=dict(model='model',expert_pack='pack',env={'STRATA_GLM_ROUTE_AFFINITY':'0.08'})
        validate_configs([('control',control),('candidate',{**control,'speculative':'mtp'})])
        with self.assertRaises(ValueError):validate_configs([('control',control),('changed',{**control,'expert_pack':'other'})])
        with self.assertRaises(ValueError):validate_configs([('control',control),('changed',{**control,'env':{'STRATA_GLM_ROUTE_AFFINITY':'0.12'}})])

    def test_speculative_control_is_rejected(self):
        with self.assertRaises(ValueError):
            validate_configs([('control',dict(model='model',speculative='external'))])
        with self.assertRaises(ValueError):
            validate_configs([])

    def test_partial_or_regressed_cells_cannot_meet_targets(self):
        rows=[]
        for config in ['control','candidate']:
            for fixture in ['prime','json_escape']:
                for trial in range(3):
                    rows.append(dict(config=config,fixture=fixture,length=2048,clean=True,
                        prefill_tok_s=800,decode_tok_s=42 if config=='control' or fixture=='prime' else 38,
                        response_seconds=5,output_sha256='same',quality={'status':'passed'}))
        summaries=length_summary(aggregate(rows),'control',{2048:['prime','json_escape']},3)
        candidate=next(r for r in summaries if r['config']=='candidate')
        self.assertFalse(candidate['timing_targets_met'])
        self.assertEqual(candidate['regressions_over_5_percent'],['json_escape'])
        incomplete=length_summary(aggregate(rows[:-1]),'control',{2048:['prime','json_escape']},3)
        self.assertFalse(next(r for r in incomplete if r['config']=='candidate')['complete'])

    def test_dirty_rows_never_enter_medians(self):
        self.assertEqual(aggregate([dict(clean=False)]),[])


if __name__=='__main__':unittest.main()
