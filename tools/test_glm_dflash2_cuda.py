"""Validate the optional V100 FP32-output GEMM helper (no model required)."""
import argparse
import json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library', required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--skinny-library')
    p.add_argument('--graph', action='store_true', help='check tiny draft graph replay, moving positions and short-window fallback')
    a = p.parse_args()
    import torch
    torch.cuda.set_device(a.device)
    torch.ops.load_library(a.library)
    if a.skinny_library:torch.ops.load_library(a.skinny_library)
    torch.manual_seed(17)
    records = []
    stream = torch.cuda.Stream(device=a.device)
    with torch.inference_mode(), torch.cuda.stream(stream):
        for rows, width, columns in [(1,256,128),(7,4096,4096),(128,20480,4096),(8,12288,4096),(8,20480,4096),(8,4096,6144),(8,4096,12288)]:
            x = torch.randn(rows,width,device=a.device,dtype=torch.float16)*.05
            w = torch.randn(columns,width,device=a.device,dtype=torch.float16)*.05
            actual = torch.ops.strata_dflash2.linear_f32(x,w)
            expected = torch.nn.functional.linear(x.float(),w.float())
            if a.skinny_library and rows<=8:
                packed=torch.ops.strata_dflash2_sm70.pack(w)
                skinny=torch.ops.strata_dflash2_sm70.linear(x,packed)
                torch.testing.assert_close(skinny,expected,atol=5e-4,rtol=1e-3)

            assert actual.dtype == torch.float32
            torch.testing.assert_close(actual,expected,atol=5e-4,rtol=1e-3)
            records.append(dict(rows=rows,width=width,columns=columns,max_abs_error=float((actual-expected).abs().max())))
        if a.skinny_library:
            n,k=64,128
            weight=torch.randn(n,k,device=a.device,dtype=torch.float16)
            packed=torch.ops.strata_dflash2_sm70.pack(weight).cpu()
            row=torch.arange(n).repeat_interleave(k);column=torch.arange(k).repeat(n)
            lane=(((row&31)>>3)<<2)+(row&3)+torch.where((row&4)!=0,16,0)
            index=((row//32*(k//16)+column//16)*32+lane)*16+column%16
            expected=torch.empty(n*k,dtype=torch.float16);expected[index]=weight.cpu().reshape(-1)
            assert torch.equal(packed.view(torch.int16).reshape(-1),expected.view(torch.int16))
            x=torch.full((7,64),200.,device=a.device,dtype=torch.float16)
            w=torch.full((32,64),20.,device=a.device,dtype=torch.float16)
            result=torch.ops.strata_dflash2_sm70.linear(x,torch.ops.strata_dflash2_sm70.pack(w))
            assert torch.equal(result,torch.full_like(result,256000.))
        x = torch.full((7,64),200.,device=a.device,dtype=torch.float16)
        w = torch.full((16,64),20.,device=a.device,dtype=torch.float16)
        actual = torch.ops.strata_dflash2.linear_f32(x,w)
        assert torch.equal(actual,torch.full_like(actual,256000.))
        for bad_x, bad_w in [(x.float(),w),(x[:0],w),(x,w[:,:32]),(x.T,w)]:
            try:
                torch.ops.strata_dflash2.linear_f32(bad_x,bad_w)
            except RuntimeError:
                pass
            else:
                raise AssertionError('invalid GEMM operand accepted')
    stream.synchronize()
    graph_checks=[]
    if a.graph:
        from test_glm_dflash2_worker import DFlash2Test
        model=DFlash2Test().tiny()
        model.device=a.device;model.gemm_f32=True
        model.w={n:v.to(device=a.device,dtype=torch.float16) for n,v in model.w.items()}
        model.qkv=[v.to(device=a.device,dtype=torch.float16) for v in model.qkv]
        model.inv=model.inv.to(a.device)
        features=torch.randn(8,80,device=a.device)*.02
        embedding=torch.randn(8,16,device=a.device)*.02
        model.append(features,0,8)
        reference=model.forward(embedding).clone()
        model.enable_cuda_graph(embedding)
        def compare(name,reference):
            actual=model.forward(embedding)
            torch.testing.assert_close(actual,reference,atol=1e-4,rtol=1e-4)
            graph_checks.append(dict(name=name,max_abs_error=float((actual-reference).abs().max())))
        def eager():
            graph=model.graph;model.graph=None
            try:return model.forward(embedding).clone()
            finally:model.graph=graph
        compare('initial',reference)
        model.append(features[:2],8,10);embedding.mul_(.7)
        compare('increment_and_moving_positions',eager())
        model.append(features[:4],0,4)
        compare('short_reset_eager_fallback',eager())
        model.append(features,120,128)
        compare('full_reset_distant_positions',eager())
    print(json.dumps(dict(passed=True,gpu=torch.cuda.get_device_name(),shapes=records,
                          output_above_fp16_range=256000,nondefault_stream=True,skinny_checked=bool(a.skinny_library),graph_checks=graph_checks)))


if __name__ == '__main__':main()
