#!/usr/bin/env python3
"""Small structural checks for unquantized export and paper-guided rotation."""
from pathlib import Path
import json
import hashlib
import random
import struct
import subprocess
import sys
import tempfile

from gyro_study import rotate, quantize, metadata_quanta
from activation_trace import read_trace, vector_key
from fixture_validation import verify_fixture


def main():
    root = Path(__file__).resolve().parent
    with tempfile.TemporaryDirectory(prefix="strata-quant-export-") as tmp:
        p = Path(tmp)
        source = p / "inputs"; source.mkdir()
        for i, name in enumerate(("calibration-large.f32", "validation.f32", "heldout.f32", "activation.f32")):
            r = random.Random(i)
            values = [r.uniform(-1,1) for _ in range(256)]
            (source/name).write_bytes(struct.pack("<256f", *values))
        for dtype, kind in (("f",0),("e",1),("bf16",30)):
            name = b"layer.weight"
            header = struct.pack("<IIQQ", 0x46554747,3,1,0)
            header += struct.pack("<Q",len(name))+name+struct.pack("<IQQIQ",2,256,16,kind,0)
            header += bytes((-len(header))%32)
            vals = [float((r*256+c)%31-15) for r in range(16) for c in range(256)]
            if dtype == "bf16":
                payload = b"".join(struct.pack("<H",struct.unpack("<I",struct.pack("<f",v))[0]>>16) for v in vals)
            else:
                payload = struct.pack("<"+dtype*len(vals),*vals)
            model=p/(dtype+".gguf"); model.write_bytes(header+payload)
            out=p/(dtype+"-out")
            cmd=[sys.executable,str(root/"export_layer.py"),"--model",str(model),"--tensor","layer.weight",
                 "--inputs",str(source),"--input-source","synthetic","--cols","128","--col-start","128","--output",str(out)]
            subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL)
            actual=list(struct.unpack("<2048f",(out/"weights.f32").read_bytes()))
            expected=[v for r in range(16) for v in vals[r*256+128:r*256+256]]
            assert actual==expected
            assert json.loads((out/"fixture.json").read_text())["input_source"]=="synthetic"
            model.write_bytes(header+payload[:-1])
            assert subprocess.run(cmd,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode!=0
        # Harmonic slots are constant-sign within each quantization group.
        for width, group in ((32,8),(128,32),(256,32)):
            for slot in range(0,width,group):
                impulse=[0.0]*width; impulse[slot]=1.0
                transformed=rotate(impulse,width)
                assert all(len(set(transformed[i:i+group]))==1 for i in range(0,width,group))
        frozen = metadata_quanta([[-1.0,1.0]*64],128,True)
        overflow = quantize([[-100.0,100.0]*64],128,True,"int8",frozen)
        assert max(map(abs,overflow[0]))<5, "calibration metadata ranges must saturate on heldout overflow"
        (source/"heldout.f32").write_bytes((source/"calibration-large.f32").read_bytes())
        model.write_bytes(header+payload)
        assert subprocess.run(cmd,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode!=0
        # Distinct full-width vectors become identical in the selected crop.
        for i,name in enumerate(("calibration-large.f32","validation.f32","heldout.f32")):
            (source/name).write_bytes(struct.pack("<256f",*([float(i+1)]*128+[0.0]*128)))
        assert subprocess.run(cmd,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode!=0
        assert vector_key(struct.pack("<f",0.0))==vector_key(struct.pack("<f",-0.0))
        # A nonfinite channel outside the crop must still be rejected.
        for i,name in enumerate(("calibration-large.f32","validation.f32","heldout.f32")):
            (source/name).write_bytes(struct.pack("<256f",*([float(i+1)]*256)))
        bad=[1.0]*256;bad[0]=float("nan")
        (source/"calibration-large.f32").write_bytes(struct.pack("<256f",*bad))
        assert subprocess.run(cmd,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode!=0
        def record(layer,position,seed):
            return (struct.pack("<6I",0x31435247,layer,position,2,4096,8)+
                    struct.pack("<16i",*(list(range(8))*2))+
                    struct.pack("<8192f",*([float(seed)]*4096+[float(seed+1)]*4096)))
        traces=[]
        for seed in (10,20,30):
            path=p/(str(seed)+".capture")
            path.write_bytes(record(11,0,seed+100)+record(10,0,seed))
            traces.append(path)
        selected,info=read_trace(traces[0],10,144,0)
        assert len(selected)==2 and info["selected"][0]["record"]==1
        convert=[sys.executable,str(root/"activation_trace.py"),"--train-trace",str(traces[0]),
                 "--validation-trace",str(traces[1]),"--heldout-trace",str(traces[2]),
                 "--layer","10","--experts","144","--output",str(p/"captured")]
        subprocess.run(convert,check=True,stdout=subprocess.DEVNULL)
        capture_meta=json.loads((p/"captured/activation-provenance.json").read_text())
        assert capture_meta["counts"]=={"train":2,"validation":2,"heldout":2}
        # Verify a complete converted fixture, then reject a changed input file.
        fixture=p/"captured"
        (fixture/"weights.f32").write_bytes(struct.pack("<65536f",*([0.0]*65536)))
        hashes={name:hashlib.sha256((fixture/name).read_bytes()).hexdigest()
                for name in ("weights.f32","calibration-large.f32","validation.f32","heldout.f32","activation.f32")}
        (fixture/"fixture.json").write_text(json.dumps({"input":4096,"output":16,"sha256":hashes}))
        verify_fixture(fixture)
        router_name=b"blk.10.ffn_gate_inp.weight"
        router_header=struct.pack("<IIQQ",0x46554747,3,1,0)
        router_header+=struct.pack("<Q",len(router_name))+router_name+struct.pack("<IQQIQ",2,4096,16,0,0)
        router_header+=bytes((-len(router_header))%32)
        router_model=p/"router.gguf";router_model.write_bytes(router_header+bytes(16*4096*4))
        captured_export=[sys.executable,str(root/"export_layer.py"),"--model",str(router_model),
                         "--tensor",router_name.decode(),"--inputs",str(fixture),"--input-source","captured",
                         "--output",str(p/"router-export")]
        subprocess.run(captured_export,check=True,stdout=subprocess.DEVNULL)
        exported_meta=verify_fixture(p/"router-export")
        assert exported_meta["activation_provenance"]["layer"]==10
        capture_meta["layer"]=11
        (fixture/"activation-provenance.json").write_text(json.dumps(capture_meta))
        assert subprocess.run(captured_export,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode!=0
        (fixture/"activation.f32").write_bytes(struct.pack("<4096f",*([99.0]*4096)))
        try:
            verify_fixture(fixture)
            raise AssertionError("tampered fixture accepted")
        except ValueError:
            pass
        traces[2].write_bytes(traces[0].read_bytes())
        assert subprocess.run(convert,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode!=0
        for raw in (record(10,0,10)[:-1],record(10,0,10)+record(10,0,20),
                    record(10,0,float("nan")),
                    struct.pack("<6I",0x31435247,10,0,2,4096,8)+struct.pack("<16i",*([0]*16))):
            badtrace=p/"bad.capture";badtrace.write_bytes(raw)
            try:
                read_trace(badtrace,10,144)
                raise AssertionError("malformed trace accepted")
            except ValueError:
                pass
    print("Export, cropped leakage, trace conversion/rejection, fixture hashes and rotation checks passed")


if __name__ == "__main__":
    main()
