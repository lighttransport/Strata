"""Read-only clock samples during the decode interval of one guarded run."""
import argparse,json,re,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('log',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
assert not a.output.exists()
d=next(x for x in Path('/sys/class/drm').glob('card[0-9]*/device') if (x/'mem_info_vram_total').exists())
samples=[]; end=time.monotonic()+600
while time.monotonic()<end:
    text=a.log.read_text() if a.log.exists() else ''
    if 'decode_tok_s=' in text: break
    if 'DS41 prefill_done_s=' in text:
        row={'time':time.time()}
        for key,name in [('core','pp_dpm_sclk'),('memory','pp_dpm_mclk')]:
            m=re.search(r'(\d+)Mhz\s*\*',(d/name).read_text())
            row[key]=int(m[1]) if m else None
        samples.append(row)
    time.sleep(.1)
result={'samples':samples,'mode':(d/'power_dpm_force_performance_level').read_text().strip()}
for key in ('core','memory'):
    values=[x[key] for x in samples if x[key] is not None]
    if values: result[key]={'min':min(values),'mean':sum(values)/len(values),'max':max(values)}
a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='samples'}), 'samples',len(samples))
