"""Supplement the finished sweep with the decoder's supported maximum cache request.

The original 14336 MiB request exceeds the unchanged CLI's 12288 MiB cap.
Keep its failures, then measure allocator admission at the supported cap with
fresh ordinary BASE controls. Run only after run_sweep.py has finished.
"""
import copy,json
import run_sweep as sweep

def main():
    path=sweep.OUT/'sweep-records.json'
    data=json.loads(path.read_text())
    if data['status']!='complete':
        raise RuntimeError('Main interleaved sweep must finish before supplement')
    rows=sorted(data['records']+data['failed_records'],key=lambda r:r['name'])
    if any(r.get('supported_cli_max_cache') for r in rows):
        raise RuntimeError('Supported maximum cache supplement already recorded')
    sweep.ROWS.extend(rows)
    sweep.COUNTER=max(int(r['name'].split('-')[0]) for r in rows)+1
    cfg=json.loads((sweep.OUT/'base.config.json').read_text())
    cfg['env']['STRATA_GLM_STEP_TRACE']='1'
    control=dict(name='base-ordinary256',config=copy.deepcopy(cfg),
                 cohort='ordinary256',kind='decode',attempt=1,lever='BASE')
    cfg['decode_cache_mib']=12288
    attempt=max(r.get('attempt',1) for r in rows
                if r.get('lever')=='largest-admitted-cache')+1
    case=dict(name='cache12288-supported',config=cfg,cohort='ordinary256',
              kind='decode',attempt=attempt,lever='largest-admitted-cache')
    sweep.block([case],control)
    measured=sweep.ROWS[-2]
    measured['supported_cli_max_cache']=True
    measured['diagnostic']['cache_request_note']=(
        '12288 MiB is the unchanged CLI maximum. Admission also enforces the '
        'physical reserve and live-allocation budget; actual slots/MiB are '
        'reported by DECODE_CACHE. Earlier 14336 MiB attempts are retained.')
    sweep.core.save(sweep.OUT/(measured['name']+'.record.json'),measured)
    sweep.summarize()
    data=json.loads(path.read_text());data['status']='complete'
    data['supported_max_cache_supplement']=True
    sweep.core.save(path,data);sweep.core.save(sweep.CURRENT,data)
    sweep.core.save(sweep.OUT/'progress.json',dict(state='complete',
        records=len(data['records']),failed_attempts=len(data['failed_records']),
        supported_max_cache_supplement=True))
    print('SUPPORTED_MAX_CACHE_COMPLETE',flush=True)

if __name__=='__main__':main()
