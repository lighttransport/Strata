"""Record live GLM code generation in Pi's TUI (text-only, no agent tools)."""
import argparse
import json
import os
import pathlib
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

REPO = pathlib.Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', type=pathlib.Path, help='first Q2 GLM GGUF shard')
    parser.add_argument('--decoder', type=pathlib.Path, default=REPO/'build-glm/strata-glm-decode')
    parser.add_argument('--output', type=pathlib.Path, default=REPO/'recordings/glm53-pi.cast')
    parser.add_argument('--port', type=int, default=18095)
    parser.add_argument('--speculative', choices=('none','mtp'), default='none',
                        help='single decode by default; GPU MTP needs more free VRAM')
    parser.add_argument('--prepare-only', action='store_true', help='write configs without starting GPU/server/Pi')
    args = parser.parse_args()
    for program in ('pi', 'asciinema'):
        if not shutil.which(program):
            parser.error(f'{program} is not installed')
    if not args.model.is_file() or not args.decoder.is_file():
        parser.error('model shard and decoder must exist')
    if not 1024 <= args.port <= 65535:
        parser.error('port must be between 1024 and 65535')
    output = args.output.resolve()
    if output.exists() and not args.prepare_only:
        parser.error('recording already exists; choose another --output')
    runtime = pathlib.Path(tempfile.mkdtemp(prefix='strata-pi-demo-'))
    profile = runtime/'pi-profile'
    profile.mkdir()
    work = runtime/'demo'
    work.mkdir()
    url = f'http://127.0.0.1:{args.port}'
    server_config = dict(log=str(runtime/'engine.log'), exe=str(args.decoder.resolve()), model=str(args.model.resolve()),
                         model_name='glm-5.3-flash-q2', context=8192, dense_cache_mib=4096,
                         expert_cache_mib=0, threads=15, prefill_batch=2048,
                         gpu_budget_mib=12288, decode_experts='cpu', speculative=args.speculative,
                         draft_depth=1, cpu_affinity='auto', cpu_prepack_mib=0)
    (runtime/'server.json').write_text(json.dumps(server_config, indent=2)+'\n')
    model = dict(id='glm-5.3-flash-q2', name='GLM5.3Flash Q2 (local Strata)',
                 reasoning=False, input=['text'], contextWindow=8192, maxTokens=512,
                 cost=dict(input=0, output=0, cacheRead=0, cacheWrite=0),
                 samplingParams=dict(temperature=0, reasoning_effort='low', max_tokens=512),
                 compat=dict(supportsDeveloperRole=False, supportsStore=False,
                             supportsReasoningEffort=False, supportsUsageInStreaming=False,
                             maxTokensField='max_tokens'))
    models = dict(providers={'strata-local': dict(baseUrl=url+'/v1', api='openai-completions',
                                                apiKey='local-demo-placeholder', models=[model])})
    (profile/'models.json').write_text(json.dumps(models, indent=2)+'\n')
    (profile/'settings.json').write_text(json.dumps(dict(enableInstallTelemetry=False,
        enableAnalytics=False, defaultProjectTrust='never', hideThinkingBlock=True,
        quietStartup=False, theme='dark', compaction={'enabled':False}), indent=2)+'\n')
    (work/'prompt.txt').write_text((REPO/'demos/glm53_pi/prompt.txt').read_text())
    pi = ['pi','--offline','--no-tools','--no-extensions','--no-skills',
          '--no-prompt-templates','--no-context-files','--no-themes','--no-approve',
          '--no-session','--provider','strata-local','--model','glm-5.3-flash-q2',
          '--system-prompt','You are a concise C++17 coding assistant. Return a complete function and a brief explanation. Keep reasoning brief. Tools are unavailable.',
          '@prompt.txt']
    print('Isolated runtime:', runtime, flush=True)
    print('Demo mode: live code generation in Pi; agent tools disabled.', flush=True)
    print('Pi command:', shlex.join(pi), flush=True)
    if args.prepare_only:
        return
    output.parent.mkdir(parents=True,exist_ok=True)
    env = dict(os.environ, PI_CODING_AGENT_DIR=str(profile), PI_OFFLINE='1',
               PI_TELEMETRY='0', STRATA_NATIVE_NUMA_LOCAL='0', TERM='xterm-256color')
    # This demo needs no user API keys or inherited API-key requirement.
    for key in list(env):
        if key.endswith(('_API_KEY','_AUTH_TOKEN','_OAUTH_TOKEN')) or key.startswith('STRATA_API'):
            env.pop(key,None)
    tools = str(REPO/'tools')
    env['PYTHONPATH'] = tools + (os.pathsep+env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    server = None
    try:
        with (runtime/'server.log').open('w') as log:
            server = subprocess.Popen([sys.executable,'-m','serve.server','--engine','glm',
                '--config',str(runtime/'server.json'),'--host','127.0.0.1','--port',str(args.port)],
                cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
            deadline = time.monotonic()+240
            while time.monotonic()<deadline:
                if server.poll() is not None:
                    raise RuntimeError('server exited; inspect '+str(runtime/'server.log'))
                try:
                    with urllib.request.urlopen(url+'/v1/models',timeout=2) as response:
                        if response.status==200:
                            break
                except (urllib.error.URLError, TimeoutError):
                    pass
                time.sleep(1)
            else:
                raise RuntimeError('server startup timed out')
            print('Server ready. Recording Pi TUI; after the answer finishes, press Ctrl+D to stop.',flush=True)
            subprocess.run(['asciinema','rec','--quiet','--cols','110','--rows','36',
                '--env','TERM','--title','GLM5.3Flash Q2 / Strata / Pi code generation (tools disabled)',
                '--command',shlex.join(pi),str(output)],cwd=work,env=env,check=True)
    finally:
        if server is not None:
            server.terminate()
            try:
                server.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
    print('Recording:', output)
    print('Replay: asciinema play '+shlex.quote(str(output)))
    print('Server log retained in:', runtime)


if __name__ == '__main__':
    main()
