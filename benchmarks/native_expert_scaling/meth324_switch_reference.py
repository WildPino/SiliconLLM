"""Bounded official Switch reference setup and model-free qualification."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / 'docs/research/NATIVE_EXPERT_SCALING_20260925'
PROTOCOL = DOC / 'METH_324_SWITCH_REFERENCE_PROTOCOL_20261003.md'
ASSETS = ROOT / 'results/native_expert_scaling/meth320_switch_metadata'
WORK = ROOT / 'results/native_expert_scaling/meth324_switch_reference'
ENV = WORK / 'venv'
PACKAGES = {'transformers': '4.57.6', 'huggingface-hub': '0.36.0'}
PRIOR = DOC / 'meth321_switch_metadata_compatibility_result.json'
PRIOR_SHA = '31351eae3934228468366501492c291f8a7f8b0a228bf3db80a146d797b9ebb1'
DIAGNOSTIC = DOC / 'meth323_switch_runtime_diagnostic_result.json'
DIAGNOSTIC_SHA = '295c2c7855a4f7f8655600545b253a6c6b5e428ed9a7d6f7e0c6ef729ae0e9dc'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, data):
    with Path(path).open('w', encoding='utf-8', newline='\r\n') as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def committed(path):
    rel = Path(path).relative_to(ROOT).as_posix()
    data = subprocess.check_output(['git', '-c', 'core.autocrlf=false', 'cat-file', '--filters', '--path='+rel, 'HEAD:'+rel], cwd=ROOT)
    assert data == Path(path).read_bytes(), ('uncommitted_or_changed', rel)


def setup(result, start):
    import requests
    import psutil
    import venv
    assert not WORK.exists(), 'Fresh setup only; preserve partial setup before any repair'
    WORK.mkdir(parents=True)
    wheel_dir = WORK / 'wheels'
    wheel_dir.mkdir()
    result['wheels'] = []
    consumed = 0

    def get(url, maximum):
        nonlocal consumed
        with requests.get(url, stream=True, timeout=(15, 30)) as response:
            response.raise_for_status()
            data = bytearray()
            for block in response.iter_content(1 << 16):
                data.extend(block)
                consumed += len(block)
                assert len(data) <= maximum and consumed <= 1 << 30
                assert time.monotonic()-start <= 1200
            return bytes(data)

    for package, version in PACKAGES.items():
        url = f'https://pypi.org/pypi/{package}/{version}/json'
        raw = get(url, 4 << 20)
        metadata_path = WORK / (package + '_pypi.json')
        metadata_path.write_bytes(raw)
        metadata = json.loads(raw)
        choices = [v for v in metadata['urls'] if v['filename'].endswith('py3-none-any.whl') and not v['yanked']]
        assert len(choices) == 1
        item = choices[0]
        assert item['url'].startswith('https://files.pythonhosted.org/') and item['size'] < 64 << 20
        wheel = wheel_dir / item['filename']
        wheel.write_bytes(get(item['url'], item['size']))
        assert wheel.stat().st_size == item['size'] and digest(wheel) == item['digests']['sha256']
        result['wheels'].append({'package': package, 'version': version, 'path': str(wheel), 'bytes': item['size'],
                                'sha256': digest(wheel), 'url': item['url'], 'metadata_url': url,
                                'metadata_sha256': digest(metadata_path)})
    write(WORK / 'wheel_manifest.json', result['wheels'])
    venv.EnvBuilder(with_pip=True).create(ENV)
    python = ENV / 'Scripts/python.exe'

    def run(command, label):
        stdout = WORK / (label + '.stdout.log')
        stderr = WORK / (label + '.stderr.log')
        with stdout.open('wb') as out, stderr.open('wb') as err:
            child = subprocess.Popen(command, stdout=out, stderr=err, cwd=ROOT, env=os.environ.copy())
            maximum_rss = 0
            while child.poll() is None:
                try:
                    process = psutil.Process(child.pid)
                    rss = process.memory_info().rss + sum(p.memory_info().rss for p in process.children(recursive=True))
                    maximum_rss = max(maximum_rss, rss)
                    if rss > 3 << 30 or time.monotonic()-start > 1200:
                        for p in process.children(recursive=True): p.kill()
                        process.kill()
                        child.wait()
                        raise RuntimeError('setup/runtime resource guard')
                except psutil.NoSuchProcess:
                    pass
                time.sleep(.25)
        result.setdefault('commands', []).append({'argv': command, 'returncode': child.returncode,
                                                 'peak_sampled_rss_bytes': maximum_rss,
                                                 'stdout_sha256': digest(stdout), 'stderr_sha256': digest(stderr)})
        assert child.returncode == 0, ('child_failed', label, child.returncode)

    run([str(python), '-m', 'pip', 'install', '--no-index', '--no-deps', *[v['path'] for v in result['wheels']]], 'install')
    donor_site = ROOT / '.venv/Lib/site-packages'
    pth = ENV / 'Lib/site-packages/donor_runtime.pth'
    pth.write_text(str(donor_site)+'\n', encoding='utf-8')
    result['controlled_runtime_path'] = {'path': str(pth), 'sha256': digest(pth), 'appended_site': str(donor_site)}
    # Bind the installed wheel model to the unmodified official release source.
    for filename in ('modeling_switch_transformers.py', 'configuration_switch_transformers.py'):
        url = 'https://raw.githubusercontent.com/huggingface/transformers/v4.57.6/src/transformers/models/switch_transformers/'+filename
        raw = get(url, 1 << 20)
        saved = WORK / ('official_'+filename)
        saved.write_bytes(raw)
        installed = ENV / 'Lib/site-packages/transformers/models/switch_transformers' / filename
        assert installed.read_bytes() == raw, ('wheel_official_release_mismatch', filename)
        result.setdefault('official_sources', []).append({'url': url, 'sha256': digest(saved), 'path': str(installed)})
    result['downloaded_bytes'] = consumed
    run([str(python), str(Path(__file__).resolve()), '--worker', '--out', str(WORK/'qualification.json')], 'qualification')
    qualification = json.loads((WORK/'qualification.json').read_text(encoding='utf-8'))
    result['qualification'] = qualification
    result['qualification_sha256'] = digest(WORK/'qualification.json')
    result['passed'] = all(qualification['gates'].values())
    result['decision'] = 'eligible_for_separate_real_source_binding' if result['passed'] else 'reference_not_qualified'


def qualify(result, start):
    import gc
    import importlib.metadata as metadata
    import inspect
    import psutil
    import torch
    import transformers
    import huggingface_hub
    from packaging.requirements import Requirement
    from transformers import SwitchTransformersConfig, SwitchTransformersForConditionalGeneration
    from transformers.models.switch_transformers.modeling_switch_transformers import SwitchTransformersSparseMLP, SwitchTransformersTop1Router
    assert transformers.__version__ == '4.57.6' and huggingface_hub.__version__ == '0.36.0'
    assert torch.__version__ == '2.6.0+cu124'
    assert Path(transformers.__file__).is_relative_to(ENV) and Path(huggingface_hub.__file__).is_relative_to(ENV)
    assert Path(torch.__file__).is_relative_to(ROOT/'.venv')
    dependencies = []
    for name in (*PACKAGES, 'torch', 'tokenizers', 'safetensors'):
        distribution = metadata.distribution(name)
        for text in distribution.requires or []:
            requirement = Requirement(text)
            if requirement.marker is None or requirement.marker.evaluate({'extra': ''}):
                version = metadata.version(requirement.name)
                assert version in requirement.specifier, ('incompatible_dependency', name, text, version)
                dependencies.append({'owner': name, 'requirement': text, 'resolved': version})
    result['dependencies'] = dependencies
    result['distributions'] = []
    for dist in sorted(metadata.distributions(), key=lambda d: (d.metadata['Name'] or '').lower()):
        files = list(dist.files or [])
        bindings = {Path(f).name: digest(dist.locate_file(f)) for f in files if Path(f).name in ('METADATA', 'RECORD') and Path(dist.locate_file(f)).is_file()}
        result['distributions'].append({'name': dist.metadata['Name'], 'version': dist.version,
                                        'location': str(dist.locate_file('')), 'metadata_hashes': bindings})
    result['model_source_sha256'] = digest(inspect.getfile(SwitchTransformersSparseMLP))
    result['config_source_sha256'] = digest(inspect.getfile(SwitchTransformersConfig))
    torch.set_num_threads(1)
    torch.manual_seed(322)
    cfg = SwitchTransformersConfig(d_model=8,d_ff=16,d_kv=4,num_heads=2,num_experts=2,
        num_layers=2,num_decoder_layers=2,num_sparse_encoder_layers=1,num_sparse_decoder_layers=1,
        expert_capacity=1,dropout_rate=0.,router_jitter_noise=0.,vocab_size=32,
        pad_token_id=0,eos_token_id=1,decoder_start_token_id=0,router_dtype='float32')
    sparse = SwitchTransformersSparseMLP(cfg).eval()
    with torch.no_grad():
        sparse.router.classifier.weight.zero_()
        sparse.router.classifier.weight[0,0] = 1.
        sparse.router.classifier.weight[1,0] = -1.
    x = torch.zeros(1,4,8)
    x[0,:,0] = torch.tensor([1.,2.,-1.,-2.])
    x[0,:,1] = 1.
    with torch.no_grad():
        mask, probability, logits = sparse.router(x)
        repeat = sparse.router(x)
        raw = sparse.router.classifier(x)
        selected = torch.softmax(raw,dim=-1).max(dim=-1,keepdim=True).values
    expected = torch.tensor([[[1,0],[0,0],[0,1],[0,0]]])
    result['tiny_config'] = cfg.to_dict()
    result['router'] = {'shapes': [list(v.shape) for v in (mask,probability,logits)], 'mask': mask.tolist(),
                        'exact_capacity_mask': bool(torch.equal(mask,expected)),
                        'exact_selected_probability': bool(torch.equal(probability,selected)),
                        'exact_raw_logits': bool(torch.equal(logits,raw)),
                        'repeatable_eval': all(torch.equal(a,b) for a,b in zip((mask,probability,logits),repeat))}
    result['sparse_cases'] = []
    for shape in ((1,1,8),(1,4,8),(2,2,8)):
        values = torch.arange(shape[0]*shape[1]*shape[2],dtype=torch.float32).reshape(shape)/32.-.5
        with torch.no_grad():
            output, route = sparse(values)
            probs = torch.softmax(sparse.router.classifier(values),dim=-1)
            chosen = probs.argmax(dim=-1)
            onehot = torch.nn.functional.one_hot(chosen,num_classes=2)
            accepted = (onehot.cumsum(dim=1)<=1)&onehot.bool()
            reference = torch.zeros_like(values)
            unmasked = torch.zeros_like(values)
            for batch in range(shape[0]):
                for token in range(shape[1]):
                    expert = int(chosen[batch,token])
                    value = sparse.experts[f'expert_{expert}'](values[batch,token])*probs[batch,token,expert]
                    unmasked[batch,token] = value
                    if bool(accepted[batch,token,expert]): reference[batch,token] = value
        norm = float(torch.linalg.vector_norm(reference))
        assert norm > 0
        error = float(torch.linalg.vector_norm(output-reference)/norm)
        dropped_norm = float(torch.linalg.vector_norm(output[~accepted.any(dim=-1)]))
        fault = float(torch.linalg.vector_norm(unmasked-reference)/norm)
        entry = {'shape': list(shape), 'relative_l2': error, 'dropped_output_norm': dropped_norm,
                 'capacity_omission_fault_relative': fault, 'zero_output_fault_detected': True,
                 'raw_logits_exact': bool(torch.equal(route[0],sparse.router.classifier(values))),
                 'masked_expert_index_exact': bool(torch.equal(route[1],accepted.long().argmax(dim=-1))),
                 'passed': output.shape == values.shape and bool(torch.isfinite(output).all()) and error<=1e-6 and dropped_norm==0 and (shape[1]==1 or fault>1e-6)}
        entry['passed'] = entry['passed'] and entry['raw_logits_exact'] and entry['masked_expert_index_exact']
        result['sparse_cases'].append(entry)
    tiny = SwitchTransformersForConditionalGeneration(cfg).eval()
    with torch.no_grad():
        first = tiny(input_ids=torch.tensor([[2,3,4]]), decoder_input_ids=torch.tensor([[0]]), use_cache=True)
    result['capacity1_full_head'] = {'shape': list(first.logits.shape), 'finite': bool(torch.isfinite(first.logits).all()),
                                     'cache_present': first.past_key_values is not None}
    # Capacity is per call. Compare caches only in a fixed unsaturated context.
    for router in tiny.modules():
        if type(router) is SwitchTransformersTop1Router: router.expert_capacity = 64
    source = torch.tensor([[2,3,4]])
    decoder = torch.tensor([[0,5,6,7]])
    cache = None
    result['cache_steps'] = []
    with torch.no_grad():
        encoded = tiny.encoder(input_ids=source,return_dict=True)
        for step in range(4):
            cached = tiny(encoder_outputs=encoded,decoder_input_ids=decoder[:,step:step+1],past_key_values=cache,use_cache=True)
            cache = cached.past_key_values
            full = tiny(input_ids=source,decoder_input_ids=decoder[:,:step+1],use_cache=False)
            a,b = cached.logits[:,-1],full.logits[:,-1]
            norm = float(torch.linalg.vector_norm(b))
            assert norm > 0
            error = float(torch.linalg.vector_norm(a-b)/norm)
            exact_top1 = bool(torch.equal(a.argmax(-1),b.argmax(-1)))
            self_lengths = [int(layer.keys.shape[-2]) for layer in cache.self_attention_cache.layers]
            cross_lengths = [int(layer.keys.shape[-2]) for layer in cache.cross_attention_cache.layers]
            result['cache_steps'].append({'step':step,'relative_logit_l2':error,'exact_top1':exact_top1,
                'self_lengths':self_lengths,'cross_lengths':cross_lengths,
                'passed':bool(torch.isfinite(a).all()) and error<=1e-5 and exact_top1 and self_lengths==[step+1]*2 and cross_lengths==[3]*2})
    del tiny,sparse,first,cache,encoded
    gc.collect()
    prior = json.loads(PRIOR.read_text(encoding='utf-8'))
    result['meta_models'] = []
    for previous in prior['models']:
        model = previous['model']
        config_path = ASSETS / (model.replace('/','_')+'_config.json')
        for binding in previous['bindings']:
            assert digest(binding['path']) == binding['sha256']
        config = json.loads(config_path.read_text())
        with torch.device('meta'): network = SwitchTransformersForConditionalGeneration(SwitchTransformersConfig(**config))
        state = network.state_dict()
        shapes = {name:list(value.shape) for name,value in sorted(state.items())}
        unique = sum(value.numel() for value in network.parameters())
        routers = [v for v in network.modules() if type(v) is SwitchTransformersTop1Router]
        result['meta_models'].append({'model':model,'unique_parameters':unique,'serialized_names':len(state),
            'exact321_shapes':shapes==previous['tensors'],'exact321_unique_count':unique==previous['unique_parameter_count_inferred'],
            'router_count':len(routers),'top1_bank_count_correct':len(routers)==previous['top1_proof']['router_modules'] and all(r.num_experts==config['num_experts'] for r in routers)})
        del state,network,routers
        gc.collect()
    head = result['capacity1_full_head']
    result['gates'] = {'direct_router': all(v for k,v in result['router'].items() if k not in ('shapes','mask')),
        'all_sparse_capacity_controls': all(v['passed'] for v in result['sparse_cases']),
        'tiny_capacity1_full_head_cache':head['shape']==[1,1,32] and head['finite'] and head['cache_present'],
        'unsaturated_cached_full_prefix':all(v['passed'] for v in result['cache_steps']),
        'all_original_meta_namespaces':all(v['exact321_shapes'] and v['exact321_unique_count'] and v['top1_bank_count_correct'] for v in result['meta_models'])}
    result['resource'] = {'seconds':time.monotonic()-start,'end_rss_bytes':psutil.Process().memory_info().rss}
    assert result['resource']['seconds'] <= 300 and result['resource']['end_rss_bytes'] <= 3 << 30
    result['scope'] = 'Unmodified official reference qualified only on tiny synthetic controls and META shapes. Capacity resets per call; unsaturated cache equivalence is not saturated full-prefix equivalence. No learned source weights, quality, native LUT or accepted token rate.'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    assert not args.out.exists() and not args.out.with_suffix('.failure.json').exists()
    start = time.monotonic()
    result = {'experiment':'METH-324-isolated-official-Switch-reference','source_payload_bytes':0}
    try:
        for path in (Path(__file__),PROTOCOL,PRIOR,DIAGNOSTIC): committed(path)
        assert digest(PRIOR)==PRIOR_SHA and digest(DIAGNOSTIC)==DIAGNOSTIC_SHA
        result.update({'controller_sha256':digest(__file__),'protocol_sha256':digest(PROTOCOL),
                       'prior_sha256':PRIOR_SHA,'diagnostic_sha256':DIAGNOSTIC_SHA})
        if args.worker: qualify(result,start)
        else: setup(result,start)
        result['seconds'] = time.monotonic()-start
        write(args.out,result)
        print(json.dumps({'sha256':digest(args.out),'seconds':result['seconds'],'gates':result.get('gates'),
                          'passed':result.get('passed'),'decision':result.get('decision')}),flush=True)
    except BaseException as error:
        result.update({'error':repr(error),'seconds':time.monotonic()-start})
        write(args.out.with_suffix('.failure.json'),result)
        raise


if __name__ == '__main__': main()
