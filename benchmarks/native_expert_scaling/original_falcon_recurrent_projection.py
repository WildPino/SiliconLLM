"""FIT-only recurrent bases and stored full-history rank96 output experiments.

Keeps all source x/gate/head functions: local optimistic diagnostics, not a compact
chatbot, target width proof, source forward replay or optimizer update.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
B = ROOT/'benchmarks/native_expert_scaling'
DOC = ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path.insert(0, str(B))
from chatbot_falcon_usability import SITE, sha, write
from original_falcon_whole_recovery import extent, check_inputs, memory_reader
sys.path.insert(0, str(SITE))


def bind(a):
    capture = DOC/'original_falcon_recurrent_adopted_result_20261009.json'
    r = json.loads(capture.read_bytes())
    terminal = Path(r['completion_terminal']['path']); t = json.loads(terminal.read_bytes())
    assert t['exit_code'] == 0 and t['resource_gates']
    assert extent(terminal) == r['completion_terminal'] and r['stored_finite_extent_hash_adoption']
    assert sha(r['completion_raw_result']['path']) == t['result_sha256']
    assert r['cases'] == 48 and r['sites_per_case'] == 24 and r['LM_head_calls'] == 0
    binding = Path(r['adoption_binding']['path'])
    cb = json.loads(binding.read_bytes()); assert sha(binding) == r['binding_sha256']
    files = [Path(__file__), B/'chatbot_falcon_usability.py', B/'chatbot_falcon_usability_launch.py',
        B/'original_falcon_whole_recovery.py', capture, terminal, binding,
        DOC/'ORIGINAL_FALCON_RECURRENT_PROJECTION_PROTOCOL_20261009.md',
        Path(cb['source'])/'model.safetensors', Path(cb['source'])/'config.json', Path(sys.executable)]
    for rec in r['records']:
        for site in rec['sites']:
            names = ('B', 'C') if site['site'] not in (0, 12, 23) else tuple(site['fields'])
            for name in names:
                item = site['fields'][name]; assert extent(item['path']) == {k:item[k] for k in ('path','bytes','sha256')}
                files.append(Path(item['path']))
    files += [SITE/'torch'/p for p in ('__init__.py', '_C.cp312-win_amd64.pyd',
        'lib/torch_cpu.dll', 'lib/torch_cuda.dll', 'cuda/__init__.py')]
    files += [SITE/'transformers/models/falcon_h1/modeling_falcon_h1.py',
        SITE/'numpy/__init__.py', SITE/'psutil/__init__.py', SITE/'safetensors/torch.py']
    files += [ROOT/p for p in ('benchmarks/phase60/engine.c',
        'benchmarks/donor_adaptation/configs/_manifest.json',
        'benchmarks/donor_adaptation/density/build_document_holdout.py', 'docs/research/RESEARCH_INDEX.md')]
    write(a.out, dict(schema='ORIGINAL_FALCON_RECURRENT_PROJECTION_BINDING_V1',
        python=str(Path(sys.executable).resolve()), worker_path=str(Path(__file__).resolve()),
        capture=str(capture.resolve()), source_weights=str((Path(cb['source'])/'model.safetensors').resolve()),
        source_config=str((Path(cb['source'])/'config.json').resolve()),
        basis_sites=list(range(24)), assay_sites=[0, 12, 23], rank=96,
        basis='Equal FIT case mean of separately trace-normalized uncentered B/C Grams; '
            'top joint eigenvectors, top diagonal coordinates, or dual biorthogonal pair-kernel bases; '
            'DEV not used to fit/select.',
        criteria=dict(reconstruction_relative_RMS=1e-4, scalar_F64_relative_RMS=1e-4,
            orthogonality=1e-10, biorthogonality=1e-8),
        limits=dict(seconds=1800, reserve_seconds=90, OS_bytes=6 << 30,
            GPU_allocated_bytes=4 << 30, GPU_reserved_bytes=5 << 30, output_bytes=96 << 20),
        inputs=[extent(p) for p in dict.fromkeys(files)],
        runtime_binding_scope='Complete stored B/C all24 sites plus all seven operands of sites0/12/23; '
            'pinned source coefficient file, primary runtime binaries; not a source-model execution. '
            'Interrupted source parent runtime/resource aggregate qualification remains missing.'))
    print(json.dumps(dict(binding=str(a.out), sha256=sha(a.out), inputs=len(dict.fromkeys(files)))), flush=True)


def launch(a):
    import psutil
    start = time.monotonic(); assert sha(a.binding) == a.binding_sha
    b = json.loads(a.binding.read_bytes())
    assert b['schema'] == 'ORIGINAL_FALCON_RECURRENT_PROJECTION_BINDING_V1'
    proc = psutil.Process(); proc.cpu_affinity([11]); own = {proc.pid, *(p.pid for p in proc.parents())}
    for p in psutil.process_iter(['name', 'cmdline']):
        name = (p.info['name'] or '').lower(); argv = p.info['cmdline'] or []
        if p.pid in own: continue
        if name == 'pythonw.exe' and len(argv) == 2 and Path(argv[1]).resolve() == Path('D:/_THINGS/Progetti/tiktok_publisher/daemon_workflow.py').resolve(): continue
        assert not name.startswith(('python', 'clang', 'gcc', 'engine', 'packed_original')), ('overlap', p.pid, name)
    log = a.out.with_suffix('.worker.log'); terminal = a.out.with_suffix('.terminal.json')
    assert not a.directory.exists() and not any(p.exists() for p in (a.out, log, terminal, a.out.with_suffix('.launcher_failure.json')))
    worker = None; peak = 0; reader = memory_reader()
    record = dict(freeze=a.freeze, binding_sha256=a.binding_sha, launcher_pid=proc.pid)
    def memory():
        nonlocal peak
        peak = max(peak, reader(worker))
    def guard():
        assert time.monotonic()-start <= b['limits']['seconds'], 'family deadline'
        assert peak+proc.memory_info().peak_wset <= b['limits']['OS_bytes'], 'family OS cap'
        assert log.stat().st_size <= 4 << 20, 'log cap'
    try:
        check_inputs(b)
        env = dict(os.environ, HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
            TOKENIZERS_PARALLELISM='false', OMP_NUM_THREADS='6', OPENBLAS_NUM_THREADS='1',
            MKL_NUM_THREADS='6', CUBLAS_WORKSPACE_CONFIG=':4096:8')
        argv = [b['python'], '-I', '-S', '-B', '-X', 'utf8', b['worker_path'], '--worker',
            '--binding', str(a.binding.resolve()), '--binding-sha', a.binding_sha, '--freeze', a.freeze,
            '--directory', str(a.directory.resolve()), '--out', str(a.out.resolve())]
        record['command'] = argv
        with log.open('xb') as stream:
            worker = subprocess.Popen(argv, stdout=stream, stderr=subprocess.STDOUT, env=env, creationflags=8)
            record.update(worker_pid=worker.pid, worker_creation_time=psutil.Process(worker.pid).create_time()); offset = 0
            while worker.poll() is None:
                memory(); guard()
                try: assert not psutil.Process(worker.pid).children(recursive=True), 'unexpected child'
                except psutil.NoSuchProcess: pass
                with log.open('rb') as f:
                    f.seek(offset); chunk = f.read(); boundary = chunk.rfind(b'\n')+1
                    if boundary: print(chunk[:boundary].decode('utf8', errors='replace'), end='', flush=True); offset += boundary
                time.sleep(.1)
        memory(); guard(); record.update(exit_code=worker.returncode, worker_OS_peak_through_exit=peak,
            launcher_OS_peak_snapshot=proc.memory_info().peak_wset)
        assert worker.returncode == 0, log.read_text(errors='replace')[-6000:]
        check_inputs(b); r = json.loads(a.out.read_bytes())
        assert r['schema'] == 'ORIGINAL_FALCON_RECURRENT_PROJECTION_RESULT_V1'
        assert len(r['bases']) == 24 and len(r['records']) == 144
        assert r['source_forwards'] == r['optimizer_updates'] == r['native_calls'] == 0
        assert r['GPU_allocated_peak'] <= b['limits']['GPU_allocated_bytes']
        assert r['GPU_reserved_peak'] <= b['limits']['GPU_reserved_bytes']
        outputs = [extent(p) for p in sorted(a.directory.iterdir()) if p.is_file()]
        assert sum(i['bytes'] for i in outputs) <= b['limits']['output_bytes']
        record.update(elapsed_seconds=time.monotonic()-start, result_sha256=sha(a.out),
            decision=r['decision'], output_files=outputs, resource_gates=True)
        write(terminal, record)
        print(json.dumps(dict(terminal=str(terminal), worker_pid=worker.pid, exit_code=worker.returncode,
            seconds=record['elapsed_seconds'], OS_peak=peak, decision=r['decision'])), flush=True)
    except BaseException as error:
        if worker is not None:
            if worker.poll() is None: worker.kill(); worker.wait()
            memory(); record.update(exit_code=worker.returncode, worker_OS_peak_through_exit=peak)
        record.update(fault=repr(error), elapsed_seconds=time.monotonic()-start,
            launcher_OS_peak_snapshot=proc.memory_info().peak_wset)
        write(a.out.with_suffix('.launcher_failure.json'), record); raise


def scan(torch, code, x, bb, cc, delta, A_log, dskip):
    """Same source SSD contractions/reduction order, arbitrary state coordinate count."""
    n = len(x); rank = bb.shape[-1]; pad = (128-n % 128) % 128
    x = x.reshape(1, n, 48, 64).float()
    bb = bb.reshape(1, n, 1, rank).float().repeat_interleave(48, dim=2)
    cc = cc.reshape(1, n, 1, rank).float().repeat_interleave(48, dim=2)
    delta = delta.reshape(1, n, 48)
    d_residual = dskip[..., None]*code.pad_tensor_by_size(x, pad)
    x = x*delta[..., None]; AA = -torch.exp(A_log.float())*delta
    x, AA, bb, cc = [code.reshape_into_chunks(t, pad, 128) for t in (x, AA, bb, cc)]
    AA = AA.permute(0, 3, 1, 2); acum = torch.cumsum(AA, dim=-1)
    decay = torch.exp(code.segment_sum(AA))
    G = torch.cat([(cc[:, i:i+1, :, None, :, :]*bb[:, i:i+1, None, :, :, :]).sum(-1)
        for i in range(cc.shape[1])], dim=1)
    M = (G[..., None]*decay.permute(0, 2, 3, 4, 1)[..., None]).sum(-1)
    diag = (M[..., None]*x[:, :, None]).sum(dim=3)
    ds = torch.exp(acum[:, :, :, -1:]-acum)
    bd = bb*ds.permute(0, -2, -1, 1)[..., None]
    states = torch.cat([(bd[:, i:i+1, ..., None, :]*x[:, i:i+1, ..., None]).sum(dim=2)
        for i in range(bd.shape[1])], dim=1)
    states = torch.cat([torch.zeros_like(states[:, :1]), states], dim=1)
    dc = torch.exp(code.segment_sum(torch.nn.functional.pad(acum[:, :, :, -1], (1, 0)))).transpose(1, 3)
    new_states = (dc[..., None, None]*states[:, :, None, ...]).sum(dim=1)
    states = new_states[:, :-1]
    reduced = torch.cat([(cc[:, i:i+1, ..., None, :]*states[:, i:i+1, None, ...]).sum(-1)
        for i in range(cc.shape[1])], dim=1)
    off = reduced*torch.exp(acum).permute(0, 2, 3, 1)[..., None]
    yy = (diag+off).reshape(1, -1, 48, 64)+d_residual
    return yy[:, :n].reshape(n, 3072)


def core_costs():
    """Exact shape accounting, no runtime/DRAM/speed claim for variants."""
    result=[]
    for dn,dt in ((512,16),(512,48),(1024,16),(1024,48)):
        D,N,L,V,E,K,H=256,96,6,65537,1152,8,128
        ssm_matrices=dn*(3*D+2*dt+2*N)
        core_matrices=5*ssm_matrices+4*D*D
        source_shaped_organs=5*(dn*(3*D+2*dt+3*N+7)+2*D)+(4*D*D+2*D)
        result.append(dict(D=D,DN=dn,N=N,DT_rank=dt,L=L,E=E,K=K,expert_H=H,V=V,
            core_matrix_products=core_matrices,
            head_matrix_products=V*D,selected_expert_matrix_products=L*K*3*D*H,
            flat_router_matrix_products=L*D*E,
            all_counted_matrix_products=core_matrices+V*D+L*K*3*D*H+L*D*E,
            core_master_coefficients=source_shaped_organs,core_F32_bytes=source_shaped_organs*4,
            recurrence_state_coefficients=5*dn*N,recurrence_F32_state_bytes=5*dn*N*4,
            untied_state_exponentials=5*dn*N,
            imported_scalar_head_exponential_hoisting='Algebraically possible for repeated A/delta only; '
                'not implemented or timed, cannot charge skipped work yet.',
            scope='L6/five SSM+one SWA/n1152/full V; excludes vector ops,conv,attention window, '
                'post-gate norm,packing/alignment and physical DRAM; variants not implemented.'))
    return result


def worker(a):
    start = time.monotonic(); assert sha(a.binding) == a.binding_sha
    b = json.loads(a.binding.read_bytes()); a.directory.mkdir(exist_ok=False)
    stage = 'startup'; bases = []; measured = []; torch = None; proc = None
    try:
        import numpy as np
        import psutil
        import torch
        from safetensors import safe_open
        from transformers.models.falcon_h1 import modeling_falcon_h1 as code
        assert (torch.__version__, np.__version__, psutil.__version__) == ('2.6.0+cu124', '2.4.6', '7.2.2')
        torch.set_num_threads(6); torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        proc = psutil.Process(); proc.cpu_affinity(list(range(6)))
        def guard():
            lim = b['limits']
            assert time.monotonic()-start <= lim['seconds']-lim['reserve_seconds'], 'worker reserve'
            assert proc.memory_info().peak_wset <= lim['OS_bytes'], 'OS cap'
            assert torch.cuda.max_memory_allocated() <= lim['GPU_allocated_bytes'], 'GPU allocated cap'
            assert torch.cuda.max_memory_reserved() <= lim['GPU_reserved_bytes'], 'GPU reserved cap'
            assert not proc.children(recursive=True), 'unexpected child'
        def event(**fields):
            guard(); print(json.dumps(dict(seconds=time.monotonic()-start, **fields)), flush=True)
        capture = json.loads(Path(b['capture']).read_bytes()); records = capture['records']
        fit = [r for r in records if r['split'] == 'FIT']; assert len(fit) == 24
        rank = b['rank']; all_bases = {}
        def raw(item):
            shape = tuple(item['shape'])
            if item['dtype'] == 'BF16':
                return (np.fromfile(item['path'], dtype='<u2').astype('<u4') << 16).view('<f4').reshape(shape)
            return np.fromfile(item['path'], dtype='<f4').reshape(shape)
        for site in b['basis_sites']:
            stage = f'basis{site}'
            joint = torch.zeros((256, 256), dtype=torch.float64)
            grams = {name:torch.zeros_like(joint) for name in ('B', 'C')}
            centered = torch.zeros_like(joint)
            for rec in fit:
                fields = rec['sites'][site]['fields']
                for name in ('B', 'C'):
                    z = torch.from_numpy(raw(fields[name]).astype('f8'))
                    gram = z.T@z; trace = torch.trace(gram); assert trace > 0
                    joint += gram/trace/48
                    grams[name] += gram/trace/24
                    centered_z = z-z.mean(0)
                    centered_gram = centered_z.T@centered_z
                    ctr = torch.trace(centered_gram)
                    if ctr > 0: centered += centered_gram/ctr/48
            ev, vec = torch.linalg.eigh(joint)
            R = vec[:, -rank:].contiguous(); coord = torch.argsort(torch.diag(joint), descending=True, stable=True)[:rank]
            orth = float((R.T@R-torch.eye(rank, dtype=torch.float64)).abs().max()); assert orth <= b['criteria']['orthogonality']
            roots = {}
            for name, gram in grams.items():
                eig, vectors = torch.linalg.eigh(gram); assert float(eig.min()) >= -1e-12
                roots[name] = (vectors*eig.clamp_min(0).sqrt())@vectors.T
            H = roots['C']@roots['B']; U, singular, Vh = torch.linalg.svd(H)
            assert float(singular[rank-1]) > 1e-12
            V = (roots['B']@Vh[:rank].T)*singular[:rank].rsqrt()
            W = (roots['C']@U[:,:rank])*singular[:rank].rsqrt()
            bio = float((W.T@V-torch.eye(rank,dtype=torch.float64)).abs().max())
            assert bio <= b['criteria']['biorthogonality']
            path = a.directory/f'site{site:02d}.basis.npz'
            with path.open('xb') as f: np.savez(f, R=R.numpy(), coordinates=coord.numpy(),
                eigenvalues=ev.numpy(), joint=joint.numpy(), centered=centered.numpy(),
                G_B=grams['B'].numpy(),G_C=grams['C'].numpy(),V=V.numpy(),W=W.numpy(),singular_values=singular.numpy())
            packet = dict(site=site, basis=extent(path), orthogonality_error=orth,
                biorthogonality_error=bio, dual_basis_V_L2=float(torch.linalg.matrix_norm(V,ord=2)),
                dual_basis_W_L2=float(torch.linalg.matrix_norm(W,ord=2)),
                minimum_retained_pair_singular_value=float(singular[rank-1]),
                optimal_pair_relative_error=float(torch.sqrt(singular[rank:].square().sum()/singular.square().sum())),
                dense_joint_retention=float(torch.trace(R.T@joint@R)/torch.trace(joint)),
                coordinate_joint_retention=float(torch.diag(joint)[coord].sum()/torch.trace(joint)),
                dense_centered_retention=float(torch.trace(R.T@centered@R)/torch.trace(centered)),
                coordinate_centered_retention=float(torch.diag(centered)[coord].sum()/torch.trace(centered)))
            bases.append(packet); all_bases[site]=(R.float().cuda(), coord.cuda(),V.float().cuda(),W.float().cuda())
            event(stage='basis', **packet)
        config = json.loads(Path(b['source_config']).read_bytes())
        reconstruction_failures = []; scalar_records = []
        def metric(value, ref):
            value = value.double(); ref = ref.double()
            return float(torch.sqrt((value-ref).square().mean())/torch.sqrt(ref.square().mean()).clamp_min(1e-12))
        with safe_open(b['source_weights'], framework='pt', device='cpu') as sf, torch.inference_mode():
            for site in b['assay_sites']:
                prefix = f'model.layers.{site}.mamba.'
                A_log = sf.get_tensor(prefix+'A_log').cuda(); dskip = sf.get_tensor(prefix+'D').cuda()
                weight = sf.get_tensor(prefix+'norm.weight').cuda()
                out_weight = sf.get_tensor(prefix+'out_proj.weight').cuda()
                def output(yy, gate):
                    gated = yy*torch.nn.functional.silu(gate.float())
                    normalized = gated*torch.rsqrt(gated.square().mean(-1, keepdim=True)+config['rms_norm_eps'])
                    return torch.nn.functional.linear((normalized*weight).bfloat16(), out_weight)
                R, coord, V, W = all_bases[site]
                for rec in records:
                    stage = f"{rec['id']}/site{site:02d}"
                    fields = rec['sites'][site]['fields']
                    values = {name:torch.from_numpy(raw(item)).cuda() for name,item in fields.items()}
                    for name in ('x', 'B', 'C', 'gate', 'delta', 'output'): values[name]=values[name].bfloat16()
                    xx, bb, cc, delta = [values[name] for name in ('x', 'B', 'C', 'delta')]
                    yy = scan(torch, code, xx, bb, cc, delta, A_log, dskip)
                    scan_error = metric(yy, values['y']); out = output(yy, values['gate'])
                    output_error = metric(out, values['output'])
                    ok = max(scan_error, output_error) <= b['criteria']['reconstruction_relative_RMS']
                    if not ok: reconstruction_failures.append(dict(id=rec['id'], site=site, scan_error=scan_error, output_error=output_error))
                    if rec['id'] == 'broad_fit_smol_magpie_ultra_022':
                        channels = np.array([0,63,64,511,1023,1535,2047,3071]); heads=channels//64
                        state = np.zeros((8,256), dtype='f8'); direct=[]
                        bb64=raw(fields['B']).astype('f8');cc64=raw(fields['C']).astype('f8')
                        x64=raw(fields['x'])[:,channels].astype('f8');dt64=raw(fields['delta'])[:,heads].astype('f8')
                        aa=-np.exp(A_log.cpu().float().numpy().astype('f8'))[heads]
                        dd=dskip.cpu().float().numpy().astype('f8')[heads]
                        for t in range(rec['history']):
                            state=state*np.exp(dt64[t,:,None]*aa[:,None])+dt64[t,:,None]*x64[t,:,None]*bb64[t]
                            direct.append((state*cc64[t]).sum(-1)+dd*x64[t])
                        actual=raw(fields['y'])[:,channels].astype('f8');direct=np.asarray(direct)
                        error=float(np.sqrt(np.mean((direct-actual)**2))/max(np.sqrt(np.mean(actual**2)),1e-12))
                        scalar_records.append(dict(id=rec['id'],site=site,channels=channels.tolist(),history=rec['history'],relative_RMS=error,
                            passed=error<=b['criteria']['scalar_F64_relative_RMS']))
                    arms = {}
                    skip = (xx.reshape(rec['history'],48,64).float()*dskip[None,:,None]).reshape(rec['history'],3072)
                    source_output = values['output'].double()
                    source_centered = source_output-source_output.mean(0)
                    for arm, Bp, Cp in (('dense96', bb.float()@R, cc.float()@R),
                        ('coordinate96', bb[:,coord], cc[:,coord]),
                        ('dual96', bb.float()@W, cc.float()@V)):
                        yp = scan(torch, code, xx, Bp, Cp, delta, A_log, dskip)
                        op = output(yp, values['gate'])
                        label = torch.tensor(rec['positions'], device='cuda')
                        arms[arm] = dict(scan_relative_RMS=metric(yp, values['y']),
                            recurrent_relative_RMS=metric(yp-skip, values['y']-skip),
                            output_relative_RMS=metric(op, values['output']),
                            centered_output_relative_RMS=metric(op.double()-op.double().mean(0), source_centered),
                            label_output_relative_RMS=metric(op[label], values['output'][label]))
                        del yp, op
                    row = dict(id=rec['id'],split=rec['split'],domain=rec['domain'],site=site,history=rec['history'],
                        reconstruction=dict(scan_relative_RMS=scan_error,output_relative_RMS=output_error,passed=ok), arms=arms)
                    measured.append(row);write(a.directory/f"{rec['id']}.site{site:02d}.metrics.json",row)
                    event(stage='assay', id=rec['id'],site=site,complete=len(measured),reconstruction=ok,arms=arms)
                    del values, xx,bb,cc,delta, yy,out, Bp,Cp,skip,source_output,source_centered
        qualified=not reconstruction_failures and len(scalar_records)==3 and all(r['passed'] for r in scalar_records)
        aggregates = {}
        for site in b['assay_sites']:
            aggregates[str(site)]={}
            for split in ('FIT','DEV'):
                rows=[r for r in measured if r['site']==site and r['split']==split]; assert len(rows)==24
                aggregates[str(site)][split]={arm:{key:dict(mean=sum(r['arms'][arm][key] for r in rows)/24,
                    worst=max(r['arms'][arm][key] for r in rows)) for key in rows[0]['arms'][arm]} for arm in ('dense96','coordinate96','dual96')}
        result=dict(schema='ORIGINAL_FALCON_RECURRENT_PROJECTION_RESULT_V1',freeze=a.freeze,binding_sha256=a.binding_sha,
            decision='LOCAL_RECURRENT_PROJECTION_MEASUREMENTS_QUALIFIED' if qualified else 'LOCAL_RECONSTRUCTION_GATE_FAIL',
            reconstruction_qualified=qualified,reconstruction_failures=reconstruction_failures,scalar_F64=scalar_records,
            bases=bases,records=measured,aggregates=aggregates,algebraic_candidate_costs=core_costs(),
            source_forwards=0,optimizer_updates=0,native_calls=0,
            optimistic_scope='All source3072 x/gate channels/48 heads and complete source generators,full norm/out_proj retained;'
                'isolates state rank,not DN512/1024 feasibility,source-to-P256 projection,head/layer composition or chatbot quality.',
            quality_admission=False,speed_admission=False,GPU_allocated_peak=torch.cuda.max_memory_allocated(),
            GPU_reserved_peak=torch.cuda.max_memory_reserved(),worker_OS_peak_snapshot=proc.memory_info().peak_wset,
            elapsed_seconds=time.monotonic()-start)
        write(a.out,result);event(stage='complete',decision=result['decision'],result_sha256=sha(a.out))
    except BaseException as error:
        write(a.directory/'first_fault.json',dict(stage=stage,error=repr(error),completed_bases=bases,
            completed_measurements=measured,elapsed_seconds=time.monotonic()-start,
            worker_OS_peak_snapshot=None if proc is None else proc.memory_info().peak_wset,
            GPU_allocated_peak=None if torch is None or not torch.cuda.is_initialized() else torch.cuda.max_memory_allocated(),
            GPU_reserved_peak=None if torch is None or not torch.cuda.is_initialized() else torch.cuda.max_memory_reserved()))
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser();m=p.add_mutually_exclusive_group(required=True)
    m.add_argument('--bind',action='store_true');m.add_argument('--launch',action='store_true');m.add_argument('--worker',action='store_true')
    p.add_argument('--binding',type=Path);p.add_argument('--binding-sha');p.add_argument('--freeze')
    p.add_argument('--directory',type=Path);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.bind:bind(a)
    elif a.launch:launch(a)
    else:worker(a)
