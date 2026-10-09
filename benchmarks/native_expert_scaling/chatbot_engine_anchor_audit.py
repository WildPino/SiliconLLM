"""Read-only reconciliation of the original engine and the current converter.

Integer shape accounting only: no tensors, inference, fitting or native timing.
The current packed export supplies an observed payload anchor, not quality.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT / 'docs/research/NATIVE_EXPERT_SCALING_20260925'


def identity(path):
    raw = path.read_bytes()
    return {'path': str(path.resolve()), 'bytes': len(raw),
            'sha256': hashlib.sha256(raw).hexdigest()}


def constants(raw, names):
    result = {}
    for name in names:
        matches = re.findall(r'^#define\s+' + re.escape(name) + r'\s+(\d+)\s*$', raw, re.M)
        assert len(matches) == 1, (name, matches)
        result[name] = int(matches[0])
    return result


def profile(c, swa, core_kind, n):
    d, l, dn, state = (c[k] for k in ('D', 'L', 'DN', 'N'))
    k = c.get('KTOP', c.get('K'))
    h = c.get('HID_E', c.get('H'))
    if core_kind == 'original':
        rank = c['DTR']
        core = (l-swa)*(3*dn*d + dn*(rank+2*state) + dn*rank) + swa*4*d*d
        conv_dim = dn
    else:
        projected = 2*dn+2*state+c['NH']
        core = (l-swa)*(projected*d+dn*d)+swa*4*d*d
        conv_dim = dn+2*state
    bank = 3*l*d*h*n
    active = 3*l*d*h*k
    route = l*n*d
    head = c['V']*d
    scales = 4*l*n*(2*h+d)
    router = 4*l*n*(d+1)
    state_bytes = 4*((l-swa)*(dn*state+conv_dim*c['CONV'])+swa*2*c['WIN']*d)
    return dict(D=d, L=l, n_per_layer=n, total_expert_functions=l*n,
                k=k, h=h, V=c['V'], core_kind=core_kind,
                bank_coefficients=bank, active_expert_products=active,
                core_matrix_products=core, full_head_products=head,
                flat_router_products=route, flat_router_coefficient_bytes=router,
                bank_pair_code_bytes=bank//2, bank_scale_bytes=scales,
                selected_pair_code_bytes=active//2,
                selected_scale_bytes=4*l*k*(2*h+d),
                matrix_products_total=core+head+active+route,
                persistent_state_payload_bytes=state_bytes,
                legacy_expert_copies_bytes_before_scales=5*bank+bank//2,
                all_bank_F32_parameter_gradient_Adam_bytes=16*bank)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--freeze', required=True)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    start = time.monotonic()
    paths = [Path(__file__), Path(sys.executable),
             ROOT/'benchmarks/phase60/engine.c',
             ROOT/'benchmarks/native_expert_scaling/chatbot_hybrid_native.c',
             ROOT/'benchmarks/native_expert_scaling/chatbot_hybrid_target.py',
             DOC/'chatbot_hybrid_export_result_20261008.json',
             DOC/'chatbot_hybrid_export_result_20261008.terminal.json',
             DOC/'chatbot_hybrid_engine_probe_result_20261009.json',
             DOC/'CHATBOT_ENGINE_ANCHOR_AUDIT_PROTOCOL_20261009.md',
             ROOT/'results/native_expert_scaling/falcon_1p5b_source_repair1_20261008/config.json']
    before = [identity(p) for p in paths]
    old_text, new_text = (p.read_text(encoding='utf-8') for p in paths[2:4])
    old = constants(old_text, ('D','L','DN','N','V','DTR','CONV','WIN','KTOP','HID_E'))
    new = constants(new_text, ('D','L','DN','N','V','NH','HD','CONV','WIN','K','H','E'))
    match = re.search(r'static int E=(\d+);', old_text)
    assert match
    old_n = int(match[1])
    source = json.loads(paths[-1].read_bytes())
    source_ffn = 3*source['hidden_size']*source['intermediate_size']*source['num_hidden_layers']
    export = json.loads(paths[5].read_bytes())
    receipt = json.loads(paths[6].read_bytes())
    assert receipt['exit_code'] == 0 and receipt['resource_gates']
    assert receipt['result_sha256'] == before[5]['sha256']
    assert export['expert_master_or_unpacked_bytes'] == 0
    assert new['DN'] == new['NH']*new['HD']
    original = profile(old, 1, 'original', old_n)
    current = profile(new, 2, 'head_shared_scan', new['E'])
    assert current['bank_pair_code_bytes'] == export['expert_code_bytes']
    assert current['persistent_state_payload_bytes'] == 9117696
    fixed_payload = export['payload_bytes'] - sum(current[k] for k in
        ('bank_pair_code_bytes','bank_scale_bytes','flat_router_coefficient_bytes'))
    metadata_bytes = export['model']['bytes']-export['payload_bytes']
    per_n = 3*new['L']*new['D']*new['H']
    rows = []
    for label, n in [('current',new['E']), ('source_FFN_coefficient_count',(source_ffn+per_n-1)//per_n),
                     ('twice_source_FFN_coefficient_count',(2*source_ffn+per_n-1)//per_n),
                     ('10B_bank',(10_000_000_000+per_n-1)//per_n),
                     ('100B_bank',(100_000_000_000+per_n-1)//per_n)]:
        row = profile(new, 2, 'head_shared_scan', n)
        packed = fixed_payload+metadata_bytes+sum(row[k] for k in
            ('bank_pair_code_bytes','bank_scale_bytes','flat_router_coefficient_bytes'))
        row.update(label=label, packed_bytes_same_tensor_table_proposal=packed,
                   accepted_by_current_fixed_E=(n==new['E']),
                   within_current_512MiB_blob_cap=(packed <= 512*1024**2),
                   source_FFN_coefficient_count_ratio=row['bank_coefficients']/source_ffn,
                   flat_router_to_selected_products=row['flat_router_products']/row['active_expert_products'])
        rows.append(row)
    assert rows[0]['packed_bytes_same_tensor_table_proposal'] == export['model']['bytes']
    assert current['matrix_products_total'] == 69632512
    assert original['active_expert_products'] == 4718592
    ratios = {key: current[key]/original[key] for key in
              ('core_matrix_products','active_expert_products','full_head_products',
               'matrix_products_total','persistent_state_payload_bytes')}
    observations = {
        'original_full_F32_expert_arrays_retained': all(s in old_text for s in
            ('egate_f[l]=rd(f','eup_f[l]=rd(f','eWd_f[l]=rd(f')),
        'original_unpacked_expert_arrays_retained': all(s in old_text for s in
            ('egate_wt[l]=rdi8(f','eup_wt[l]=rdi8(f','eWd_wt[l]=rdi8(f')),
        'current_fixed_E_header_contract': 'uint32_t want[14]={1,D,L,V,E,K,H,DN,N,NH,HD,CONV,WIN,' in new_text,
        'current_512MiB_loader_cap': 'blob_bytes>512ULL*1024*1024' in new_text,
        'original_flat_score_loop': 'for(int e=0;e<E;e++) rp[e]=dotf(router_w[l]' in old_text}
    assert all(observations.values()), observations
    after = [identity(p) for p in paths]
    assert before == after, 'input mutated'
    elapsed = time.monotonic()-start
    assert elapsed <= 10
    result = dict(schema='ENGINE_ANCHOR_AUDIT_V1', freeze=a.freeze,
        scope='static code and exact integer deductions; no new tensor/model/timing observations',
        inputs=before, input_hashes_unchanged=True, code_observations=observations,
        original=original, current=current, current_to_original_ratios=ratios,
        source_FFN_coefficients=source_ffn, current_bank_to_source_FFN_count=current['bank_coefficients']/source_ffn,
        fixed_current_payload_bytes=fixed_payload, model_metadata_bytes=metadata_bytes,
        capacity_profiles=rows, elapsed_seconds_before_serialization=elapsed,
        local_FFN_fidelity_is_required_for_joint_chatbot_transfer=False,
        quality_admission=False, physical_DRAM_measurement=False, new_speed_measurement=False,
        limitations=['Coefficient counts and trace energy are not knowledge fractions.',
            'Matrix counts exclude scalar nonlinearities, recurrence work, normalization, routing selection, interface and prefill.',
            'Payload projections assume the current tensor table and flat F32 router, not an implemented new format.',
            '16 bytes per bank coefficient counts F32 parameter, gradient and two Adam moments only; it excludes all other tensors.',
            'No full model or checkpoint tensor was read; input SHA checks are file provenance, not new operator qualification.',
            'All old gates/failures remain recorded; local FFN thresholds describe their original experiment.'])
    raw = (json.dumps(result,indent=2,ensure_ascii=False)+'\n').encode('utf-8')
    assert len(raw) <= 1024*1024
    with a.out.open('xb') as f:
        f.write(raw)
    print(json.dumps({'out': str(a.out.resolve()), 'sha256': hashlib.sha256(raw).hexdigest(),
                      'bytes': len(raw), 'seconds_before_serialization': elapsed,
                      'current_to_original': ratios, 'scope': result['scope']}))


if __name__ == '__main__':
    main()
