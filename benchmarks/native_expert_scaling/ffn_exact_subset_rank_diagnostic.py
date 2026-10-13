"""Observe the exact failed audit frame without overwriting its receipt."""
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
DOC = ROOT/'docs/research/NATIVE_EXPERT_SCALING_20260925'
sys.path[:0] = [str(ROOT/'benchmarks/native_expert_scaling'),
               str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
from falcon_auxiliary_s6_transport import extent,write


def main():
    import ffn_exact_subset_audit as audit
    import numpy as np
    start = time.monotonic()
    failed = DOC/'ffn_exact_subset_adjudication_20261013.terminal.json'
    failure_pin = extent(failed); original = json.loads(failed.read_bytes())
    assert original['exit_code'] == 1 and original['error'] == 'AssertionError()'
    out = DOC/'ffn_exact_subset_rank_discrepancy_20261013.json'; assert not out.exists()
    try:
        audit.main()
        raise RuntimeError('Expected audit assertion did not repeat')
    except AssertionError as error:
        tb = error.__traceback__; target = None
        while tb:
            if tb.tb_frame.f_code.co_name == 'main' and tb.tb_frame.f_code.co_filename == audit.__file__:
                target = tb.tb_frame.f_locals
            tb = tb.tb_next
        assert target is not None
        saved,ranking,counts,scores = [target[k] for k in ('saved','ranking','counts','scores')]
        assert saved['scores'].shape == scores.shape == (64,4608)
        directory = Path('E:/SiliconLLM/results/native_expert_scaling/ffn_exact_subset_rank_diagnostic_20261013')
        assert not directory.exists(); directory.mkdir()
        arrays = directory/'rank_discrepancy.npz'
        np.savez(arrays,saved_scores=saved['scores'],recomputed_scores=scores,saved_ranking=saved['ranking'],
                 recomputed_ranking=ranking,saved_counts=saved['FIT_counts'],recomputed_counts=counts)
        differences = []
        for cell in range(64):
            positions = np.flatnonzero(saved['ranking'][cell] != ranking[cell])
            if not len(positions): continue
            old,new = saved['ranking'][cell],ranking[cell]
            differences.append(dict(cell=cell,FIT_count=int(counts[cell]),positions=len(positions),
                first_positions=positions[:12].tolist(),
                selected_set_changes={str(k):len(set(old[:k].tolist())^set(new[:k].tolist())) for k in (32,128,512,2048)},
                largest_affected_score=float(max(saved['scores'][cell,old[positions]].max(),scores[cell,new[positions]].max())),
                max_cell_score=float(max(scores[cell].max(),saved['scores'][cell].max()))))
        assert extent(failed) == failure_pin
        report = dict(schema='FFN_EXACT_SUBSET_RANK_DIAGNOSTIC_V1',source=extent(Path(__file__)),
            failed_terminal=failure_pin,auditor=extent(Path(audit.__file__)),parent_binding=extent(target['r']['binding']['path']),
            site=target['site'],counts_equal=bool(np.array_equal(saved['FIT_counts'],counts)),
            rankings_equal=bool(np.array_equal(saved['ranking'],ranking)),
            maximum_score_absolute_difference=float(np.max(np.abs(saved['scores']-scores))),
            score_relative_difference=float(np.linalg.norm(saved['scores']-scores)/max(np.linalg.norm(scores),1e-30)),
            cells=differences,arrays=extent(arrays),seconds=time.monotonic()-start,
            scope='Replay of failed audit frame only,no producer rerun/quality admission. '
                  'Original failed terminal preserved. Full audit still required after root cause is established.')
        write(out,report); print(json.dumps(report),flush=True)


if __name__ == '__main__': main()
