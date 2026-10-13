"""CPU-only StrictSink stream segmentation/mismatch/truncation tests."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'benchmarks/native_expert_scaling'),
               str(ROOT/'results/native_expert_scaling/chatbot_source_runtime/site')]
from ffn_operand_capture import StrictSink
from falcon_auxiliary_s6_transport import extent,write


def main(a):
    import numpy as np
    assert not a.directory.exists() and not a.out.exists(); a.directory.mkdir()
    fixture = np.array([[0x3f80,0xbf80,0x4000],[0x4040,0x4080,0x40a0]],'u2')
    path = a.directory/'fixture.bf16'; fixture.tofile(path)
    descriptor = dict(**extent(path),shape=[2,3],dtype='BF16')
    sink = StrictSink(descriptor,np)
    assert sink.write(fixture[0,:2].tobytes()) == 4
    assert sink.write(fixture.reshape(-1)[2:5].tobytes()) == 6
    assert sink.write(fixture.reshape(-1)[5:].tobytes()) == 2
    sink.close(); exact = sink.digest.hexdigest() == descriptor['sha256']
    wrong = fixture.copy(); wrong[0,0] ^= 1; mismatch = False
    try: StrictSink(descriptor,np).write(wrong.tobytes())
    except AssertionError: mismatch = True
    short = StrictSink(descriptor,np); short.write(fixture[:1].tobytes()); truncated = False
    try: short.close()
    except AssertionError: truncated = True
    closed = False
    try: sink.write(b'\x00\x00')
    except AssertionError: closed = True
    assert exact and mismatch and truncated and closed
    report = dict(schema='FFN_OPERAND_CAPTURE_STREAM_SMOKE_V1',source=extent(Path(__file__)),
        collector_source=extent(ROOT/'benchmarks/native_expert_scaling/ffn_operand_capture.py'),
        fixture=descriptor,segmented_write_exact=exact,mismatch_rejected=mismatch,
        truncation_rejected=truncated,write_after_close_rejected=closed,
        source_model_forwards=0,GPU_calls=0,
        scope='Strict readonly stream verifier wiring only;not source FFN/model/numerical qualification.')
    write(a.out,report); print(json.dumps(dict(result=extent(a.out),all_pass=True)),flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for k in ('directory','out'): p.add_argument('--'+k,type=Path,required=True)
    main(p.parse_args())
