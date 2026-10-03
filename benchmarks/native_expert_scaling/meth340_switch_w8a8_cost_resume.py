"""Unchanged339 experiment after byte serialization repair, new result paths."""
from pathlib import Path
import meth339_switch_w8a8_cost as R

PROTOCOL=R.M.DOC/'METH_340_SWITCH_W8A8_COST_RESUME_PROTOCOL_20261003.md'
FAILURE=R.M.DOC/'meth339_switch_w8a8_cost_result.failure.json'
FAILURE_SHA='7470a8ee05dc0c8076c62735eed2fff0557b21f1099b32e20017d6e940e4e0ba'
ORIGINAL_PROTOCOL=R.PROTOCOL


def main():
    for path in (Path(__file__),PROTOCOL,FAILURE,Path(R.__file__),ORIGINAL_PROTOCOL):R.M.committed(path)
    assert R.M.digest(FAILURE)==FAILURE_SHA
    original_write=R.M.write
    def write(path,data):
        data.update({'experiment':'METH-340-unchanged339-cost-after-byte-serialization-repair',
                     'resume_wrapper_sha256':R.M.digest(__file__),'preserved339_failure_sha256':FAILURE_SHA,
                     'unchanged339_protocol_sha256':R.M.digest(ORIGINAL_PROTOCOL)})
        original_write(path,data)
    R.M.write=write;R.OUT=R.M.ROOT/'results/native_expert_scaling/meth340_switch_w8a8_cost_resume';R.PROTOCOL=PROTOCOL
    R.main()


if __name__=='__main__':main()
