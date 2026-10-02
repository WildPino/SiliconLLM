#!/usr/bin/env python3
"""Read completed record headers of a live capture without inspecting vectors."""
import argparse
import json
from pathlib import Path
import struct

def main():
    ap=argparse.ArgumentParser();ap.add_argument('path',type=Path)
    args=ap.parse_args();p=args.path
    if not p.exists():
        print(json.dumps({'capture_exists':False}));return
    size=p.stat().st_size;rows=0;last=None;footer=False
    with p.open('rb') as f:
        assert f.read(8)==b'M298ACT1'
        while f.tell()+32<=size:
            raw=f.read(32)
            if len(raw)<32:break
            l,o,e,c,b,s,t,w=struct.unpack('<8I',raw)
            if l==0xffffffff:
                footer=f.tell()+8<=size;break
            assert l in (1,13,25) and o<3 and e in (0,32,63)
            assert b<4 and s<4 and t<128 and w==(1280 if o==2 else 1536)
            if f.tell()+w*4>size:break
            f.seek(w*4,1);rows+=1;last={'chunk_index':c,'batch_index':b,'layer':l,'organ':o}
    print(json.dumps({'bytes_at_open':size,'completed_records_at_open':rows,
                      'last_record_header':last,'footer_visible':footer,
                      'scope':'Progress only; no closure or spectral result.'}))

if __name__=='__main__':main()
