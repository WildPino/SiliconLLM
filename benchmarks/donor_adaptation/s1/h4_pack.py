#!/usr/bin/env python3
"""Freeze H4 inputs and CPU anchors into a checked, non-overwriting T4 bundle."""
import hashlib
import json
import os
import shutil
import sys

HERE=os.path.dirname(os.path.abspath(__file__))
RES=os.path.join(HERE,"results","h4")
OUT=os.path.join(HERE,"_h4_bundle")
MODEL_ID="Qwen/Qwen2.5-1.5B"
REVISION="8faed761d45a263340a0528343f099c05c9a4323"
CALIB_SHA="c5509846cdc3aa44e45e77895b59a4638c49eb03790030d851e7bf1357ca4c0c"
EVAL_SHA="a1a48dc9fc5a6dc17d49cb3d16892dcf56e523f54f72eac5b63fff01b0d52f65"
TRAIN_SHA="9bb5229fad6542aa8b3fd7edfc3b8d1dff965a573d7aa8379385caacc9d614da"
BRIEF=os.path.abspath(os.path.join(HERE,"..","..","..","docs","research","donor_adaptation","briefs","BRIEF_H4_TRAIN_THE_AGGRESSIVE_RANK.md"))
CLI={"steps":4000,"bs":2,"accum":8,"lr":2e-4,"every":250,"max_hours":2.8,"seed":1717}

def sha(path):
    h=hashlib.sha256()
    with open(path,"rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()

def require(condition,message):
    if not condition: raise ValueError(message)

RUN_MD="""# H4 Stage A: rank 48 on one T4

Use the T4 only after preregistration and local factorization, self-test, intact and init
controls have passed. Run one signal session of at most 2.8 hours:

```bash
python h4_qat.py --factors h4_factors.npz --train h4_train.npz --probe h4_probe.json --out h4_trained.npz --steps 4000 --bs 2 --accum 8 --lr 2e-4 --every 250 --max-hours 2.8 --seed 1717
```

The script writes only a terminal checkpoint. Report `h4_trained.npz`,
`h4_trained.json`, and the complete log. GPU progress is not a result. CPU fp32
`h4_eval.py` adjudicates the preregistered bands. This run makes no claim about
10B, token rate, width transfer, or export. Do not select a checkpoint by its
GPU score.
"""

def main():
    h0=os.path.join(HERE,"results","h0")
    sources={
        "h4_qat.py":os.path.join(HERE,"h4_qat.py"),
        "h4_factorize.py":os.path.join(HERE,"h4_factorize.py"),
        "h4_eval.py":os.path.join(HERE,"h4_eval.py"),
        "h4_selftest.py":os.path.join(HERE,"h4_selftest.py"),
        "h4_pack.py":__file__,
        "h4_factors.npz":os.path.join(RES,"h4_factors.npz"),
        "h4_factors.json":os.path.join(RES,"h4_factors.json"),
        "h4_selftest.json":os.path.join(RES,"h4_selftest.json"),
        "h4_eval_intact.json":os.path.join(RES,"h4_eval_intact.json"),
        "h4_eval_init.json":os.path.join(RES,"h4_eval_init.json"),
        "h4_train.npz":os.path.join(h0,"h0_train.npz"),
        "h4_train.json":os.path.join(h0,"h0_train.json"),
        "h4_probe.json":os.path.join(h0,"h0_probe.json"),
        "BRIEF_H4_TRAIN_THE_AGGRESSIVE_RANK.md":BRIEF,
    }
    for name,path in sources.items():require(os.path.isfile(path),"missing "+name+": "+path)
    require(not os.path.exists(OUT),"bundle already exists; preserve it: "+OUT)
    meta=json.load(open(sources["h4_factors.json"],encoding="utf-8"))
    test=json.load(open(sources["h4_selftest.json"],encoding="utf-8"))
    intact=json.load(open(sources["h4_eval_intact.json"],encoding="utf-8"))
    init=json.load(open(sources["h4_eval_init.json"],encoding="utf-8"))
    train=json.load(open(sources["h4_train.json"],encoding="utf-8"))
    probe=json.load(open(sources["h4_probe.json"],encoding="utf-8"))
    require(all(x.get("model")==MODEL_ID and x.get("revision")==REVISION for x in (meta,intact,init,train,probe)),"model/revision mismatch")
    require(meta.get("stage")==test.get("stage")==intact.get("stage")==init.get("stage")=="H4","H4 stage mismatch")
    require(meta.get("smoke") is False and meta.get("rank")==48 and meta.get("d_model")==1536 and meta.get("layers")==list(range(28)) and meta.get("organs")==["q_proj","o_proj"] and len(meta.get("rows",[]))==56 and meta.get("trainable_params")==8_260_224,"full factor shape/parameter metadata mismatch")
    gate=meta.get("G_H4a",{})
    require(gate.get("fires") is True and gate.get("bad_organs")==[] and gate.get("worst_scale_rel")==0 and gate.get("worst_prod_rel")==0,"G-H4a failed")
    cal=meta.get("calib_slice",{})
    require(cal.get("part")=="calib" and cal.get("n_seq")==32 and cal.get("seq_len")==512 and cal.get("seed")==42424 and cal.get("ids_sha256")==CALIB_SHA,"calibration slice mismatch")
    fac_hash=sha(sources["h4_factors.npz"])
    require(test.get("passes") is True and test.get("full_bundle") is True and test.get("smoke") is False and test.get("rank")==48 and test.get("organs")==56 and test.get("bundle_sha256")==fac_hash and test.get("factor_meta_sha256")==sha(sources["h4_factors.json"]),"self-test bundle/result mismatch")
    require(all(test.get("G_H4"+k,{}).get("passes") is True for k in "bcdfg"),"H4 self-test subgate failed")
    for label,record,mode in (("intact",intact,"intact"),("init",init,"factored_init")):
        ev=record.get("eval_slice",{})
        require(record.get("mode")==mode and record.get("counted")==160 and ev.get("part")=="heldout" and ev.get("n_seq")==24 and ev.get("seq_len")==512 and ev.get("seed")==1234 and ev.get("ids_sha256")==EVAL_SHA,label+" eval slice mismatch")
    require(intact.get("rank") is None and intact.get("factors_sha256") is None and intact.get("intact_control",{}).get("passes") is True and intact.get("free")==160 and intact.get("teacher_forced")==160 and abs(intact.get("bpb",float("inf"))-.7675949641196624)<=1e-7,"intact control failed")
    require(init.get("rank")==48 and init.get("factors_sha256")==fac_hash and "bands" not in init,"init factors hash/rank/mode mismatch")
    require(train.get("smoke") is False and train.get("part")=="calib" and train.get("seed")==90011 and train.get("seq_len")==512 and train.get("n_seq")==31250 and train.get("ids_sha256")==TRAIN_SHA and train.get("eval_slice_sha256_NOT_THIS")==EVAL_SHA,"training stream metadata mismatch")
    require(probe.get("n_new")==32 and probe.get("counted")==160 and len(probe.get("prompt_ids",[]))==len(probe.get("target_ids",[]))==5 and all(len(x)==32 for x in probe["target_ids"]),"probe mismatch")
    os.mkdir(OUT)
    files={}
    for name,src in sources.items():
        dst=os.path.join(OUT,name);source_hash=sha(src);shutil.copy2(src,dst)
        require(sha(dst)==source_hash,"copied file hash mismatch: "+name)
        files[name]={"sha256":source_hash,"bytes":os.path.getsize(dst),"source":os.path.abspath(src)}
    run=os.path.join(OUT,"RUN.md")
    with open(run,"x",encoding="utf-8") as f:f.write(RUN_MD)
    files["RUN.md"]={"sha256":sha(run),"bytes":os.path.getsize(run),"source":"generated by h4_pack.py"}
    manifest={"stage":"H4","model":MODEL_ID,"revision":REVISION,"rank":48,"seed":1717,"cli":CLI,"payload":{"trainer":"h4_qat.py","factors":"h4_factors.npz","train":"h4_train.npz","probe":"h4_probe.json","output":"h4_trained.npz"},"input_hashes":{"factors_sha256":fac_hash,"factor_meta_sha256":files["h4_factors.json"]["sha256"],"selftest_sha256":files["h4_selftest.json"]["sha256"],"intact_eval_sha256":files["h4_eval_intact.json"]["sha256"],"init_eval_sha256":files["h4_eval_init.json"]["sha256"],"pack_source_sha256":files["h4_pack.py"]["sha256"]},"calib_slice":cal,"eval_slice":init["eval_slice"],"files":files}
    with open(os.path.join(OUT,"MANIFEST.json"),"x",encoding="utf-8") as f:json.dump(manifest,f,indent=1)
    print("wrote",OUT,"with",len(files),"hashed files",flush=True)
    return 0

if __name__=="__main__":
    try:sys.exit(main())
    except (OSError,ValueError,KeyError) as exc:
        print("REFUSE H4 pack:",exc,flush=True)
        sys.exit(2)
