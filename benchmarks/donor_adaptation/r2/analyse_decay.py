"""Analysis of the Amendment-1 decay/window arms against the main sweep's floors."""
import json, numpy as np
M=json.load(open('results/r2a_main.json')); D=json.load(open('results/r2a_decay.json'))
mi={(r['layer'],r['head'],r['arm']):r for r in M['rows']}
di={(r['layer'],r['head'],r['arm']):r for r in D['rows']}
al={**mi,**di}
L=sorted(set(r['layer'] for r in D['rows'])); H=sorted(range(12))
arms=list(dict.fromkeys(r['arm'] for r in D['rows']))
print("ACHIEVED envelope:", json.dumps(D['achieved_envelope']))
print("T/D achieved:", D['rows'][0]['TD_achieved'], " rows:", len(D['rows']), " wall_s:", round(D['wall_s']))
print("\nPOOLED median residual")
for a in arms:
    v=[r['residual'] for r in D['rows'] if r['arm']==a]
    print(f"  {a:22s} n={len(v):4d}  median {np.median(v):.4f}  p25 {np.percentile(v,25):.4f}  p75 {np.percentile(v,75):.4f}")
print("\nREFERENCE floors from the main sweep, same T/D, same layers/heads:")
for a in ['C6_causal_uniform','C8_no_mixing','elu1','taylor2','C5p_wrong_sequence']:
    v=[mi[(l,h,a)]['residual'] for l in L for h in H if (l,h,a) in mi]
    print(f"  {a:22s} n={len(v):4d}  median {np.median(v):.4f}")
def paired(a,b,lay=None):
    lay=lay or L
    ks=[(l,h) for l in lay for h in H if (l,h,a) in al and (l,h,b) in al]
    ds=np.array([al[(l,h,a)]['residual']-al[(l,h,b)]['residual'] for l,h in ks])
    return ds
print("\nPAIRED (negative = first arm better)")
print(f"{'comparison':52s} {'n':>4s} {'mean':>9s} {'sd':>8s} {'median':>9s} {'better':>10s}")
SUB=[0,7,14,21,27]
pairs=[]
for a in arms:
    if a in ('favor_d14','elu1_noscale','hedgehog0_noscale'): continue
    for b in ['C6_causal_uniform','C8_no_mixing','elu1']:
        pairs.append((a,b,None))
    if a.startswith('decay_'): pairs.append((a,'decayonly_g'+a.split('_g')[1],None))
    if a.startswith('win_'):   pairs.append((a,'winonly_w'+a.split('_w')[1],None))
for a in ['favor_d14']:
    for b in ['favor','C6_causal_uniform','C8_no_mixing']: pairs.append((a,b,None))
for a in ['elu1_noscale','hedgehog0_noscale']:
    pairs.append((a,a.replace('_noscale',''),SUB)); pairs.append((a,'C6_causal_uniform',SUB))
seen=set()
for a,b,lay in pairs:
    if (a,b) in seen or a==b: continue
    seen.add((a,b))
    ds=paired(a,b,lay)
    if len(ds)==0: continue
    print(f"{a+' - '+b:52s} {len(ds):4d} {ds.mean():+9.4f} {ds.std():8.4f} {np.median(ds):+9.4f} {int((ds<0).sum()):5d}/{len(ds):<4d}")
print("\nENTROPY STRATA (median), decay/window arms + reference floors")
ST=['peaked_d1','mid','diffuse_d10']
print(f"{'arm':22s} " + " ".join(f"{s:>13s}" for s in ST) + f" {'peaked-diffuse':>16s}")
for a in arms+['C6_causal_uniform','C8_no_mixing','elu1','taylor2']:
    ks=[(l,h) for l in L for h in H if (l,h,a) in al]
    if not ks: continue
    g=lambda s: np.median([al[(l,h,a)][f'res_{s}'] for l,h in ks])
    dd=np.array([al[(l,h,a)]['res_peaked_d1']-al[(l,h,a)]['res_diffuse_d10'] for l,h in ks])
    print(f"{a:22s} " + " ".join(f"{g(s):13.4f}" for s in ST) + f" {dd.mean():+16.4f}")
print("\nPEAKED-STRATUM paired vs C8_no_mixing (the comparison that decides)")
for a in arms:
    ks=[(l,h) for l in L for h in H if (l,h,a) in al and (l,h,'C8_no_mixing') in al]
    if not ks: continue
    ds=np.array([al[(l,h,a)]['res_peaked_d1']-al[(l,h,'C8_no_mixing')]['res_peaked_d1'] for l,h in ks])
    print(f"  {a:22s} mean {ds.mean():+.4f}  better in {int((ds<0).sum())}/{len(ds)}")
print("\nRANK / gamma ACHIEVED per arm")
for a in arms:
    rs=[r for r in D['rows'] if r['arm']==a]
    print(f"  {a:22s} rank_Z median {np.median([r['rank_Z'] for r in rs]):6.0f}  gamma_req {rs[0]['gamma_requested']}  window_req {rs[0]['window_requested']}")
