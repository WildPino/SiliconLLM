"""Literal506 numerical rubrics; no Torch or corpus imports."""
import struct
from pathlib import Path
import numpy as np
MASK=np.array([1,2,4,5,7,8,10,11])

def read_output(path):
    data=path.read_bytes();assert data[:8]==b'SWR32O01'
    s,t,d,enc,dec,vocab,nroutes=struct.unpack_from('<7I',data,8);offset=36
    def floats(shape):
        nonlocal offset
        count=int(np.prod(shape));result=np.frombuffer(data,dtype='<f4',count=count,offset=offset).reshape(shape).copy();offset+=count*4;return result
    a=floats((enc+2,s,d));b=floats((t,dec+2,d));c=floats((t,vocab))
    routes=np.array([struct.unpack_from('<iif',data,offset+12*i) for i in range(nroutes)],dtype=np.float64)
    assert len(data)==offset+12*nroutes
    return a,b,c,routes

def relative(a,b):
    denominator=float(np.linalg.norm(b.astype(np.float64)));assert denominator>0
    return float(np.linalg.norm(a.astype(np.float64)-b.astype(np.float64))/denominator)

def metrics(logits,target,spans):
    assert logits.shape==(14,32128) and len(target)==14 and np.asarray(spans).shape==(4,2)
    values=logits.astype(np.float64);shift=values.max(-1,keepdims=True)
    nll=np.log(np.exp(values-shift).sum(-1))+shift[:,0]-values[np.arange(14),target]
    choice=logits.argmax(-1);truth=np.asarray(spans).reshape(8)
    return {'mean_nll':float(nll.mean()),'mean_span_nll':float(nll[MASK].mean()),
            'correct_span_tokens':int(np.sum(choice[MASK]==truth)),
            'correct_fields':int(np.all((choice[MASK]==truth).reshape(4,2),axis=1).sum()),
            'all_span_teacher_forced_correct':bool(np.array_equal(choice[MASK],truth)),
            'greedy_teacher_forced_ids':choice.tolist(),'per_target_nll':nll.tolist()}

def edit_distance(a,b):
    row=list(range(len(b)+1))
    for i,x in enumerate(a):
        new=[i+1]
        for j,y in enumerate(b):new.append(min(new[-1]+1,row[j+1]+1,row[j]+(x!=y)))
        row=new
    return row[-1]

def lcs(a,b):
    row=[0]*(len(b)+1)
    for x in a:
        new=[0]
        for j,y in enumerate(b):new.append(row[j]+1 if x==y else max(new[-1],row[j+1]))
        row=new
    return row[-1]

def task(ids,truth):
    fields=[[] for _ in range(4)];expected=0;current=-1;malformed=False;closed=False
    for token in ids:
        if token==1:break
        if token==0:malformed=True;continue
        if token>=32000:
            if expected<=4 and token==32099-expected:
                if expected==4:closed=True;current=-1
                else:current=expected
                expected+=1
            else:malformed=True
        elif 0<=current<4:fields[current].append(token)
        else:malformed=True
    if malformed:fields=[[] for _ in range(4)]
    trigrams=[[(field[i],field[i+1],field[i+2]) for i in range(max(len(field)-2,0))] for field in fields]
    total=sum(len(values) for values in trigrams)
    repeated=(total-sum(len(set(values)) for values in trigrams))/max(total,1)
    healthy=closed and expected==5 and not malformed and all(fields) and ids[-1]==32095 and repeated<=.50
    exact=sum(field==gold for field,gold in zip(fields,truth))
    correct=sum(sum(i<len(field) and field[i]==gold[i] for i in range(2)) for field,gold in zip(fields,truth))
    f1=np.mean([2*lcs(field,gold)/(len(field)+len(gold)) for field,gold in zip(fields,truth)])
    prose=[token for token in ids if 1<token<32000]
    return {'healthy_complete_nonempty_fields':bool(healthy),'fields':fields,'correct_fields':exact,
            'known_token_accuracy':correct/8,'known_field_exact_accuracy':exact/4,'known_field_lcs_f1':float(f1),
            'generated_ids':ids,'generated_tokens':len(ids),'prose_tokens':len(prose),'prose_ids':prose,
            'within_field_repeated_trigram_fraction':repeated,
            'cap_reached_without_terminal':len(ids)==64 and ids[-1] not in (1,32095)}

def paired_interval(values):
    rng=np.random.default_rng(351351);values=np.asarray(values,dtype=np.float64);assert values.shape==(24,)
    draws=values[rng.integers(0,24,size=(10000,24))].mean(-1)
    return {'mean':float(values.mean()),'one_sided_lower95':float(np.quantile(draws,.05)),
            'one_sided_upper95':float(np.quantile(draws,.95)),'bootstrap_unit':'book','draws':10000,'seed':351351}
