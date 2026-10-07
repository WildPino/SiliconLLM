"""Separate certificate/statistical arithmetic; no main transformation import."""
import math
import numpy as np


def dot(a,b):return np.einsum('ij,ij->i',a,b,optimize=False)
def contraction(a,b):return np.einsum('ij,kj->ik',a,b,optimize=False)


def identity_bound(inp,h32,wi,si,wo,so,u,f,l,abs_atoms,abs_m):
    h=np.nextafter(np.nextafter(h32.astype('<f8')+math.ldexp(1.,-149),math.inf)/(1.-math.ldexp(1.,-24)),math.inf)
    o=wo.astype('<f8')*so.astype('<f8')[:,None];x=np.abs(inp['q'].astype('<f8')*inp['alpha'].astype('<f8')[:,None])
    source=contraction(h,np.abs(o));old=contraction(x,np.abs(l))+contraction(h[:,u],np.abs(o[:,u]))
    ii=np.abs(wi[f].astype('<f8')*si[f].astype('<f8')[:,None])
    fold=contraction(contraction(x,ii),np.abs(o[:,f]))
    bound=1e-10*(1+source+old+abs_atoms+abs_m)+5e-12*(np.sum(x,axis=1)[:,None]+fold)
    gamma=65536/((1<<53)-65536);return np.nextafter(bound*(1+8*gamma),math.inf)


def metrics(g,negative,physical,old,ref,target,p,bound):
    pp=old[:,0]-g;rr=old[:,0]-old[:,2]
    ratio=np.max(np.abs(rr-pp-negative)/bound,axis=1);assert np.max(ratio)<=1
    blocks=[];tolerances=[]
    for weighted in (False,True):
        factor=p.astype('<f8')[:,None] if weighted else 1.;truth=target.astype('<f8') if weighted else ref.astype('<f8')
        a=np.multiply(physical[:,0],p[:,None],dtype=np.float32).astype('<f8') if weighted else physical[:,0].astype('<f8')
        rot=np.multiply(physical[:,1],p[:,None],dtype=np.float32).astype('<f8') if weighted else physical[:,1].astype('<f8')
        oldp=np.multiply(old[:,4].astype('<f4'),p[:,None],dtype=np.float32).astype('<f8') if weighted else old[:,4]
        parts=[truth,g*factor-truth,a-truth,rot-truth,oldp-truth,pp*factor,negative*factor,
               a-g*factor,old[:,0]*factor-truth,rr*factor]
        pairs=[(v,v) for v in parts[:7]]+[(parts[5],parts[6])]+[(v,v) for v in parts[7:]]
        blocks.append(np.column_stack((*[dot(a,b) for a,b in pairs],ratio)))
        tolerances.append(np.column_stack((*[3e-11*np.maximum(1,np.sum(np.abs(a*b),axis=1)) for a,b in pairs],np.ones(len(g)))))
    return np.column_stack(blocks),np.column_stack(tolerances)


def eligibility(views):
    role=[v for v in views if v['kind']=='role'];rare=[v for v in views if v['kind']=='rare' and v['split']=='consumed_validation' and v['development_class'] in ('1..4','5..15')]
    consumed=[v for v in views if v['kind']=='uid' and v['split']=='consumed_validation'];assert len(role)==6 and len(consumed)==1
    answer={}
    for prefix,arm in (('unweighted','continuous_atoms'),('unweighted','physical_atoms'),('weighted','physical_atoms')):
        answer[prefix+'_'+arm+'_ALL_six_RMS_1pct']=all(v['count']>0 and isinstance(v[prefix+'_'+arm+'_RMS'],float) and v[prefix+'_'+arm+'_RMS']<=.01 for v in role)
    answer['ALL_nonempty_rare_consumed_physical_RMS_1pct']=all(v['count']==0 or all(v[p+'_physical_atoms_RMS'] is not None and v[p+'_physical_atoms_RMS']<=.01 for p in ('unweighted','weighted')) for v in rare)
    answer['ALL_six_selected_FFN_bytes_including_scales_IDs_header_le75pct_source']=all(v['count']>0 and v['selected_FFN_byte_ratio']<=.75 for v in role)
    answer['consumed_AND_ALL_six_source_informed_physical_strictly_better_than_matched_rotation']=all(v['count']>0 and all(v[p+'_physical_atoms_RMS'] is not None and v[p+'_physical_rotation_RMS'] is not None and v[p+'_physical_atoms_RMS']<v[p+'_physical_rotation_RMS'] for p in ('unweighted','weighted')) for v in consumed+role)
    return answer
