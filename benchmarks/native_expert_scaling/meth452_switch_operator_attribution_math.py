"""Discrete WI/WO factorial readout contrasts, including baseline and interaction."""
import numpy as np


def pair_loses(gap, a, b):
    return bool(gap < 0 or (gap == 0 and b < a))


def terms(original, reference, wi_only, wo_only, compact, a, b):
    assert a != b
    values = [v.astype(np.float64) for v in (original, reference, wi_only, wo_only, compact)]
    z0,z00,z10,z01,z11 = values
    base=z00-z0;wi=z10-z00;wo=z01-z00;interaction=z11-z10-z01+z00
    residual=float(np.max(np.abs(base+wi+wo+interaction-(z11-z0))))
    assert residual <= 1e-10, 'full_vocabulary_factorial_identity'
    contrast=lambda v:float(v[b]-v[a])
    m=float(z0[a]-z0[b]);gap=float(z11[a]-z11[b])
    parts={'baseline':contrast(base),'WI':contrast(wi),'WO':contrast(wo),'interaction':contrast(interaction)}
    other=m-parts['baseline']-parts['WI']-parts['WO']-parts['interaction']
    assert abs(other-gap)<=1e-10, 'paired_margin_factorial_identity'
    return {'original_pair_gap':m,'reference_pair_gap':float(z00[a]-z00[b]),
            'WI_only_pair_gap':float(z10[a]-z10[b]),'WO_only_pair_gap':float(z01[a]-z01[b]),
            'compact_pair_gap':gap,'parts':parts,'pair_residual':float(other-gap),
            'max_full_vocab_identity_residual':residual,
            'pair_gap_without_explicit_interaction':m-parts['baseline']-parts['WI']-parts['WO']}


def tiny_qualification():
    arrays=[np.asarray(x,np.float32) for x in ([10,7,0],[10,7,0],[9,7,2],[10,9,-1],[8,10,4])]
    result=terms(*arrays,0,1)
    assert result['parts']=={'baseline':0.,'WI':1.,'WO':2.,'interaction':2.}
    assert result['original_pair_gap']==3 and result['compact_pair_gap']==-2
    # Independent scalar bookkeeping and signed/tie ordering.
    assert 3-(0+1+2+2)==-2
    assert pair_loses(-1,0,1) and not pair_loses(1,0,1)
    assert not pair_loses(0,0,1) and pair_loses(0,1,0)
    other=[np.asarray(x,np.float32) for x in ([4,3],[5,3],[4,3],[5,5],[5,4])]
    result=terms(*other,0,1);assert result['parts']=={'baseline':-1.,'WI':1.,'WO':2.,'interaction':-2.}
    assert result['compact_pair_gap']==1
    return {'known_nonzero_main_joint_and_baseline_terms_exact':True,
            'native_selected_pair_lowest_ID_ties_qualified':True}
