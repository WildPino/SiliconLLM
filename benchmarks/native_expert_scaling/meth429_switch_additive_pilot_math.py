"""Tiny checker precision repair only; original427 real math remains immutable."""
import meth427_switch_additive_math as P
import meth428_switch_additive_fd_precision as J
forward=P.forward
affine=P.affine
qualify_affine=P.qualify_affine


def qualify(arm,label,pre,base,scores,chosen,ff,fn,wi,wo,head,parameters,target,tiny,path,expected_logits=None):
    if not tiny:
        return P.qualify(arm,label,pre,base,scores,chosen,ff,fn,wi,wo,head,parameters,target,tiny,path,expected_logits)
    result=J.diagnose(arm,label,pre,base,scores,chosen,ff,fn,wi,wo,head,parameters,target,tiny,path,expected_logits)
    for c in result['checks']:
        e=c['errors']
        assert e['decimal80_FD_reference']<=1e-5 and e['imaginary_reference']<=1e-5 and e['decimal80_FD_imaginary']<=1e-5,('tiny_precise_FD',label,c)
        assert e['decimal80_FD_native']<=1e-3 and e['imaginary_native']<=1e-3,('tiny_native_precise_FD',label,c)
    result['label']=label; result['only_checker_precision_changed']=True
    return result
