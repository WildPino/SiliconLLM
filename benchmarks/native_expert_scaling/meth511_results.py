"""Prospectively fixed whole quality, accepted rates and matched cost decisions."""
import numpy as np
import meth511_quality as Q

def summarize(cases):
    assert len(cases)==96
    def interval(fn):return Q.paired_interval([float(np.mean([fn(c) for c in cases[4*i:4*i+4]])) for i in range(24)])
    paired={}
    for k in ['mean_nll','mean_span_nll']:
        paired[k]=interval(lambda c:c['candidate_prediction'][k]-c['donor_prediction'][k])
    paired['teacher_masked_token_accuracy']=interval(lambda c:(c['candidate_prediction']['correct_span_tokens']-c['donor_prediction']['correct_span_tokens'])/8)
    paired['teacher_field_exact_accuracy']=interval(lambda c:(c['candidate_prediction']['correct_fields']-c['donor_prediction']['correct_fields'])/4)
    for k in ['known_token_accuracy','known_field_exact_accuracy','known_field_lcs_f1','healthy_complete_nonempty_fields','within_field_repeated_trigram_fraction']:
        paired['generation_'+k]=interval(lambda c:float(c['candidate_task'][k])-float(c['donor_task'][k]))
    paired['prose_edit']=interval(lambda c:c['normalized_prose_edit'])
    top1=float(np.mean([c['top1_agreement'] for c in cases]));masked=float(np.mean([c['masked_top1_agreement'] for c in cases]))
    signal=float(np.mean([c['donor_task']['known_field_exact_accuracy'] for c in cases]));donor_health=float(np.mean([c['donor_task']['healthy_complete_nonempty_fields'] for c in cases]));health=float(np.mean([c['candidate_task']['healthy_complete_nonempty_fields'] for c in cases]))
    upper=lambda k,v:paired[k]['one_sided_upper95']<=v
    lower=lambda k,v:paired[k]['one_sided_lower95']>=v
    quality={'all_nll_upper_le.05':upper('mean_nll',.05),'masked_nll_upper_le.05':upper('mean_span_nll',.05),
             'teacher_masked_accuracy_lower_ge-.02':lower('teacher_masked_token_accuracy',-.02),'teacher_field_accuracy_lower_ge-.05':lower('teacher_field_exact_accuracy',-.05),
             'all_top1_ge.95':top1>=.95,'masked_top1_ge.95':masked>=.95,'donor_field_signal_ge.10':signal>=.10,'donor_health_ge.80':donor_health>=.80,'candidate_health_ge.80':health>=.80,
             'generation_token_accuracy_lower_ge-.02':lower('generation_known_token_accuracy',-.02),'generation_field_accuracy_lower_ge-.05':lower('generation_known_field_exact_accuracy',-.05),
             'generation_LCS_lower_ge-.02':lower('generation_known_field_lcs_f1',-.02),'generation_health_lower_ge-.05':lower('generation_healthy_complete_nonempty_fields',-.05),
             'generation_repeat_upper_le.05':upper('generation_within_field_repeated_trigram_fraction',.05),'prose_edit_upper_le.10':upper('prose_edit',.10)}
    rates={};times={}
    ix=np.random.default_rng(485485).integers(0,24,size=(10000,24))
    for arm in ['source','candidate']:
        ts=[];cold=[];load=[];ordinary=[];prose=[]
        for bi in range(24):
            group=cases[4*bi:4*bi+4];t=l=coldtime=oc=pc=0.
            for c in group:
                rows=c['native'][arm]['generation'];t+=float(np.mean([r['full_generation_seconds'] for r in rows if r['repetition']>=0]));l+=rows[0]['load_seconds'];coldtime+=rows[0]['load_seconds']+rows[0]['full_generation_seconds']
                task=c[arm+'_task'];oc+=task['generated_tokens'] if task['healthy_complete_nonempty_fields'] else 0;pc+=task['prose_tokens'] if task['healthy_complete_nonempty_fields'] else 0
            ts.append(t);load.append(l);cold.append(coldtime);ordinary.append(oc);prose.append(pc)
        times[arm]=ts;rates[arm]={'book_seconds':ts,'sum_case_mean_seconds':sum(ts),'sum_load_seconds':sum(load),'sum_first_request_with_load_seconds':sum(cold),'bootstrap_unit':'book','draws':10000,'seed':485485}
        for name,nums in [('ordinary',ordinary),('prose',prose)]:
            draws=np.array(nums)[ix].sum(-1)/np.array(ts)[ix].sum(-1);total=sum(nums)
            rates[arm][name]={'accepted_ids':int(total),'warm_rate':total/sum(ts),'warm_lower95':float(np.quantile(draws,.05)),'warm_upper95':float(np.quantile(draws,.95)),'first_request_load_charged_rate':total/sum(cold),'amortized_rates_requests_per_process':{str(k):total/(sum(ts)+sum(load)/k) for k in [1,10,100]}}
    s=np.array(times['source']);c=np.array(times['candidate']);ratios=c/s;draws=c[ix].sum(-1)/s[ix].sum(-1)
    cost={'ratio_of_sums':float(c.sum()/s.sum()),'book_ratios':ratios.tolist(),'bootstrap_lower95':float(np.quantile(draws,.05)),'bootstrap_upper95':float(np.quantile(draws,.95)),'regressing_books':np.flatnonzero(ratios>1).tolist(),'worst_book_ratio':float(ratios.max()),'seed':485485,'draws':10000}
    economics={'whole_mean_ratio_le.95':cost['ratio_of_sums']<=.95,'whole_ratio_upper_le1':cost['bootstrap_upper95']<=1,'all_book_ratio_le1':bool(np.all(ratios<=1)),
               'candidate_ordinary_warm_and_lower_ge50':rates['candidate']['ordinary']['warm_rate']>=50 and rates['candidate']['ordinary']['warm_lower95']>=50,
               'candidate_prose_warm_and_lower_ge50':rates['candidate']['prose']['warm_rate']>=50 and rates['candidate']['prose']['warm_lower95']>=50}
    return {'paired':paired,'donor_field_signal':signal,'donor_health':donor_health,'candidate_health':health,'all_top1_agreement':top1,'masked_top1_agreement':masked,'quality_gates':quality,'rates':rates,'matched_cost':cost,'economic_gates':economics,'whole_recipe_eligible':all(quality.values()) and all(economics.values())}
