"""Independent decoder/extrema/F32-selection/all-report verification; no main/math import."""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
import argparse
import datetime
from fractions import Fraction
import json
from pathlib import Path
import struct
import traceback
from meth492_operations import Context, DOC, ROOT, write

OUT = ROOT / 'results/native_expert_scaling/meth492_selected_amplitude_audit'
RET = DOC / 'RETENTION_492_20261006.json'
RAW = DOC / 'meth492_selected_amplitude_result.json'

def descriptor(ctx,path):
    return {'path':str(path),'bytes':path.stat().st_size,'sha256':ctx.digest(path)}

def fraction(bits):
    assert 0 < int(bits) <= 0x3f800000
    return Fraction.from_float(struct.unpack('<f',struct.pack('<I',int(bits)))[0])

def rational(value):
    return {'numerator':str(value.numerator),'denominator':str(value.denominator)}

def select(minimum,maximum):
    if minimum is None:
        return {'decision':'NO_DEVELOPMENT_CELL','ideal_constant_exists':None,'candidate_bits':None}
    small,big=fraction(minimum),fraction(maximum)
    low,high=big*100/101,small*101/100
    feasible=big*10000<=small*10201
    assert feasible==(low<=high)
    chosen=None
    if feasible:
        # Nearest F32 is only an initial guess. Exact inequalities and predecessor
        # determine the result, independently of the main's bit-space bisection.
        guess=struct.unpack('<I',struct.pack('<f',float(low)))[0]
        guess=max(1,guess)
        while guess<=0x3f800000 and fraction(guess)<low:guess+=1
        while guess>1 and fraction(guess-1)>=low:guess-=1
        if guess<=0x3f800000 and fraction(guess)<=high:chosen=guess
        if chosen:
            assert low<=fraction(chosen)<=high and (chosen==1 or fraction(chosen-1)<low)
    return {'decision':'IDEAL_CONSTANT_EXCLUDED' if not feasible else
            'IDEAL_ELIGIBLE_NO_F32' if chosen is None else 'F32_CONSTANT_ELIGIBLE',
            'ideal_constant_exists':feasible,'candidate_bits':chosen,
            'allowed_constant_interval':[rational(low),rational(high)],'extrema_ratio':rational(big/small)}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',required=True,type=Path)
    parser.add_argument('--binding-sha',required=True);args=parser.parse_args()
    assert args.out.resolve()==RET.resolve()
    ctx=Context(OUT,RET,300,256<<20)
    try:
        binding=ctx.admit(args.binding_sha);ctx.binding=binding
        failure=RAW.with_suffix('.failure.json');path=failure if failure.exists() else RAW
        raw=json.loads(path.read_bytes());raw_desc=descriptor(ctx,path)
        registrationpath=DOC/'meth492_main_sole_first_registration.json'
        registration=json.loads(registrationpath.read_bytes())
        assert registration['attempt']==1 and registration['record_sha256']==raw_desc['sha256']
        assert registration['process_instance']==raw['process_instance']
        completed=path==RAW
        assert (registration['actual_exit_code']==0)==completed
        eventpath=ROOT/'results/native_expert_scaling/meth492_windows_terminal.json'
        event=json.loads(eventpath.read_bytes())
        assert event['query_available'] and event['query_error'] is None and not event['matching_scientific_events']
        instance=event['instances'][0]
        assert instance['pid']==raw['process_instance']['pid'] and instance['create_time_unix']==raw['process_instance']['create_time_unix']
        assert all(datetime.datetime.fromisoformat(instance[k])==datetime.datetime.fromisoformat(raw[k]) for k in ('start_utc','end_utc'))
        ctx.r['gates']['sole_actual_main_registration_and_available_fault_free_Windows_query']=True
        inventory=raw.get('output_inventory',[])
        for item in inventory:
            assert Path(item['path']).parent==ROOT/'results/native_expert_scaling/meth492_selected_amplitude'
            ctx.exact(item)
        if not completed:
            result=ctx.finish({'raw':raw_desc,'main_windows_sha256':ctx.digest(eventpath),'main_completed':False,
                'native_calls':0,'optimizer_updates':0,'SVD_calls':0,
                'decision':'FIRST_MAIN_FAULT_AND_PARTIALS_METADATA_ADMITTED_NO_NUMERICAL_PROMOTION'})
            print(json.dumps({'gates':result['gates'],'decision':result['decision'],'resource':result['resource']}));return
        assert all(raw['gates'].values()) and raw['completed_cells']==1536
        assert (raw['unique_inputs'],raw['occurrences'],raw['development_inputs'],raw['consumed_validation_inputs'])==(238872,387036,159414,79458)
        assert raw['state_labels']==['PASS','SOURCE_P_BELOW_CONSTANT_RANGE','SOURCE_P_ABOVE_CONSTANT_RANGE','NO_SCALAR','INVALID_SOURCE_P']
        assert raw['rho_exact']==[101,100] and raw['native_calls']==raw['optimizer_updates']==raw['SVD_calls']==0
        terminalpath=ROOT/'results/native_expert_scaling/meth492_selected_amplitude/terminal_resources.json'
        assert raw['terminal_resource_path']==str(terminalpath)
        terminal=json.loads(terminalpath.read_bytes())
        assert terminal['raw_sha256']==raw_desc['sha256'] and terminal['raw_path']==str(path)
        assert terminal['resource']['wall_seconds']<=180
        assert terminal['resource']['parent_peak_bytes']+terminal['resource']['native_peak_bytes']<=256<<20
        import numpy as np
        import threadpoolctl
        ctx.r['numerical_imports']=True;threadpoolctl.threadpool_limits(limits=1)
        assert all(v['num_threads']==1 for v in threadpoolctl.threadpool_info())
        ctx.phase='independent_new_exact_fraction_controls'
        controls=json.loads((ROOT/'results/native_expert_scaling/meth492_selected_amplitude/math_controls.json').read_bytes())
        def bits(value):return struct.unpack('<I',struct.pack('<f',value))[0]
        boundary=(bits(10000/16384),bits(10201/16384))
        fixtures=[('one_point',(0x3f000000,0x3f000000),'F32_CONSTANT_ELIGIBLE'),
            ('exact_rho_squared_boundary',boundary,'F32_CONSTANT_ELIGIBLE'),
            ('one_F32_beyond_boundary',(boundary[0],boundary[1]+1),'IDEAL_CONSTANT_EXCLUDED'),
            ('ideal_interval_without_F32',tuple(0x3f000000+n-(1<<23) for n in (10000050,10201051)),'IDEAL_ELIGIBLE_NO_F32'),
            ('positive_subnormal',(1,1),'F32_CONSTANT_ELIGIBLE'),
            ('probability_one',(0x3f800000,0x3f800000),'F32_CONSTANT_ELIGIBLE'),
            ('empty_cell',(None,None),'NO_DEVELOPMENT_CELL'),
            ('validation_only_cell',(None,None),'NO_DEVELOPMENT_CELL')]
        assert len(controls)==len(fixtures)
        for control,(name,pair,decision) in zip(controls,fixtures):
            assert (control['control'],tuple(control['input_extrema_bits']),control['expected'])==(name,pair,decision)
        for control in controls:
            reconstructed=select(*control['input_extrema_bits'])
            assert reconstructed['decision']==control['expected']
            assert all(control[k]==v for k,v in reconstructed.items())
        assert len(controls)==8 and fraction(1)==Fraction(1,1<<149) and fraction(0x3f800000)==1
        ctx.r['gates']['independent_exact_Fraction_controls_and_F32_predecessor_selection']=True
        U,Q=238872,387036
        ctx.r['source_scan_bytes']=0
        def scan(name,magic,width,reserved,count,dtype):
            with Path(binding['data'][name]['path']).open('rb') as stream:
                assert stream.read(24)==struct.pack('<8sIIQ',magic,width,reserved,count)
                ctx.r['source_scan_bytes']+=24
                for begin in range(0,count,2048):
                    data=stream.read(min(2048,count-begin)*width)
                    values=np.frombuffer(data,dtype);assert len(values)==min(2048,count-begin)
                    ctx.r['source_scan_bytes']+=len(data);ctx.guard();yield begin,values
                assert not stream.read(1)
        unique_dtype=np.dtype([('meta','<u4',(12,)),('hashes','u1',(64,)),('value','<f8',(3,)),('other','u1',(3584,))])
        metadata=np.empty((U,12),'<u4');pbits=np.empty(U,'<u4')
        for begin,part in scan('unique_inputs.bin',b'M479UNI1',3720,768,U,unique_dtype):
            metadata[begin:begin+len(part)]=part['meta']
            converted=part['value'][:,1].astype('<f4')
            assert converted.astype('<f8').tobytes()==part['value'][:,1].tobytes()
            pbits[begin:begin+len(part)]=converted.view('<u4')
        del part,converted
        owner=np.empty((U,8),'<u4');links=np.empty((Q,13),'<u4')
        for begin,part in scan('ownership.bin',b'M479OWN1',32,0,U,np.dtype(('<u4',(8,)))):owner[begin:begin+len(part)]=part
        for begin,part in scan('query_links.bin',b'M479LNK1',52,0,Q,np.dtype(('<u4',(13,)))):links[begin:begin+len(part)]=part
        assert np.array_equal(metadata[:,0],np.arange(U)) and np.array_equal(owner[:,0],metadata[:,0])
        assert np.array_equal(metadata[:,2],owner[:,1]) and np.all(metadata[:,2]<12) and np.all(metadata[:,7]<128)
        assert np.array_equal(links[:,0],np.arange(Q)) and np.all(links[:,12]<U)
        assert np.all(links[:,2]<192) and np.all(links[:,4]<2) and np.isin(links[:,10],[0,1]).all()
        expectedrole=np.where(links[:,2]<64,0,np.where(links[:,2]<128,1,2))
        assert np.array_equal(expectedrole,links[:,5])
        assert np.array_equal(links[:,6],metadata[links[:,12],2]) and np.array_equal(links[:,9],metadata[links[:,12],7])
        source_dtype=np.dtype([('meta','<u4',(12,)),('value','<f8',(3,))])
        for begin,part in scan('source_fields.bin',b'M479SRC1',72,0,Q,source_dtype):
            current=links[begin:begin+len(part)]
            assert current[:,:12].tobytes()==part['meta'].tobytes()
            values=part['value'][:,1].astype('<f4')
            assert values.astype('<f8').tobytes()==part['value'][:,1].tobytes()
            assert values.view('<u4').tobytes()==pbits[current[:,12]].tobytes()
        del part,values,current,expectedrole
        for source_column,target_column in ((5,2),(4,3),(10,4)):
            flags=np.zeros(U,'<u4')
            np.bitwise_or.at(flags,links[:,12],np.left_shift(np.uint32(1),links[:,source_column]))
            assert np.array_equal(flags,owner[:,target_column])
        assert np.array_equal(np.bincount(links[:,12],minlength=U),owner[:,5]);del flags
        dev=(owner[:,2]&5)!=0;val=(owner[:,2]&2)!=0
        assert (int(dev.sum()),int(val.sum()))==(159414,79458) and not np.any(dev&val) and np.all(dev|val)
        ctx.r['gates']['ALL_source_UID_bank_ID_occurrence_role_mode_accepted_and_p_BYTE_joins']=True
        accum=[{'development_count':0,'consumed_validation_count':0,'invalid_development_count':0,
                'minimum_bits':None,'maximum_bits':None,'minimum_UID':None,'maximum_UID':None,
                'first_invalid_UID':None} for _ in range(1536)]
        for uid in range(U):
            bits=int(pbits[uid]);key=int(metadata[uid,2])*128+int(metadata[uid,7]);item=accum[key]
            if val[uid]:item['consumed_validation_count']+=1
            if dev[uid]:
                item['development_count']+=1
                if not 0<bits<=0x3f800000:
                    item['invalid_development_count']+=1
                    if item['first_invalid_UID'] is None:item['first_invalid_UID']=uid
                else:
                    if item['minimum_bits'] is None or bits<item['minimum_bits']:item['minimum_bits']=bits;item['minimum_UID']=uid
                    if item['maximum_bits'] is None or bits>item['maximum_bits']:item['maximum_bits']=bits;item['maximum_UID']=uid
            if uid%4096==0:ctx.guard()
        constants={};cells=[]
        ctx.phase='independent_ALL1536_extrema_intervals_and_fixed_scalar_selection'
        for key,item in enumerate(accum):
            bank,expert=divmod(key,128)
            row={k:item[k] for k in ('development_count','consumed_validation_count','invalid_development_count')}
            row.update(bank=bank,source_ID=expert)
            if item['invalid_development_count']:
                row.update(decision='INVALID_DEVELOPMENT_PROBABILITY',ideal_constant_exists=None,
                           candidate_bits=None,first_invalid_UID=item['first_invalid_UID'])
            elif not item['development_count']:row.update(select(None,None))
            else:
                row.update({k:item[k] for k in ('minimum_bits','maximum_bits','minimum_UID','maximum_UID')})
                row.update(select(item['minimum_bits'],item['maximum_bits']))
            assert row==raw['cells'][key],key
            if row['candidate_bits'] is not None:constants[key]=fraction(row['candidate_bits'])
            cells.append(row)
        ctx.r['gates']['ALL1536_earliest_UID_extrema_exact_intervals_and_F32_candidates']=True
        wirepath=ROOT/'results/native_expert_scaling/meth492_selected_amplitude/uid_amplitude.bin'
        with wirepath.open('rb') as stream:
            assert stream.read(24)==struct.pack('<8sIIQ',b'M492AMP1',28,0,U)
            wire=np.frombuffer(stream.read(),'<u4').reshape(U,7)
        assert np.array_equal(wire[:,0],metadata[:,0]) and np.array_equal(wire[:,1],metadata[:,2])
        assert np.array_equal(wire[:,2],metadata[:,7]) and np.array_equal(wire[:,3],owner[:,2])
        assert np.array_equal(wire[:,4],pbits)
        state=np.empty(U,'<u4')
        for uid in range(U):
            bits=int(pbits[uid]);key=int(metadata[uid,2])*128+int(metadata[uid,7])
            c=constants.get(key);expected=4 if not 0<bits<=0x3f800000 else 3
            if expected!=4 and c is not None:
                q=fraction(bits)
                expected=1 if q*101<c*100 else 2 if q*100>c*101 else 0
            assert int(wire[uid,5])==(cells[key]['candidate_bits'] or 0) and int(wire[uid,6])==expected
            state[uid]=expected
            if uid%4096==0:ctx.guard()
        ctx.r['gates']['ALL238872_saved_UID_source_bits_scalar_bits_and_exact_amplitude_states']=True
        groups={name:{} for name in ('unique','occurrence','book','source_ID','accepted')}
        def add(kind,key,uid):
            counts,first=groups[kind].setdefault(key,([0]*5,[None]*5));s=int(state[uid]);counts[s]+=1
            if first[s] is None or uid<first[s]:first[s]=uid
        ctx.phase='independent_ALL_report_counts_and_first_UIDs'
        for uid in range(U):
            bank=int(metadata[uid,2]);role='development' if dev[uid] else 'consumed_validation'
            for b in (-1,bank):
                add('unique',(b,'ALL'),uid);add('unique',(b,role),uid)
        for position,link in enumerate(links):
            bank,role,mode,e,accepted,uid,book=map(int,(link[6],link[5],link[4],link[9],link[10],link[12],link[2]))
            add('occurrence',(bank,role,mode),uid);add('book',(bank,book,mode),uid)
            add('source_ID',(bank,role,mode,e),uid);add('accepted',(bank,role,mode,accepted),uid)
            if position%4096==0:ctx.guard()
        checked={}
        for kind,rows in raw['reports'].items():
            names={'unique':('bank','role'),'occurrence':('bank','role','mode'),
                'book':('bank','book','mode'),'source_ID':('bank','role','mode','source_ID'),
                'accepted':('bank','role','mode','accepted')}[kind]
            seen=set()
            for row in rows:
                key=tuple(row[k] for k in names);assert key not in seen;seen.add(key)
                counts,first=groups[kind].get(key,([0]*5,[None]*5))
                assert row['count']==sum(counts) and row['state_counts']==counts and row['first_UID_by_state']==first
            assert set(groups[kind])<=seen
            checked[kind]=len(seen)
        assert checked=={'unique':39,'occurrence':72,'book':4608,'source_ID':9216,'accepted':144}
        ctx.r['gates']['ALL39_72_4608_9216_144_reports_independent_counts_and_first_UIDs']=True
        outcomes={name:sum(row['decision']==name for row in cells) for name in sorted({row['decision'] for row in cells})}
        assert outcomes==raw['outcome_counts']
        whole=bool(np.all(state==0));excluded=outcomes.get('IDEAL_CONSTANT_EXCLUDED',0)
        decision='CONSTANT_AMPLITUDE_CLASS_EXCLUDED_ON_DEVELOPMENT' if excluded else 'FIXED_F32_SOURCE_DOMAIN_ELIGIBLE' if whole else 'INCOMPLETE_CONSTANT_AMPLITUDE_ELIGIBILITY'
        assert raw['decision']==decision and raw['candidate_whole_source_amplitude_pass']==whole
        ctx.r['gates']['exact_class_decision_with_no_function_task_speed_or_scaling_promotion']=True
        result=ctx.finish({'raw':raw_desc,'main_windows_sha256':ctx.digest(eventpath),'main_completed':True,
            'main_terminal_resources':descriptor(ctx,terminalpath),
            'cells_audited':1536,'unique_inputs_audited':U,'occurrences_audited':Q,'report_counts_audited':checked,
            'outcome_counts':outcomes,'source_amplitude_decision':decision,'native_calls':0,'optimizer_updates':0,'SVD_calls':0,
            'decision':'ALL_EXACT_SELECTED_AMPLITUDE_OUTCOMES_INDEPENDENTLY_ADMITTED',
            'scope':'Only the fixed selected-amplitude constant class on source states; full goal active/incomplete.'})
        print(json.dumps({'gates':result['gates'],'outcome_counts':outcomes,'resource':ctx.terminal_resources['resource']}))
    except BaseException as exc:
        ctx.fail(exc);print(json.dumps({'error':str(exc),'traceback':traceback.format_exc()}),flush=True);raise

if __name__=='__main__':main()
