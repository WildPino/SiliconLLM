"""Read-only PE export contract. Does not load a DLL, CUDA, model or operator."""
import mmap
from pathlib import Path
import re
import struct

def exported_names(path):
    with Path(path).open('rb') as f,mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ) as image:
        assert image[:2]==b'MZ';pe=struct.unpack_from('<I',image,60)[0];assert image[pe:pe+4]==b'PE\x00\x00'
        machine,sections=struct.unpack_from('<HH',image,pe+4);assert machine==0x8664 and 0<sections<128
        optional=pe+24;size=struct.unpack_from('<H',image,pe+20)[0];assert struct.unpack_from('<H',image,optional)[0]==0x20b and size>=120
        er,es=struct.unpack_from('<II',image,optional+112);assert er>0 and es>0;table=[]
        for j in range(sections):
            at=optional+size+40*j;vs,va,rs,ro=struct.unpack_from('<4I',image,at+8);table.append((va,max(vs,rs),ro))
        def offset(rva):
            for va,length,start in table:
                if va<=rva<va+length:
                    result=start+rva-va;assert result<len(image);return result
            raise AssertionError(('PE_RVA',str(path),rva))
        directory=struct.unpack_from('<IIHH7I',image,offset(er));nf,nn,funcs,names,ordinals=directory[6:11];assert 0<nn<=nf<20000
        result={};na,oa,fa=offset(names),offset(ordinals),offset(funcs)
        for j in range(nn):
            rva=struct.unpack_from('<I',image,na+4*j)[0];at=offset(rva);end=image.find(b'\x00',at,at+512);assert end>at
            name=image[at:end].decode('ascii');ordinal=struct.unpack_from('<H',image,oa+2*j)[0];assert ordinal<nf
            function=struct.unpack_from('<I',image,fa+4*ordinal)[0];assert function>0 and name not in result
            forward=None
            if er<=function<er+es:
                fp=offset(function);fe=image.find(b'\x00',fp,fp+512);assert fe>fp;forward=image[fp:fe].decode('ascii')
            result[name]={'ordinal_index':ordinal,'function_RVA':function,'forwarder':forward}
        return result

def contract(backend,cuda_dir):
    text=Path(backend).read_text();calls=re.findall(r'G486_BIND\(\w+,\w+,(rt|bl|dr),"([^"]+)"\)',text)
    assert len(calls)==25 and len(set(calls))==len(calls),len(calls)
    mapping={'rt':Path(cuda_dir)/'cudart64_12.dll','bl':Path(cuda_dir)/'cublas64_12.dll','dr':Path('C:/Windows/System32/nvcuda.dll')}
    tables={key:exported_names(path) for key,path in mapping.items()};rows=[]
    for key,name in calls:
        assert name in tables[key],('missing_requested_export',str(mapping[key]),name)
        rows.append({'module_path':str(mapping[key]),'symbol':name,**tables[key][name]})
    assert any(r['symbol']=='cublasSetWorkspace_v2' for r in rows)
    assert 'cublasSetWorkspace' not in tables['bl']
    return {'no_Dll_loading_or_CUDA_calls':True,'requested_symbol_count':len(calls),'ALL_requested_exports_present':True,'rows':rows,
            'export_name_counts':{key:len(v) for key,v in tables.items()}}
