/* Scheduling-only assay: source BF generators and unchanged I16 readout. */
#include <windows.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <fenv.h>
#include "contextual_generator_cost.c"
#define bf integer_bf
#include "integer_readout.c"
#undef bf
static void require(int ok,const char*why){if(!ok){fprintf(stderr,"%s\n",why);abort();}}
static void*alloc(size_t n,size_t bytes){void*p=calloc(n,bytes);require(p!=NULL,"allocation");return p;}
#include "meth388_switch_thread_binding.h"

API int po_setup(void){
    worker_setup(6);return 0;
}
API int po_threads(int n){
    if(n!=1&&n!=3&&n!=6)return 1;
    worker_requested=n;omp_set_dynamic(0);omp_set_num_threads(n);return 0;
}
API int po_team_check(void){
    #pragma omp parallel num_threads(worker_requested)
    {require(omp_get_num_threads()==worker_requested,"metadata_team_size");worker_bind();}
    return 0;
}
API uint64_t po_binding(int slot,int field){
    if(slot<0||slot>=6)return UINT64_MAX;
    WorkerBinding b=worker_latest[slot];
    HANDLE h=OpenThread(THREAD_QUERY_LIMITED_INFORMATION,FALSE,b.tid);
    GROUP_AFFINITY actual;memset(&actual,0,sizeof(actual));
    require(h&&GetThreadGroupAffinity(h,&actual),"worker_readback");CloseHandle(h);
    require(actual.Group==b.group&&actual.Mask==b.mask,"worker_final_binding");
    if(field==0)return b.tid;if(field==1)return b.mask;if(field==2)return b.group;
    if(field==3)return worker_physical_cores;if(field==4)return worker_event_count;
    return UINT64_MAX;
}
API int po_generator(const uint16_t*w,const uint16_t*h,const uint16_t*mup,float*raw,uint16_t*gg,uint16_t*fields){
    if(!w||!h||!mup||!raw||!gg||!fields)return 1;
    if(fegetround()!=FE_TONEAREST||cg_control()!=0x1f80u)return 3;
    if(worker_requested==1)return cg_step(w,h,mup,raw,gg,fields);
    float input[2048];for(int i=0;i<2048;i++){if((h[i]&0x7f80u)==0x7f80u)return 4;input[i]=decode(h[i]);}
    #pragma omp parallel num_threads(worker_requested)
    {
        require(omp_get_num_threads()==worker_requested,"generator_team_size");
        worker_bind();require(fegetround()==FE_TONEAREST&&cg_control()==0x1f80u,"generator_worker_FP");
        #pragma omp for schedule(static)
        for(int o=0;o<6704;o++){
            const uint16_t*row=w+(size_t)o*2048;__m256 sum=_mm256_setzero_ps();
            for(int j=0;j<2048;j+=8){
                __m256i bits=_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(row+j))),16);
                sum=_mm256_fmadd_ps(_mm256_castsi256_ps(bits),_mm256_loadu_ps(input+j),sum);
            }
            float lane[8];_mm256_storeu_ps(lane,sum);
            float value=lane[0]+lane[1]+lane[2]+lane[3]+lane[4]+lane[5]+lane[6]+lane[7];
            raw[o]=value;gg[o]=round_bf(value);fields[o]=round_bf(decode(gg[o])*decode(mup[o]));
        }
    }return 0;
}
API int po_readout(int n,const int16_t*w,const double*scale,const int16_t*x,const double*a,int64_t*dots,double*y,uint16_t*bits){
    if(n<1||n>1507)return 1;
    if(fegetround()!=FE_TONEAREST||cg_control()!=0x1f80u)return 3;
    if(worker_requested==1)return ir_history(n,w,scale,x,a,dots,y,bits);
    #pragma omp parallel num_threads(worker_requested)
    {
        require(omp_get_num_threads()==worker_requested,"readout_team_size");
        worker_bind();require(fegetround()==FE_TONEAREST&&cg_control()==0x1f80u,"readout_worker_FP");
        #pragma omp for schedule(static)
        for(int o=0;o<2048;o++)for(int t=0;t<n;t++){
            int64_t z=dot16(w+(size_t)o*3072,x+(size_t)t*3072);dots[(size_t)t*2048+o]=z;
            double f=(double)z*a[t];f=f*scale[o];y[(size_t)t*2048+o]=f;bits[(size_t)t*2048+o]=integer_bf((float)f);
        }
    }return 0;
}
