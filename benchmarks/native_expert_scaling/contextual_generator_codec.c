/* Source-only weight codecs. BF h/GG/MuP boundaries remain explicit. */
#include "physical_worker_organs.c"
API int qc_probe(int format,int n,const uint8_t*codes,int8_t*out){
    if(n<1||n>4096||!codes||!out||(format!=8&&format!=4))return 1;
    for(int i=0;i<n;i++){
        int v=format==8?(int)(int8_t)codes[i]:(int)((codes[i/2]>>(4*(i%2)))&15)-7;
        if((format==8&&v==-128)||(format==4&&v==8))return 4;
        out[i]=(int8_t)v;
    }return 0;
}
API void qc_set_control_for_test(unsigned v){_mm_setcsr((_mm_getcsr()&0x3fu)|(v&0xffc0u));}
API int qc_step(int format,const uint8_t*w,const float*scale,const uint16_t*h,const uint16_t*mup,float*raw,uint16_t*gg,uint16_t*fields){
    if((format!=8&&format!=4)||!w||!scale||!h||!mup||!raw||!gg||!fields)return 1;
    if(fegetround()!=FE_TONEAREST||cg_control()!=0x1f80u)return 3;
    float input[2048];
    for(int i=0;i<2048;i++){if((h[i]&0x7f80u)==0x7f80u)return 4;input[i]=decode(h[i]);}
    #pragma omp parallel num_threads(worker_requested)
    {
        require(omp_get_num_threads()==worker_requested,"codec_team_size");worker_bind();
        require(fegetround()==FE_TONEAREST&&cg_control()==0x1f80u,"codec_worker_FP");
        #pragma omp for schedule(static)
        for(int o=0;o<6704;o++){
            __m256 total=_mm256_setzero_ps();
            if(format==8){
                const int8_t*row=(const int8_t*)w+(size_t)o*2048;
                for(int j=0;j<2048;j+=8){
                    __m256 q=_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(row+j))));
                    total=_mm256_fmadd_ps(q,_mm256_loadu_ps(input+j),total);
                }
            }else{
                const uint8_t*row=w+(size_t)o*1024;
                for(int group=0;group<16;group++){
                    __m256 part=_mm256_setzero_ps();
                    for(int j=group*128;j<(group+1)*128;j+=16){
                        __m128i packed=_mm_loadl_epi64((const __m128i*)(row+j/2));
                        __m128i mask=_mm_set1_epi8(15);
                        __m128i low=_mm_and_si128(packed,mask),high=_mm_and_si128(_mm_srli_epi16(packed,4),mask);
                        __m128i paired=_mm_unpacklo_epi8(low,high);
                        __m256i q0=_mm256_sub_epi32(_mm256_cvtepu8_epi32(paired),_mm256_set1_epi32(7));
                        __m256i q1=_mm256_sub_epi32(_mm256_cvtepu8_epi32(_mm_srli_si128(paired,8)),_mm256_set1_epi32(7));
                        part=_mm256_fmadd_ps(_mm256_cvtepi32_ps(q0),_mm256_loadu_ps(input+j),part);
                        part=_mm256_fmadd_ps(_mm256_cvtepi32_ps(q1),_mm256_loadu_ps(input+j+8),part);
                    }
                    total=_mm256_fmadd_ps(part,_mm256_set1_ps(scale[(size_t)o*16+group]),total);
                }
            }
            float lane[8];_mm256_storeu_ps(lane,total);
            float v=lane[0]+lane[1]+lane[2]+lane[3]+lane[4]+lane[5]+lane[6]+lane[7];
            if(format==8)v=v*scale[o];
            raw[o]=v;gg[o]=round_bf(v);fields[o]=round_bf(decode(gg[o])*decode(mup[o]));
        }
    }return 0;
}
