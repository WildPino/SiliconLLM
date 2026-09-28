// METH-130: paired exact-BF16/Q7-LUT factor screen on 256 actual E1280 states.
#define main meth124_reference_main
#include "meth124_centered_factor_cpu.c"
#undef main

typedef struct { char magic[8]; uint32_t layers,tokens,width,experts; } VHeader;
typedef struct { const uint8_t *records; } QLayer;

static void selected_route(const Layer *layer,const float *x,const Header *h,
                           int ids[4],float gates[4]) {
    int parents[4];
    route_parent(layer,x,h,parents,gates);
    route_child(layer,x,h,parents,ids);
}

static float qscale(const uint8_t *record) {
    float value;
    memcpy(&value,record,4);
    return value;
}

static void q7_residual(const Layer *layer,const QLayer *quant,const float *x,
                        const Header *h,const int ids[4],const float gates[4],float *out) {
    float hidden[4][8];
    for(int k=0;k<4;k++) for(uint32_t r=0;r<h->r;r++) {
        const uint16_t *row=layer->fa+(((size_t)ids[k]/h->children)*h->r+r)*h->d;
        float sum=0;
        for(uint32_t d=0;d<h->d;d++) sum+=bf_float(row[d])*x[d];
        sum=round_bf(sum);
        hidden[k][r]=round_bf(sum/(1.0f+expf(-sum)));
    }
    float lut[4][4][256];
    for(int k=0;k<4;k++) for(int pair=0;pair<4;pair++)
        for(int code=0;code<256;code++) {
            int low=(code&15)-3, high=(code>>4)-3;
            lut[k][pair][code]=(float)low*hidden[k][2*pair]
                              +(float)high*hidden[k][2*pair+1];
        }
    for(uint32_t d=0;d<h->d;d++) {
        float sum=0;
        for(int k=0;k<4;k++) {
            const uint8_t *record=quant->records+((size_t)ids[k]*h->d+d)*8;
            float part=qscale(record)*(lut[k][0][record[4]]+lut[k][1][record[5]]+
                                       lut[k][2][record[6]]+lut[k][3][record[7]]);
            part=round_bf(part);
            sum+=round_bf(part*gates[k]);
        }
        out[d]=round_bf(sum);
    }
}

static double median5(const double values[5]) {
    double sorted[5]; memcpy(sorted,values,sizeof sorted);
    qsort(sorted,5,sizeof(double),cmp_double);
    return sorted[2];
}

int main(int argc,char **argv) {
    if(argc!=4) { fprintf(stderr,"usage: %s BF16_BANK Q7_BANK VECTORS\n",argv[0]); return 2; }
    double started=now_s();
    size_t original_size,quant_size,vector_size;
    const uint8_t *original=read_all(argv[1],&original_size);
    const uint8_t *qbank=read_all(argv[2],&quant_size);
    const uint8_t *vector_data=read_all(argv[3],&vector_size);
    Header h,qh; VHeader v;
    if(original_size<sizeof h || quant_size<sizeof qh || vector_size<sizeof v) return 2;
    memcpy(&h,original,sizeof h); memcpy(&qh,qbank,sizeof qh);
    memcpy(&v,vector_data,sizeof v);
    if(memcmp(h.magic,"M126FB01",8) || memcmp(qh.magic,"M130FB01",8) ||
       memcmp((const uint8_t*)&h+8,(const uint8_t*)&qh+8,sizeof h-8) ||
       h.l!=24 || h.d!=896 || h.rank!=64 || h.na!=8 || h.nb!=16 ||
       h.child_rank!=32 || h.children!=10 || h.r!=8 ||
       memcmp(v.magic,"M125HX01",8) || v.layers!=24 || v.tokens!=256 ||
       v.width!=896 || v.experts!=1280) {
        fprintf(stderr,"bad header\n"); return 2;
    }
    size_t router=((size_t)h.rank*h.d+(h.na+h.nb)*h.rank+
                   (size_t)h.child_rank*h.d+(size_t)v.experts*h.child_rank)*4;
    size_t a_bytes=(size_t)h.na*h.nb*h.r*h.d*2;
    size_t rows=(size_t)v.experts*h.d;
    size_t original_layer=router+a_bytes+rows*h.r*2;
    size_t quant_layer=router+a_bytes+rows*8;
    if(original_size!=sizeof h+(size_t)h.l*original_layer ||
       quant_size!=sizeof qh+(size_t)h.l*quant_layer ||
       vector_size!=sizeof v+(size_t)v.tokens*v.layers*v.width*2) {
        fprintf(stderr,"bad length\n"); return 2;
    }
    Layer layers[24],quant_router[24]; QLayer qlayers[24];
    size_t parent_bytes=((size_t)h.rank*h.d+(h.na+h.nb)*h.rank)*4;
    size_t projection_bytes=(size_t)h.child_rank*h.d*4;
    for(int li=0;li<24;li++) {
        const uint8_t *old=original+sizeof h+(size_t)li*original_layer;
        const uint8_t *q=qbank+sizeof qh+(size_t)li*quant_layer;
        if(memcmp(old,q,router+a_bytes)) { fprintf(stderr,"router/A mismatch at layer %d\n",li); return 1; }
        Layer *l=&layers[li];
        l->p=(const float*)old; l->a=l->p+(size_t)h.rank*h.d;
        l->b=l->a+(size_t)h.na*h.rank;
        l->child_projection=(const float*)(old+parent_bytes);
        l->child_keys=(const float*)(old+parent_bytes+projection_bytes);
        l->fa=(const uint16_t*)(old+router);
        l->fb=(const uint16_t*)(old+router+a_bytes);
        quant_router[li]=*l;
        quant_router[li].p=(const float*)q;
        quant_router[li].a=quant_router[li].p+(size_t)h.rank*h.d;
        quant_router[li].b=quant_router[li].a+(size_t)h.na*h.rank;
        quant_router[li].child_projection=(const float*)(q+parent_bytes);
        quant_router[li].child_keys=(const float*)(q+parent_bytes+projection_bytes);
        qlayers[li].records=q+router+a_bytes;
        for(size_t row=0;row<rows;row++) {
            const uint8_t *record=qlayers[li].records+row*8;
            float scale=qscale(record);
            if(!isfinite(scale) || scale<0) { fprintf(stderr,"bad Q7 scale\n"); return 1; }
            for(int pair=0;pair<4;pair++)
                if((record[4+pair]&15)>6 || (record[4+pair]>>4)>6) {
                    fprintf(stderr,"bad Q7 nibble\n"); return 1;
                }
        }
    }
    size_t states=(size_t)v.tokens*v.layers;
    float *x=malloc(states*v.width*4);
    int (*selected)[24][4]=malloc((size_t)v.tokens*sizeof *selected);
    float (*gates)[24][4]=malloc((size_t)v.tokens*sizeof *gates);
    double *errors=malloc(states*sizeof(double));
    if(!x || !selected || !gates || !errors) return 2;
    const uint16_t *xbf=(const uint16_t*)(vector_data+sizeof v);
    for(size_t i=0;i<states*v.width;i++) x[i]=bf_float(xbf[i]);
    double numerator=0,denominator=0,layer_num[24]={0},layer_den[24]={0};
    double maximum=0; int zero_norms=0,route_mismatches=0,nonfinite=0;
    unsigned char seen[24][1280]={{0}};
    volatile double checksum=0;
    for(int token=0;token<256;token++) for(int li=0;li<24;li++) {
        size_t state=(size_t)token*24+li;
        const float *input=x+state*896;
        int compare_ids[4]; float compare_gates[4];
        selected_route(&layers[li],input,&h,selected[token][li],gates[token][li]);
        selected_route(&quant_router[li],input,&h,compare_ids,compare_gates);
        for(int k=0;k<4;k++) {
            if(compare_ids[k]!=selected[token][li][k] || compare_gates[k]!=gates[token][li][k])
                route_mismatches++;
            seen[li][selected[token][li][k]]=1;
        }
        float reference[896],qout[896];
        residual(&layers[li],input,&h,selected[token][li],gates[token][li],reference);
        q7_residual(&layers[li],&qlayers[li],input,&h,selected[token][li],gates[token][li],qout);
        double num=0,den=0;
        for(int d=0;d<896;d++) {
            if(!isfinite(reference[d]) || !isfinite(qout[d])) nonfinite++;
            double diff=(double)reference[d]-qout[d];
            num+=diff*diff; den+=(double)reference[d]*reference[d];
        }
        if(den==0) { zero_norms++; errors[state]=num==0?0:INFINITY; }
        else errors[state]=sqrt(num/den);
        if(errors[state]>maximum) maximum=errors[state];
        numerator+=num; denominator+=den;
        layer_num[li]+=num; layer_den[li]+=den;
        checksum+=qout[(token+li)%896];
    }
    qsort(errors,states,sizeof(double),cmp_double);
    int unique_min=1280,unique_max=0;
    for(int li=0;li<24;li++) {
        int unique=0; for(int e=0;e<1280;e++) unique+=seen[li][e];
        if(unique<unique_min) unique_min=unique;
        if(unique>unique_max) unique_max=unique;
        printf("Q7_LAYER layer=%d unique_children=%d relative_l2=%.9f\n",li,unique,
               sqrt(layer_num[li]/layer_den[li]));
    }
    double exact_ms[5],quant_ms[5];
    for(int rep=0;rep<5;rep++) {
        for(int arm=0;arm<2;arm++) {
            int q_first=rep&1;
            int use_q=(arm==0)?q_first:!q_first;
            double begun=now_s();
            for(int token=0;token<256;token++) for(int li=0;li<24;li++) {
                const float *input=x+((size_t)token*24+li)*896;
                float out[896];
                if(use_q) q7_residual(&layers[li],&qlayers[li],input,&h,
                                     selected[token][li],gates[token][li],out);
                else residual(&layers[li],input,&h,
                              selected[token][li],gates[token][li],out);
                checksum+=out[(token+li)%896];
            }
            double ms=(now_s()-begun)*1000.0/256.0;
            if(use_q) quant_ms[rep]=ms; else exact_ms[rep]=ms;
        }
        printf("Q7_REP rep=%d exact_factor_ms=%.9f q7_factor_ms=%.9f\n",
               rep,exact_ms[rep],quant_ms[rep]);
    }
    double relative=sqrt(numerator/denominator);
    double p95=errors[(size_t)ceil(0.95*states)-1];
    double exact_med=median5(exact_ms),quant_med=median5(quant_ms);
    int pass=route_mismatches==0 && nonfinite==0 && relative<=0.10 && p95<=0.20 &&
             quant_size<=300000000 && quant_med<=1.20*exact_med &&
             now_s()-started<=600 && rss_bytes()<4ULL*(1ULL<<30);
    printf("Q7_SUMMARY states=%zu route_mismatches=%d nonfinite=%d zero_reference_norms=%d pooled_relative_l2=%.9f p95_relative_l2=%.9f max_relative_l2=%.9f unique_min=%d unique_max=%d exact_factor_median_ms=%.9f q7_factor_median_ms=%.9f factor_ratio=%.6f source_bytes=%zu q7_bytes=%zu rss_bytes=%zu elapsed_seconds=%.6f checksum=%.9f gate=%s\n",
           states,route_mismatches,nonfinite,zero_norms,relative,p95,maximum,
           unique_min,unique_max,exact_med,quant_med,quant_med/exact_med,
           original_size,quant_size,rss_bytes(),now_s()-started,checksum,pass?"PASS":"FAIL");
    return pass?0:1;
}
