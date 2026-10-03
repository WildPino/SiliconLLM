// Timers only; immutable303 source/fixtures/math retained and exact-hash checked.
#define main meth303_immutable_main
#include "meth303_compact_i8_cpu.c"
#undef main

enum {MLA,ROUTED,SHARED,DENSE,MACRO,CHILD,INPUT,MIXTURE,HEAD,ORGANS};
static double organ_seconds[ORGANS],projection_seconds[ORGANS];
#define P_BEGIN(label) double profile_begin_##label=clock_s()
#define P_END(label,slot) organ_seconds[slot]+=clock_s()-profile_begin_##label
static void profile_apply(Matrix *m,const Input *q,unsigned slot){
    double begin=clock_s();checked_apply(m,q);projection_seconds[slot]+=clock_s()-begin;
}
#define P_APPLY(m,q,slot) profile_apply(m,q,slot)
static void print_array(const double *values){
    printf("[");for(unsigned i=0;i<ORGANS;i++)printf("%s%.12f",i?",":"",values[i]);printf("]");
}
static void profile_execute(void){
    for(unsigned l=0;l<LAYERS;l++){
        /*P*/ P_BEGIN(mla);
        Layer *s=&layers[l];norm(s->ra,s->an,D,s->xa);checked_quant(s->xa,D,&s->qa);
        P_APPLY(&s->q,&s->qa,MLA);P_APPLY(&s->kv,&s->qa,MLA);norm(s->kv.y,s->kn,512,s->kvnorm);
        for(unsigned h=0;h<HEADS;h++){checked_quant(s->q.y+h*192,128,&s->qk[h]);checked_quant(s->latent+h*512,512,&s->qv[h]);}
        P_APPLY(&s->kb,s->qk,MLA);P_APPLY(&s->vb,s->qv,MLA);checked_quant(s->vb.y,D,&s->qo);P_APPLY(&s->wo,&s->qo,MLA);
        /*P*/ P_END(mla,MLA); P_BEGIN(input);
        norm(s->rf,s->fn,D,s->xf);checked_quant(s->xf,D,&s->qf);
        /*P*/ P_END(input,INPUT);
        if(l){
            /*P*/ P_BEGIN(macro);
            macro_router(s);
            /*P*/ P_END(macro,MACRO); P_BEGIN(child);
            P_APPLY(&s->cq,&s->qf,CHILD);child_router(s);
            /*P*/ P_END(child,CHILD);
        }
        /*P*/ unsigned ffn_slot=l?ROUTED:DENSE; P_BEGIN(ffn);
        P_APPLY(&s->g,&s->qf,ffn_slot);P_APPLY(&s->u,&s->qf,ffn_slot);
        for(unsigned k=0;k<s->g.nactive;k++){
            for(unsigned j=0;j<s->g.o;j++)s->zg[k*s->g.o+j]=silu(s->g.y[k*s->g.o+j])*s->u.y[k*s->g.o+j];
            checked_quant(s->zg+k*s->g.o,s->g.o,&s->qz[k]);
        }
        P_APPLY(&s->down,s->qz,ffn_slot);
        /*P*/ P_END(ffn,ffn_slot);
        if(l){
            /*P*/ P_BEGIN(shared);
            P_APPLY(&s->sg,&s->qf,SHARED);P_APPLY(&s->su,&s->qf,SHARED);
            for(unsigned j=0;j<SH;j++)s->sz[j]=silu(s->sg.y[j])*s->su.y[j];
            checked_quant(s->sz,SH,&s->qs);P_APPLY(&s->sd,&s->qs,SHARED);
            /*P*/ P_END(shared,SHARED);
        }
        /*P*/ P_BEGIN(mixture);
        for(unsigned j=0;j<D;j++){
            float v=l?s->sd.y[j]:s->down.y[j];if(l)for(unsigned k=0;k<4;k++)v+=s->gates[k]*s->down.y[k*D+j];
            s->mixture[j]=v;
        }
        /*P*/ P_END(mixture,MIXTURE);
    }
    /*P*/ P_BEGIN(head);
    Layer *s=&layers[LAYERS-1];for(unsigned j=0;j<D;j++)head_input[j]=(s->xf[j]+s->wo.y[j])+s->mixture[j];
    norm(head_input,final_norm,D,head_input);if(!strat01_quantize_q8_k_row(head_input,D,hq))die("head Q8_K quantization");
    /*P*/ double head_projection_begin=clock_s();
    #pragma omp parallel for schedule(static)
    for(int r=0;r<V;r++)head_logits[r]=strat01_q6k_q8k_dot_avx2(head+(size_t)r*(D/256)*210,hq,D);
    /*P*/ projection_seconds[HEAD]+=clock_s()-head_projection_begin; P_END(head,HEAD);
}
int main(int argc,char **argv){
    if(argc!=3)die("usage: SPEC MODEL");fesetround(FE_TONEAREST);omp_set_dynamic(0);omp_set_num_threads(6);started=clock_s();initialize(argv[1],argv[2]);
    printf("{\"event\":\"ready\",\"parent_count\":64,\"children_per_parent\":10,\"functions_per_layer\":640,\"threads\":6,\"allocated_bytes\":%llu,\"active_i8_coefficients\":%llu,\"active_row_scale_bytes\":%llu,\"stored_i8_coefficients\":%llu,\"initialization_seconds\":%.9f}\n",(unsigned long long)allocated,(unsigned long long)active_i8,(unsigned long long)active_scales,(unsigned long long)code_capacity,clock_s()-started);fflush(stdout);
    selftest();unsigned char selected[26][BANKS];memset(selected,0,sizeof(selected));
    for(unsigned rep=0;rep<3;rep++){
        for(unsigned token=0;token<10;token++){
            prepare(token);memset(organ_seconds,0,sizeof(organ_seconds));memset(projection_seconds,0,sizeof(projection_seconds));double t=clock_s();profile_execute();double seconds=clock_s()-t;uint64_t hash=output_hash();
            for(unsigned l=1;l<LAYERS;l++)for(unsigned k=0;k<4;k++)selected[l][layers[l].flat[k]]=1;
            printf("{\"event\":\"operator\",\"rep\":%u,\"input\":%u,\"warmup\":%s,\"seconds\":%.12f,\"output_route_hash\":\"%016llx\",\"organ_seconds\":",rep,token,token<2?"true":"false",seconds,(unsigned long long)hash);
            print_array(organ_seconds);printf(",\"projection_seconds\":");print_array(projection_seconds);printf("}\n");fflush(stdout);
        }
        guards();
    }
    unsigned lo=BANKS,hi=0;for(unsigned l=1;l<LAYERS;l++){unsigned n=0;for(unsigned j=0;j<BANKS;j++)n+=selected[l][j];if(n<lo)lo=n;if(n>hi)hi=n;}
    printf("{\"event\":\"finished\",\"minimum_selected_child_union\":%u,\"maximum_selected_child_union\":%u,\"peak_rss_bytes\":%llu,\"total_seconds\":%.9f}\n",lo,hi,(unsigned long long)peak_rss(),clock_s()-started);return 0;
}
