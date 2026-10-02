// Original285 equation; cache positions relative, RoPE positions absolute.
static void cc_forward_window(CcState *s,int token,int pos,int experts,int head,int position_offset){
    if(token<0 || token>=CC_V || pos<0 || pos>=s->maxseq)die("token/context binding");
    for(int d=0;d<D;d++)s->x[d]=bf16_to_float(cc_embedding[(size_t)token*D+d]);
    float co[32],si[32];
    for(int j=0;j<32;j++){float frequency=1.0f/powf(1000000.0f,(float)(2*j)/64.0f);float phase=(float)(pos+position_offset)*frequency;co[j]=cc_bf(cosf(phase));si[j]=cc_bf(sinf(phase));}
    for(int li=0;li<L;li++){
        const CcLayer *l=&cc_layers[li];cc_norm(s->x,l->in_norm,s->norm);
        cc_mat(l->q,s->norm,l->qb,s->q,D,D);cc_mat(l->k,s->norm,l->kb,s->k,128,D);cc_mat(l->v,s->norm,l->vb,s->v,128,D);
        cc_rope(s->q,CC_NH,co,si);cc_rope(s->k,CC_NKV,co,si);
        memcpy(s->kc+((size_t)li*s->maxseq+pos)*128,s->k,128*4);memcpy(s->vc+((size_t)li*s->maxseq+pos)*128,s->v,128*4);
        #pragma omp parallel for schedule(static)
        for(int h=0;h<CC_NH;h++){
            float *scores=s->scores+(size_t)h*s->maxseq;float maximum=-INFINITY;
            for(int t=0;t<=pos;t++){
                const float *k=s->kc+((size_t)li*s->maxseq+t)*128+(h/7)*64;float value=cc_dot(k,s->q+h*64,64)*0.125f;
                scores[t]=value;if(value>maximum)maximum=value;
            }
            float total=0;for(int t=0;t<=pos;t++){scores[t]=expf(scores[t]-maximum);total+=scores[t];}
            for(int d=0;d<64;d++){
                float value=0;for(int t=0;t<=pos;t++)value+=(scores[t]/total)*s->vc[((size_t)li*s->maxseq+t)*128+(h/7)*64+d];
                s->attention[h*64+d]=cc_bf(value);
            }
        }
        cc_mat(l->o,s->attention,NULL,s->tmp,D,D);for(int d=0;d<D;d++)s->x[d]=cc_bf(s->x[d]+s->tmp[d]);
        cc_norm(s->x,l->post_norm,s->norm);uint16_t bits[D];for(int d=0;d<D;d++)bits[d]=cc_bf_bits(s->norm[d]);
        row_forward(li,bits,s->tmp);
        float conditional[D];int parents[4],children[4],aliases[4];float gates[4];
        if(experts)cc_conditional(li,s->norm,conditional,parents,children,aliases,gates);
        for(int d=0;d<D;d++){float ffn=cc_bf(s->tmp[d]);if(experts)ffn=cc_bf(ffn+conditional[d]);s->x[d]=cc_bf(s->x[d]+ffn);}
    }
    cc_norm(s->x,cc_final_norm,s->norm);
    if(head)cc_mat(cc_embedding,s->norm,NULL,s->logits,CC_V,D);
}
