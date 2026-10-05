/* METH476: cached source-prefix last-layer reconstruction, no candidate. */
#include "meth393_switch_router_audit.c"

static uint32_t current_batch,current_position;
static void same(const void *a,const void *b,size_t n,const char *why){
    if(memcmp(a,b,n)){fprintf(stderr,"batch:%u position:%u bytes:%llu\n",current_batch,current_position,(unsigned long long)n);require(0,why);}
}
static unsigned char *readfile(const char *path,size_t *bytes){
    FILE *f=fopen(path,"rb");require(f!=NULL,"cached_open");require(!_fseeki64(f,0,SEEK_END),"cached_seek");
    long long n=_ftelli64(f);require(n>0&&n<(1LL<<30),"cached_size");rewind(f);unsigned char *p=alloc((size_t)n,1);
    require(fread(p,1,(size_t)n,f)==(size_t)n&&fgetc(f)==EOF&&fclose(f)==0,"cached_read");*bytes=(size_t)n;return p;
}
static uint32_t at32(const void *p){uint32_t v;memcpy(&v,p,4);return v;}
static uint64_t at64(const void *p){uint64_t v;memcpy(&v,p,8);return v;}
static uint16_t at16(const void *p){uint16_t v;memcpy(&v,p,2);return v;}
static void put(FILE *f,const void *p,size_t n){require(fwrite(p,1,n,f)==n,"context_write");}

int main(int argc,char **argv){
    require(argc==8,"manifest jobs queries reference context integer_fixture integer_output");
    require(SetProcessAffinityMask(GetCurrentProcess(),21),"prospective_worker_pool");worker_setup(3);
    require(worker_physical_cores==6&&fegetround()==FE_TONEAREST,"actual_topology_rounding");
    head_integer_controls(argv[6],argv[7]);
    Model *m=load(argv[1]);Config c=m->c;
    require(c.d==768&&c.ff==3072&&c.enc==12&&c.dec==12&&c.n==128&&c.vocab==32128&&m->dec[11].sparse,"original128_dimensions");
    FILE *jobs=fopen(argv[2],"rb"),*out=fopen(argv[5],"wb");require(jobs&&out,"job_context_open");char magic[8];
    require(fread(magic,1,8,jobs)==8&&!memcmp(magic,"M476JOB1",8),"jobs_magic");
    uint32_t batches=u32(jobs),rows=u32(jobs),qbytes=u32(jobs),rowbytes=u32(jobs);uint64_t qoffset=u64(jobs),foffset=u64(jobs);
    require(batches==608&&rows==7993&&qbytes==4732&&rowbytes==4628,"jobs_counts");
    size_t qb,fb;unsigned char *queries=readfile(argv[3],&qb),*functions=readfile(argv[4],&fb);
    require(qb==qoffset+19962ULL*4732&&fb==foffset+19962ULL*3072,"query_reference_sizes");
    put(out,"M476CTX1",8);put(out,&rows,4);put(out,&rowbytes,4);
    uint32_t written=0,goldens=0,validation=0;uint64_t checked_logits=0;
    Block *b=m->dec+11;
    for(current_batch=0;current_batch<batches;current_batch++){
        uint32_t kind=u32(jobs),book=u32(jobs),ca=u32(jobs),mode=u32(jobs),s=u32(jobs),t=u32(jobs);uint64_t qstart=u64(jobs),ledgerbase=u64(jobs);
        require(ca<4&&mode<2&&s==29&&t>0&&t<=64,"batch_metadata");
        require((current_batch<96&&kind==0&&book<24&&mode==0&&t==14&&qstart==UINT64_MAX)||
                (current_batch>=96&&kind==1&&book>=64&&book<128&&qstart+t<=19962),"fixed_cohorts");
        char wholepath[1024],tracepath[1024],controlpath[1024];string(jobs,wholepath,sizeof(wholepath));string(jobs,tracepath,sizeof(tracepath));string(jobs,controlpath,sizeof(controlpath));
        uint32_t sourceids[CTX],decoderids[CTX];for(uint32_t i=0;i<s;i++)sourceids[i]=u32(jobs);for(uint32_t i=0;i<t;i++)decoderids[i]=u32(jobs);
        size_t wb,tb,cb=0;unsigned char *whole=readfile(wholepath,&wb),*trace=readfile(tracepath,&tb),*control=kind?NULL:readfile(controlpath,&cb);
        uint32_t nr=6*(s+t),header[]={s,t,768,12,12,32128,nr};same(whole,"SWR32O01",8,"whole_magic");same(whole+8,header,28,"whole_header");
        size_t en=(size_t)14*s*768,de=(size_t)14*t*768,lo=(size_t)t*32128;
        require(wb==36+4*(en+de+lo)+12*(size_t)nr&&tb==16+3592*(size_t)nr,"whole_trace_exact_EOF");
        same(trace,"SWRTA001",8,"trace_magic");uint32_t td[]={128,768};same(trace+8,td,8,"trace_dimensions");
        if(!kind){uint32_t ch[]={768,3072,32128,128,11,1};require(cb==32+52264*(size_t)t,"golden_size");same(control,"SWFUN001",8,"golden_magic");same(control+8,ch,24,"golden_header");}
        const float *enc=(const float *)(whole+36),*dec=enc+en,*original_logits=dec+de;
        const unsigned char *original_routes=(const unsigned char *)(original_logits+lo);
        for(uint32_t i=0;i<s;i++){require(sourceids[i]<32128,"source_id");same(enc+i*768,m->embed+(size_t)sourceids[i]*768,3072,"source_embedding");}
        Cache a={alloc((size_t)t*768,4),alloc((size_t)t*768,4),alloc((size_t)s*768,4),alloc((size_t)s*768,4)};
        mv_batch(b->cross.k,enc+(size_t)13*s*768,a.crossk,768,768,s);mv_batch(b->cross.v,enc+(size_t)13*s*768,a.crossv,768,768,s);
        for(current_position=0;current_position<t;current_position++){
            uint32_t pos=current_position,ix=6*s+5+6*pos;float h[768],pre[768],x[768],q[768],att[768],tmp[768],scores[128],upraw[3072],up[3072],down[768],logits[32128];int16_t wi_codes[768],wo_codes[3072],head_codes[768];
            const float *snap=dec+(size_t)pos*14*768;const unsigned char *tr=trace+16+(size_t)ix*3592,*route=original_routes+(size_t)ix*12,*gold=kind?NULL:control+32+(size_t)pos*52264;
            require(decoderids[pos]<32128,"decoder_id");same(snap,m->embed+(size_t)decoderids[pos]*768,3072,"decoder_embedding");require(at32(tr)==ix&&at32(tr+4)==1,"trace_position");
            memcpy(h,snap+11*768,3072);norm(h,b->selfnorm,x,768,c.epsilon);mv(b->self.q,x,q,768,768);mv(b->self.k,x,a.selfk+(size_t)pos*768,768,768);mv(b->self.v,x,a.selfv+(size_t)pos*768,768,768);
            attend(c,q,a.selfk,a.selfv,pos+1,pos,0,0,m->decbias,att);mv(b->self.o,att,tmp,768,768);for(int j=0;j<768;j++)h[j]+=tmp[j];
            norm(h,b->crossnorm,x,768,c.epsilon);mv(b->cross.q,x,q,768,768);attend(c,q,a.crossk,a.crossv,s,pos,0,1,NULL,att);mv(b->cross.o,att,tmp,768,768);for(int j=0;j<768;j++)h[j]+=tmp[j];
            memcpy(pre,h,3072);for(int j=0;j<768;j++)require(isfinite(pre[j]),"pre_finite");norm(h,b->ffnorm,x,768,c.epsilon);same(x,tr+8,3072,"actual_ffn_norm_input");
            mv(b->router,x,scores,128,768);same(scores,tr+3080,512,"ALL128_router_scores");int chosen=0;for(int e=1;e<128;e++)if(scores[e]>scores[chosen])chosen=e;
            double sum=0.;for(int e=0;e<128;e++)sum+=(double)(float)exp((double)(scores[e]-scores[chosen]));float probability=(float)(1./sum);
            require(at32(route)==(uint32_t)chosen&&at32(route+4)==1,"native_winner_accepted");same(&probability,route+8,4,"native_normalization_mass");
            float alpha=head_activation_codes(x,wi_codes,768);mv(b->expertwi[chosen],x,upraw,3072,768);memcpy(up,upraw,sizeof(up));for(int j=0;j<3072;j++)if(up[j]<0.f)up[j]=0.f;
            float wo_alpha=head_activation_codes(up,wo_codes,3072);mv(b->expertwo[chosen],up,down,768,3072);
            uint64_t qid=kind?qstart+pos:UINT64_MAX;
            if(kind){const unsigned char *qr=queries+qoffset+qid*4732;
                require(at64(qr)==ledgerbase+ix&&at16(qr+8)==book&&qr[10]==ca&&qr[11]==mode&&qr[12]==1&&qr[13]==1&&at16(qr+14)==chosen&&at32(qr+16)==ix,"ALL_validation_query_identity");
                same(&alpha,qr+20,4,"query_A16_alpha");same(&probability,qr+24,4,"query_probability");same(x,qr+28,3072,"query_input");same(wi_codes,qr+3100,1536,"query_A16_codes");same(down,functions+foffset+qid*3072,3072,"saved472_reference_function");validation++;
            }else{
                require(at32(gold)==pos&&at32(gold+4)==decoderids[pos]&&at32(gold+8)==s&&at32(gold+12)==ix,"golden_identity");
                same(pre,gold+16,3072,"known_true_pre");same(x,gold+3088,3072,"golden_input");same(&alpha,gold+6160,4,"golden_WI_alpha");same(wi_codes,gold+6164,1536,"golden_WI_codes");same(upraw,gold+7700,12288,"golden_raw_WI");same(up,gold+19988,12288,"golden_ReLU");same(&wo_alpha,gold+32276,4,"golden_WO_alpha");same(wo_codes,gold+32280,6144,"golden_WO_codes");same(down,gold+38424,3072,"golden_down");same(route,gold+41496,12,"golden_route");goldens++;
            }
            for(int j=0;j<768;j++)h[j]+=probability*down[j];same(h,snap+12*768,3072,"actual_post_last_FFN");if(!kind)same(h,gold+41508,3072,"golden_post");
            norm(h,m->decnorm,x,768,c.epsilon);same(x,snap+13*768,3072,"actual_final_norm");if(!kind)same(x,gold+44580,3072,"golden_final");float rescale=(float)(1./sqrt(768.));for(int j=0;j<768;j++)x[j]*=rescale;
            float head_alpha=head_activation_codes(x,head_codes,768);if(!kind){same(x,gold+47652,3072,"golden_head_input");same(&head_alpha,gold+50724,4,"golden_head_alpha");same(head_codes,gold+50728,1536,"golden_head_codes");}
            head_mv(m->head,x,logits,32128,768);same(logits,original_logits+(size_t)pos*32128,32128*4,"ALL32128_original_logits");checked_logits+=32128;
            put(out,&qid,8);put(out,&current_batch,4);put(out,&pos,4);put(out,pre,3072);put(out,head_codes,1536);put(out,&head_alpha,4);written++;
        }
        require(fflush(out)==0,"context_flush");printf("{\"batch\":%u,\"kind\":%u,\"book\":%u,\"case\":%u,\"mode\":%u,\"positions\":%u,\"cumulative_positions\":%u,\"byte_exact\":true}\n",current_batch,kind,book,ca,mode,t,written);fflush(stdout);
        free(a.selfk);free(a.selfv);free(a.crossk);free(a.crossv);free(whole);free(trace);free(control);
    }
    require(written==7993&&goldens==1344&&validation==6649&&checked_logits==7993ULL*32128&&fgetc(jobs)==EOF&&fclose(jobs)==0&&fclose(out)==0,"complete_ALL_positions_EOF");
    printf("{\"terminal\":true,\"positions\":%u,\"goldens\":%u,\"validation\":%u,\"checked_logits\":%llu",written,goldens,validation,(unsigned long long)checked_logits);worker_json();printf("}\n");fflush(stdout);
    free(queries);free(functions);unload(m);return 0;
}
