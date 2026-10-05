/* Source-prefix observable diagnostic; no candidate FFN recomputation or fit. */
#include "meth393_switch_router_audit.c"
#include <bcrypt.h>
static BCRYPT_ALG_HANDLE sha_algorithm;
static void sha256(const void *data,ULONG bytes,unsigned char *result){BCRYPT_HASH_HANDLE hash;require(BCryptCreateHash(sha_algorithm,&hash,NULL,0,NULL,0,0)>=0,"SHA_create");require(BCryptHashData(hash,(PUCHAR)data,bytes,0)>=0&&BCryptFinishHash(hash,result,32,0)>=0,"SHA_bytes_finish");require(BCryptDestroyHash(hash)>=0,"SHA_destroy");}
static void same(const void *a,const void *b,size_t n,const char *why){require(!memcmp(a,b,n),why);}
static unsigned char *readfile(const char *p,size_t *bytes){FILE *f=fopen(p,"rb");require(f!=NULL,"input_open");require(!_fseeki64(f,0,SEEK_END),"input_seek");long long n=_ftelli64(f);require(n>0&&n<(1LL<<30),"input_size");rewind(f);unsigned char *d=alloc((size_t)n,1);require(fread(d,1,n,f)==(size_t)n&&fgetc(f)==EOF&&fclose(f)==0,"input_EOF");*bytes=(size_t)n;return d;}
static uint32_t at32(const void *p){uint32_t v;memcpy(&v,p,4);return v;}
static uint64_t at64(const void *p){uint64_t v;memcpy(&v,p,8);return v;}
static uint16_t at16(const void *p){uint16_t v;memcpy(&v,p,2);return v;}
static void put(FILE *f,const void *p,size_t n){require(fwrite(p,1,n,f)==n,"output_write");}
typedef struct{double value,correction;} Sum;
static void add(Sum *s,double x){double t=s->value+x;if(fabs(s->value)>=fabs(x))s->correction+=(s->value-t)+x;else s->correction+=(x-t)+s->value;s->value=t;}
static double total(Sum s){return s.value+s.correction;}
typedef struct{uint32_t arg0,arg1,ties0,ties1;double v[16];} Metric;
static Metric measure(const float *s,const float *c,int n,uint32_t label){
    Metric m={0};float ms=s[0],mc=c[0];for(int j=1;j<n;j++){require(isfinite(s[j])&&isfinite(c[j]),"logits_finite");if(s[j]>ms){ms=s[j];m.arg0=j;}if(c[j]>mc){mc=c[j];m.arg1=j;}}
    require(isfinite(ms)&&isfinite(mc)&&n>=2,"softmax_domain");Sum zs={0},zc={0},mu={0};double dmin=INFINITY,dmax=-INFINITY;
    for(int j=0;j<n;j++){add(&zs,exp((double)s[j]-ms));add(&zc,exp((double)c[j]-mc));double d=(double)c[j]-(double)s[j];if(d<dmin)dmin=d;if(d>dmax)dmax=d;}
    double z0=total(zs),z1=total(zc);require(z0>=1.&&z1>=1.&&isfinite(z0)&&isfinite(z1),"softmax_partition");
    for(int j=0;j<n;j++)add(&mu,(exp((double)s[j]-ms)/z0)*((double)c[j]-s[j]));double mean=total(mu);
    Sum tv={0},fisher={0},curvature={0},center_mean={0};float second0=-INFINITY,second1=-INFINITY;
    for(int j=0;j<n;j++){
        double p=exp((double)s[j]-ms)/z0,q=exp((double)c[j]-mc)/z1,z=(double)c[j]-s[j]-mean;add(&tv,fabs(p-q));add(&fisher,p*z*z);
        if(dmax-dmin<=.5){add(&curvature,p*(expm1(z)-z));add(&center_mean,p*z);}
        m.ties0+=s[j]==ms;m.ties1+=c[j]==mc;if(j!=(int)m.arg0&&s[j]>second0)second0=s[j];if(j!=(int)m.arg1&&c[j]>second1)second1=c[j];
    }
    double centered=total(center_mean),cor=total(curvature);
    double kl=dmax-dmin<=.5?log1p(centered+cor)-centered:((double)mc-ms)+log(z1)-log(z0)-mean;
    double range=dmax-dmin,bound=range*range/8.;require(isfinite(kl)&&kl>=-1e-11&&kl<=bound+1e-11+1e-10*bound,"KL_numeric_nonnegative_range_bound");
    m.v[0]=kl;m.v[1]=total(tv)/2.;m.v[2]=(double)ms-second0;m.v[3]=(double)mc-second1;m.v[4]=dmin;m.v[5]=dmax;m.v[6]=1./z0;m.v[7]=1./z1;m.v[8]=total(fisher)/2.;
    if(label!=(uint32_t)-1){require(label<(uint32_t)n,"true_label_range");m.v[9]=((double)ms-s[label])+log(z0);m.v[10]=((double)mc-c[label])+log(z1);m.v[11]=m.v[10]-m.v[9];}
    m.v[15]=bound;return m;
}
static void toy_controls(void){
    float s[][3]={{0,1,-1},{10000,9999,-10000},{0,0,0},{1000,999,-1000},{0,-1,-2},{0,0,-1}};
    float c[][3]={{0,1+0x1p-10f,-1-0x1p-10f},{10016,10015,-9984},{0,0,0},{999,1000,-999},{0,-1+0x1p-20f,-2-0x1p-20f},{0,0x1p-10f,-1}};uint32_t labels[]={1,0,2,1,2,0};
    for(int i=0;i<6;i++){Metric v=measure(s[i],c[i],3,labels[i]);printf("{\"control\":%d,\"metrics\":[",i);for(int j=0;j<16;j++)printf("%s%.17g",j?",":"",v.v[j]);printf("]}\n");}
    unsigned char h0[32],h1[32];sha256("",0,h0);sha256("abc",3,h1);printf("{\"SHA_controls\":[\"");for(int j=0;j<32;j++)printf("%02x",h0[j]);printf("\",\"");for(int j=0;j<32;j++)printf("%02x",h1[j]);printf("\"]}\n");fflush(stdout);
}
int main(int argc,char **argv){
    require(SetProcessAffinityMask(GetCurrentProcess(),21),"worker_pool");worker_setup(3);require(worker_physical_cores==6&&fegetround()==FE_TONEAREST,"topology_rounding");require(BCryptOpenAlgorithmProvider(&sha_algorithm,BCRYPT_SHA256_ALGORITHM,NULL,0)>=0,"SHA_algorithm");
    if(argc==4&&!strcmp(argv[1],"--controls")){head_integer_controls(argv[2],argv[3]);toy_controls();printf("{\"control_terminal\":true,\"integer_fixtures\":13");worker_json();printf("}\n");fflush(stdout);require(BCryptCloseAlgorithmProvider(sha_algorithm,0)>=0,"SHA_control_close");return 0;}
    require(argc==8,"manifest jobs queries reference candidate context metrics");Model *model=load(argv[1]);Config cf=model->c;require(cf.d==768&&cf.ff==3072&&cf.vocab==32128&&cf.n==128&&cf.dec==12,"original_dimensions");
    FILE *jobs=fopen(argv[2],"rb"),*out=fopen(argv[7],"wb");require(jobs&&out,"jobs_output");char magic[8];require(fread(magic,1,8,jobs)==8&&!memcmp(magic,"M477JOB1",8),"job_magic");uint32_t batches=u32(jobs),positions=u32(jobs),observations=u32(jobs);uint64_t qo=u64(jobs),fo=u64(jobs),co=u64(jobs);
    require(batches==608&&positions==7993&&observations==6649,"ALL_cohorts");size_t qb,fb,cb,xb;unsigned char *queries=readfile(argv[3],&qb),*reference=readfile(argv[4],&fb),*candidate=readfile(argv[5],&cb),*context=readfile(argv[6],&xb);
    require(qb==qo+19962ULL*4732&&fb==fo+19962ULL*3072&&cb==co+19962ULL*3072&&xb==16+7993ULL*4628,"input_sizes");same(context,"M476CTX1",8,"qualified_context_magic");require(at32(context+8)==7993&&at32(context+12)==4628,"context_dimensions");
    uint32_t rowbytes=1748;put(out,"M477OBS1",8);put(out,&observations,4);put(out,&rowbytes,4);uint32_t observed=0,checked=0,labels=0,masked=0,fallback=0,changed=0;uint64_t head_rows=0;
    for(uint32_t batch=0;batch<batches;batch++){
        uint32_t kind=u32(jobs),book=u32(jobs),ca=u32(jobs),mode=u32(jobs),s=u32(jobs),t=u32(jobs);uint64_t qs=u64(jobs),xs=u64(jobs),ledgerbase=u64(jobs);char wp[1024],gp[1024];string(jobs,wp,sizeof(wp));string(jobs,gp,sizeof(gp));
        require(ca<4&&mode<2&&s==29&&t>0&&t<=64&&xs+t<=7993,"batch_shape");require((!kind&&batch<96&&book<24&&mode==0&&t==14)||(kind==1&&batch>=96&&book>=64&&book<128&&qs+t<=19962),"fixed_roles");
        uint32_t target[CTX],flags[CTX];for(uint32_t p=0;p<t;p++){target[p]=u32(jobs);flags[p]=u32(jobs);require(flags[p]<32,"prospective_flags");}
        size_t wb,gb=0;unsigned char *whole=readfile(wp,&wb),*golden=kind?NULL:readfile(gp,&gb);uint32_t h[]={s,t,768,12,12,32128,6*(s+t)};same(whole,"SWR32O01",8,"whole_magic");same(whole+8,h,28,"whole_header");
        size_t en=(size_t)14*s*768,de=(size_t)14*t*768,lo=(size_t)t*32128;require(wb==36+4*(en+de+lo)+12*(size_t)6*(s+t),"whole_EOF");const float *snap=(const float *)(whole+36)+en,*original=snap+de;const unsigned char *routes0=(const unsigned char *)(original+lo);
        if(!kind)require(gb==32+52264*(size_t)t,"golden_EOF");
        for(uint32_t pos=0;pos<t;pos++){
            const unsigned char *ctx=context+16+(xs+pos)*4628,*qr=kind?queries+qo+(qs+pos)*4732:NULL,*gd=kind?NULL:golden+32+(size_t)pos*52264;const float *pre=(const float *)(ctx+16),*f=kind?(const float *)(reference+fo+(qs+pos)*3072):(const float *)(gd+38424);
            uint64_t qid=kind?qs+pos:UINT64_MAX;require(at64(ctx)==qid&&at32(ctx+8)==batch&&at32(ctx+12)==pos,"context_row_identity");uint32_t ix=179+6*pos;const unsigned char *route=routes0+(size_t)ix*12;float probability;memcpy(&probability,route+8,4);
            float post[768],x[768],l0[32128],l1[32128];int16_t hc[768];if(kind){require(at64(qr)==ledgerbase+ix&&at16(qr+8)==book&&qr[10]==ca&&qr[11]==mode&&qr[12]==1&&qr[13]==1&&at32(qr+16)==ix&&at16(qr+14)==at32(route)&&at32(route+4)==1,"query_route_identity");same(qr+24,&probability,4,"original_probability");norm(pre,model->dec[11].ffnorm,x,768,cf.epsilon);same(x,qr+28,3072,"original_FFN_input");}
            else {same(pre,gd+16,3072,"truepre_golden");same(route,gd+41496,12,"golden_route");}
            for(int j=0;j<768;j++)post[j]=pre[j]+probability*f[j];same(post,snap+((size_t)pos*14+12)*768,3072,"source_residual");norm(post,model->decnorm,x,768,cf.epsilon);same(x,snap+((size_t)pos*14+13)*768,3072,"source_finalnorm");float scale=(float)(1./sqrt(768.));for(int j=0;j<768;j++)x[j]*=scale;
            float ha=head_activation_codes(x,hc,768);same(hc,ctx+3088,1536,"source_head_codes");same(&ha,ctx+4624,4,"source_head_alpha");head_mv(model->head,x,l0,32128,768);same(l0,original+(size_t)pos*32128,32128*4,"ALL_source_logits");checked++;head_rows+=32128;
            if(kind){
                const float *f1=(const float *)(candidate+co+(qs+pos)*3072);if(!(flags[pos]&1)){same(f,f1,3072,"fallback_functions_BYTE_exact");fallback++;}
                for(int j=0;j<768;j++){require(isfinite(pre[j])&&isfinite(f[j])&&isfinite(f1[j]),"functions_pre_finite");post[j]=pre[j]+probability*f1[j];}
                norm(post,model->decnorm,x,768,cf.epsilon);for(int j=0;j<768;j++)x[j]*=scale;float candidate_alpha=head_activation_codes(x,hc,768);head_mv(model->head,x,l1,32128,768);head_rows+=32128;unsigned char candidate_sha[32];sha256(l1,32128*4,candidate_sha);
                require((mode==0&&(flags[pos]&4)&&target[pos]<32128)||(mode==1&&!(flags[pos]&12)&&target[pos]==UINT32_MAX),"label_provenance_role");Metric m=measure(l0,l1,32128,target[pos]);
                Sum ee={0},re={0};for(int j=0;j<768;j++){double d=(double)f1[j]-f[j];add(&ee,d*d);add(&re,(double)f[j]*f[j]);}m.v[12]=total(ee);m.v[13]=total(re);m.v[14]=probability;
                uint32_t fl=flags[pos];if(m.ties0==1)fl|=32;if(m.ties1==1)fl|=64;if(m.arg0!=m.arg1){fl|=128;changed++;}if(m.ties0==1&&m.v[2]>m.v[5]-m.v[4]){fl|=256;require(m.arg0==m.arg1,"margin_stability_theorem");}
                if(!(flags[pos]&1)){same(l0,l1,32128*4,"fallback_ALL_logits_identity");require(m.arg0==m.arg1&&m.v[0]==0.&&m.v[1]==0.,"fallback_observable_identity");}
                uint32_t cri=(uint32_t)(xs+pos),zero=0;uint16_t bk=(uint16_t)book,e=at16(qr+14),ff=(uint16_t)fl;uint8_t cc=(uint8_t)ca,mm=(uint8_t)mode;
                put(out,&qid,8);put(out,&cri,4);put(out,&bk,2);put(out,&cc,1);put(out,&mm,1);put(out,&e,2);put(out,&ff,2);put(out,&pos,4);put(out,&m.arg0,4);put(out,&m.arg1,4);put(out,&target[pos],4);put(out,&m.ties0,4);put(out,&m.ties1,4);put(out,&zero,4);put(out,m.v,16*8);put(out,hc,1536);put(out,&candidate_alpha,4);put(out,candidate_sha,32);observed++;labels+=!!(flags[pos]&4);masked+=!!(flags[pos]&8);
            }
        }
        require(fflush(out)==0,"observations_flush");printf("{\"batch\":%u,\"source_positions\":%u,\"candidate_positions\":%u}\n",batch,checked,observed);fflush(stdout);free(whole);free(golden);
    }
    require(checked==7993&&observed==6649&&labels==3584&&masked==2048&&head_rows==470422176&&fgetc(jobs)==EOF&&fclose(jobs)==0&&fclose(out)==0,"ALL_terminal_counts_EOF");
    printf("{\"terminal\":true,\"source_positions\":%u,\"candidate_positions\":%u,\"teacher_labels\":%u,\"masked_content_labels\":%u,\"fallback\":%u,\"argmax_changed\":%u,\"head_rows\":%llu",checked,observed,labels,masked,fallback,changed,(unsigned long long)head_rows);worker_json();printf("}\n");fflush(stdout);free(queries);free(reference);free(candidate);free(context);unload(model);require(BCryptCloseAlgorithmProvider(sha_algorithm,0)>=0,"SHA_close");return 0;
}
