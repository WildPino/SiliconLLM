// Phase 50.0 - Unit-Choice Cartography (the MAP).
//
// Cheap, model-light, NO substrate, NO training. Pure corpus statistics on TinyStories
// (train = first 90%, held-out = last 10%). For each candidate UNIT it reports, with a FAIR
// n-gram model (interpolated absolute-discounting / KN-lite, order 1 and 2), the cross-entropy
// expressed in bits per BYTE (unit-invariant) INCLUDING byte-fallback for OOV (lossless/honest),
// plus repeat-mass (flood propensity), mean bytes/unit, vocab size, coverage, and per-unit entropy.
//
// Candidates: byte | word (whitespace+punct, top-N + byte-fallback) | BPE-512/1024/4096
//             | bytepair (2-byte stride, lossless V=65792) | hash3gram (3-byte stride hashed, LOSSY).
//
// The unit "wins consideration" (Architect reads; ADVISORY) if it LOWERS-OR-HOLDS BPB/byte AND
// COLLAPSES repeat-mass (dissolves the flood) AND shortens the sequence AND stays feasible.
//
// Build: gcc -O3 -march=native phase50_0_map.c -o bin/phase50_0_map.exe
// Run:   bin/phase50_0_map.exe <corpus> [--max-bytes N] [--word-topn N]

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

// ----------------------------------------------------------------------------- FNV-1a 64
static inline uint64_t fnv64(const unsigned char* p, int n){
    uint64_t h=1469598103934665603ULL;
    for(int i=0;i<n;i++){ h^=p[i]; h*=1099511628211ULL; }
    return h;
}

// ----------------------------------------------------------------------------- generic u64->slot map
// open addressing; stores key + a 32-bit value + a 32-bit aux. Returns slot index; *isnew set.
typedef struct { uint64_t* key; uint32_t* val; uint32_t* aux; size_t cap, mask, n; } Map;
static void map_init(Map* m, size_t cap_pow2){
    m->cap=cap_pow2; m->mask=cap_pow2-1; m->n=0;
    m->key=(uint64_t*)calloc(cap_pow2,sizeof(uint64_t));
    m->val=(uint32_t*)calloc(cap_pow2,sizeof(uint32_t));
    m->aux=(uint32_t*)calloc(cap_pow2,sizeof(uint32_t));
    if(!m->key||!m->val||!m->aux){ fprintf(stderr,"map_init OOM cap=%zu\n",cap_pow2); exit(1); }
}
static void map_free(Map* m){ free(m->key); free(m->val); free(m->aux); m->key=0; }
// find-or-insert; stored key = true_key+1 so that sentinel 0 (empty) never collides with key 0.
static inline size_t map_slot(Map* m, uint64_t k, int* isnew){
    uint64_t sk = k+1ULL;
    size_t i = (size_t)(sk*11400714819323198485ULL) & m->mask;
    for(;;){
        if(m->key[i]==0){ m->key[i]=sk; m->n++; *isnew=1; return i; }
        if(m->key[i]==sk){ *isnew=0; return i; }
        i=(i+1)&m->mask;
    }
}

// ----------------------------------------------------------------------------- n-gram eval (KN-lite)
// train_ids[0..ntr), held_ids[0..nhe), vocab V, held byte count -> bpb order1/order2 + bits/unit + p90.
static void eval_ngram(const uint32_t* tr, size_t ntr, const uint32_t* he, size_t nhe,
                       int V, double held_bytes,
                       double* bpb1, double* bpb2, double* bpu2, double* p90){
    const double D=0.75;
    uint32_t* uni  = (uint32_t*)calloc((size_t)V,sizeof(uint32_t));
    uint32_t* ctot = (uint32_t*)calloc((size_t)V,sizeof(uint32_t));  // # times h is a left context
    uint32_t* ndis = (uint32_t*)calloc((size_t)V,sizeof(uint32_t));  // # distinct continuations of h
    if(!uni||!ctot||!ndis){ fprintf(stderr,"ngram arrays OOM V=%d\n",V); exit(1); }
    // size bigram map by #distinct bigrams (<= ntr-1) to keep load < 0.7 and never fill (map_slot
    // has no full-table guard, so a full map would spin forever). Reused/freed per candidate.
    size_t want=(size_t)((double)ntr*1.6)+1024; size_t bcap=1; while(bcap<want) bcap<<=1;
    if(bcap>(1ULL<<28)) bcap=1ULL<<28;
    Map bg; map_init(&bg,bcap);
    // counts
    for(size_t i=0;i<ntr;i++) uni[tr[i]]++;
    size_t types=0; for(int w=0;w<V;w++) if(uni[w]) types++;
    for(size_t i=0;i+1<ntr;i++){
        uint32_t h=tr[i], w=tr[i+1];
        uint64_t k=(uint64_t)h*(uint64_t)V + w; int isnew;
        size_t s=map_slot(&bg,k,&isnew); bg.val[s]++;
        ctot[h]++; if(isnew) ndis[h]++;
    }
    double Tt=(double)ntr;
    // eval
    double b1=0.0,b2=0.0; long hist[200]; for(int i=0;i<200;i++) hist[i]=0; long hn=0;
    for(size_t i=0;i<nhe;i++){
        uint32_t w=he[i];
        double p1 = (double)(uni[w]>0? (uni[w]-D):0.0)/Tt + (D*(double)types/Tt)*(1.0/(double)V);
        double s1=-log2(p1); b1+=s1;
        double p2;
        if(i==0){ p2=p1; }
        else {
            uint32_t h=he[i-1];
            if(ctot[h]>0){
                uint64_t k=(uint64_t)h*(uint64_t)V + w; int isnew;
                size_t s=map_slot(&bg,k,&isnew); uint32_t c=isnew?0:bg.val[s];
                double disc=(c>0)?((double)c-D):0.0;
                p2 = disc/(double)ctot[h] + (D*(double)ndis[h]/(double)ctot[h])*p1;
            } else p2=p1;
        }
        double s2=-log2(p2); b2+=s2;
        int bi=(int)(s2/0.25); if(bi<0)bi=0; if(bi>199)bi=199; hist[bi]++; hn++;
    }
    *bpb1=b1/held_bytes; *bpb2=b2/held_bytes; *bpu2=b2/(double)nhe;
    long acc=0; double pp=0; for(int i=0;i<200;i++){ acc+=hist[i]; if((double)acc/(double)hn>=0.90){ pp=(i+0.5)*0.25; break; } }
    *p90=pp;
    map_free(&bg); free(uni); free(ctot); free(ndis);
}

// ----------------------------------------------------------------------------- candidate result
typedef struct {
    const char* name; int V; double bytes_per_unit; double repeat_mass;
    double bpb1, bpb2, bpu2, p90; double cover; size_t ntok; int lossless;
} Row;

static double repeat_mass(const uint32_t* ids, size_t n){
    if(n<2) return 0.0; size_t r=0; for(size_t i=1;i<n;i++) if(ids[i]==ids[i-1]) r++;
    return (double)r/(double)(n-1);
}

// =============================================================================== BYTE
static size_t tok_byte(const unsigned char* buf, size_t a, size_t b, uint32_t* out){
    size_t n=0; for(size_t i=a;i<b;i++) out[n++]=buf[i]; return n;
}

// =============================================================================== WORD (letters), top-N + byte-fallback
// build word freq on train, select top-N by count -> ids 256.. ; tokenize letter-runs, OOV->bytes.
static int isletter(unsigned char c){ return (c>='A'&&c<='Z')||(c>='a'&&c<='z'); }
static Map g_word;            // hash(word)-> val=id (0=unset sentinel via aux), aux=count
static int g_word_built=0; static int g_word_nvocab=0;
static void word_build(const unsigned char* buf, size_t trn, int topn){
    map_init(&g_word, 1ULL<<21);
    // count
    size_t i=0; while(i<trn){
        if(isletter(buf[i])){ size_t j=i; while(j<trn&&isletter(buf[j])) j++;
            uint64_t h=fnv64(buf+i,(int)(j-i)); int nw; size_t s=map_slot(&g_word,h,&nw); g_word.aux[s]++; i=j; }
        else i++;
    }
    // collect counts, find a threshold for top-N via simple histogram of counts
    // (counts can be large; pick threshold so #(count>=thr) <= topn)
    size_t ne=0; for(size_t s=0;s<g_word.cap;s++) if(g_word.key[s]) ne++;
    // gather counts
    uint32_t* cnts=(uint32_t*)malloc(ne*sizeof(uint32_t)); size_t k=0;
    for(size_t s=0;s<g_word.cap;s++) if(g_word.key[s]) cnts[k++]=g_word.aux[s];
    // partial sort descending: simple threshold search by counting (counts bounded)
    // find thr = the topn-th largest count
    uint32_t thr=2;
    if(ne>(size_t)topn){
        // histogram over counts up to cap 1<<20
        size_t HC=1<<20; uint32_t* hc=(uint32_t*)calloc(HC,sizeof(uint32_t));
        for(size_t x=0;x<ne;x++){ uint32_t c=cnts[x]; if(c>=HC)c=HC-1; hc[c]++; }
        long need=topn; long acc=0; thr=HC-1;
        for(long c=(long)HC-1;c>=1;c--){ acc+=hc[c]; if(acc>=need){ thr=(uint32_t)c; break; } }
        free(hc);
        if(thr<2) thr=2;
    }
    free(cnts);
    // assign ids to entries with count>=thr (cap at topn)
    int nextid=256; for(size_t s=0;s<g_word.cap;s++){
        if(g_word.key[s] && g_word.aux[s]>=thr && nextid<256+topn){ g_word.val[s]=(uint32_t)nextid; nextid++; }
        else if(g_word.key[s]) g_word.val[s]=0; // 0 => fallback (id 0 is byte 0x00, but we mark fallback by val==0 AND treat specially)
    }
    g_word_nvocab=nextid; g_word_built=1;
}
static size_t tok_word(const unsigned char* buf, size_t a, size_t b, uint32_t* out, double* cover_bytes){
    size_t n=0; double cb=0; size_t i=a;
    while(i<b){
        if(isletter(buf[i])){ size_t j=i; while(j<b&&isletter(buf[j])) j++;
            uint64_t h=fnv64(buf+i,(int)(j-i)); int nw; size_t s=map_slot(&g_word,h,&nw);
            uint32_t id = (!nw)? g_word.val[s] : 0;
            if(id>=256){ out[n++]=id; cb+=(double)(j-i); }
            else { for(size_t t=i;t<j;t++) out[n++]=buf[t]; }   // byte fallback
            i=j;
        } else { out[n++]=buf[i]; i++; }                          // non-letter byte = byte unit
    }
    *cover_bytes=cb; return n;
}

// =============================================================================== BPE (byte-level, word/space chunks)
// pre-token = maximal run of non-space OR maximal run of space. Train merges to 4096 on train chunks.
typedef struct { uint32_t off,len,freq; } Word;
static unsigned char* g_arena=0; static size_t g_arena_n=0;
static Word* g_words=0; static size_t g_nwords=0;
// learned merges: pair (a,b) -> new id; ranks in order
static uint32_t g_mA[4096], g_mB[4096]; static int g_nmerge=0;
static Map g_mergemap; // key = (a<<20)|b ... but ids up to 256+4096<4352 -> key=a*8192+b
static int issp(unsigned char c){ return c==' '||c=='\t'||c=='\n'||c=='\r'; }

static void bpe_collect_words(const unsigned char* buf, size_t trn){
    Map dedupe; map_init(&dedupe,1ULL<<21);
    size_t arena_cap=1ULL<<24; g_arena=(unsigned char*)malloc(arena_cap); g_arena_n=0;
    size_t wcap=1ULL<<20; g_words=(Word*)malloc(wcap*sizeof(Word)); g_nwords=0;
    size_t i=0;
    while(i<trn){
        size_t j=i; int sp=issp(buf[i]);
        while(j<trn && (issp(buf[j])?1:0)==sp) j++;
        int len=(int)(j-i); uint64_t h=fnv64(buf+i,len);
        int nw; size_t s=map_slot(&dedupe,h,&nw);
        if(nw){
            if(g_arena_n+len>arena_cap){ arena_cap*=2; g_arena=(unsigned char*)realloc(g_arena,arena_cap); }
            if(g_nwords>=wcap){ wcap*=2; g_words=(Word*)realloc(g_words,wcap*sizeof(Word)); }
            dedupe.val[s]=(uint32_t)g_nwords;
            g_words[g_nwords].off=(uint32_t)g_arena_n; g_words[g_nwords].len=(uint32_t)len; g_words[g_nwords].freq=1;
            memcpy(g_arena+g_arena_n,buf+i,len); g_arena_n+=len; g_nwords++;
        } else { g_words[dedupe.val[s]].freq++; }
        i=j;
    }
    map_free(&dedupe);
}
// symbol sequences per word (ids), with freq. Train BPE up to target merges.
static void bpe_train(int target_merges){
    // sym arrays per word
    uint32_t** sym=(uint32_t**)malloc(g_nwords*sizeof(uint32_t*));
    int* slen=(int*)malloc(g_nwords*sizeof(int));
    for(size_t w=0;w<g_nwords;w++){ int L=g_words[w].len; sym[w]=(uint32_t*)malloc(L*sizeof(uint32_t));
        for(int t=0;t<L;t++) sym[w][t]=g_arena[g_words[w].off+t]; slen[w]=L; }
    g_nmerge=0; map_init(&g_mergemap,1ULL<<24);
    Map pc; // pair counts: key=a*8192+b, val=count (capped 32-bit)
    for(int it=0; it<target_merges; it++){
        map_init(&pc,1ULL<<22);
        for(size_t w=0;w<g_nwords;w++){ uint32_t* s=sym[w]; int L=slen[w]; uint32_t f=g_words[w].freq;
            for(int t=0;t+1<L;t++){ uint64_t k=(uint64_t)s[t]*8192ULL+s[t+1]; int nw; size_t sl=map_slot(&pc,k,&nw); pc.val[sl]+=f; } }
        // find max
        uint64_t bestk=0; uint32_t bestv=0;
        for(size_t s=0;s<pc.cap;s++) if(pc.key[s] && pc.val[s]>bestv){ bestv=pc.val[s]; bestk=pc.key[s]-1; /*folded +1*/ }
        // NOTE: map folds key|1 only when key==0; our keys >0 so unfold not needed unless k was 0
        if(pc.key){ /* recover true key: we stored k|1 only if k==0, impossible here */ }
        if(bestv==0){ map_free(&pc); break; }
        uint32_t a=(uint32_t)(bestk/8192ULL), b=(uint32_t)(bestk%8192ULL);
        uint32_t nid=(uint32_t)(256+g_nmerge);
        g_mA[g_nmerge]=a; g_mB[g_nmerge]=b;
        int nw; size_t ms=map_slot(&g_mergemap,(uint64_t)a*8192ULL+b,&nw); g_mergemap.val[ms]=nid; g_mergemap.aux[ms]=(uint32_t)g_nmerge;
        g_nmerge++;
        // apply merge a,b -> nid in all words
        for(size_t w=0;w<g_nwords;w++){ uint32_t* s=sym[w]; int L=slen[w]; if(L<2) continue;
            int o=0; for(int t=0;t<L;){ if(t+1<L && s[t]==a && s[t+1]==b){ s[o++]=nid; t+=2; } else { s[o++]=s[t]; t++; } }
            slen[w]=o; }
        map_free(&pc);
    }
    for(size_t w=0;w<g_nwords;w++) free(sym[w]); free(sym); free(slen);
}
// encode one chunk [a,b) at a given merge cutoff (only merges with rank<cutoff allowed)
static int bpe_encode_chunk(const unsigned char* buf, size_t a, size_t b, int cutoff, uint32_t* out){
    int L=(int)(b-a); if(L<=0) return 0;
    static uint32_t sbuf[8192]; uint32_t* s = (L<=8192)? sbuf : (uint32_t*)malloc(L*sizeof(uint32_t));
    for(int t=0;t<L;t++) s[t]=buf[a+t]; int n=L;
    for(;;){
        int bestrank=1<<30, bp=-1;
        for(int t=0;t+1<n;t++){ int nw; size_t ms=map_slot(&g_mergemap,(uint64_t)s[t]*8192ULL+s[t+1],&nw);
            if(!nw){ int rank=(int)g_mergemap.aux[ms]; if(rank<cutoff && rank<bestrank){ bestrank=rank; bp=t; } } }
        if(bp<0) break;
        uint32_t nid=(uint32_t)(256+bestrank);
        int o=0; for(int t=0;t<n;){ if(t==bp){ s[o++]=nid; t+=2; } else { s[o++]=s[t]; t++; } } n=o;
    }
    for(int t=0;t<n;t++) out[t]=s[t];
    if(s!=sbuf) free(s);
    return n;
}
static size_t tok_bpe(const unsigned char* buf, size_t a, size_t b, int cutoff, uint32_t* out, double* cover_bytes){
    size_t n=0; double cb=0; size_t i=a;
    while(i<b){ size_t j=i; int sp=issp(buf[i]); while(j<b && (issp(buf[j])?1:0)==sp) j++;
        uint32_t tmp[8192]; size_t need=j-i;
        if(need<=8192){ int m=bpe_encode_chunk(buf,i,j,cutoff,tmp); for(int t=0;t<m;t++) out[n++]=tmp[t]; }
        else { uint32_t* tb=(uint32_t*)malloc(need*sizeof(uint32_t)); int m=bpe_encode_chunk(buf,i,j,cutoff,tb); for(int t=0;t<m;t++) out[n++]=tb[t]; free(tb); }
        // count multi-byte coverage: a token covers >1 byte if id>=256
        i=j;
    }
    // coverage: recompute as bytes inside multi-byte tokens
    (void)cover_bytes; *cover_bytes=0; // computed by caller via bytes_per_unit instead
    return n;
}

// =============================================================================== bytepair (2-byte stride, lossless)
static size_t tok_bytepair(const unsigned char* buf, size_t a, size_t b, uint32_t* out){
    size_t n=0,i=a; while(i+1<b){ out[n++]=(uint32_t)((buf[i]<<8)|buf[i+1]); i+=2; }
    if(i<b){ out[n++]=(uint32_t)(65536+buf[i]); } return n;
}
// =============================================================================== hash3gram (3-byte stride, LOSSY)
static size_t tok_hash3(const unsigned char* buf, size_t a, size_t b, int cap, uint32_t* out){
    size_t n=0,i=a; while(i+2<b){ uint64_t h=fnv64(buf+i,3); out[n++]=(uint32_t)(h%(uint64_t)cap); i+=3; }
    while(i<b){ out[n++]=(uint32_t)(cap+buf[i]); i++; } return n;
}

// ----------------------------------------------------------------------------- driver
int main(int argc,char**argv){
    if(argc<2){ fprintf(stderr,"usage: %s <corpus> [--max-bytes N] [--word-topn N] [--save-bpe path --bpe-vocab N] [--bpe-only]\n",argv[0]); return 1; }
    const char* path=argv[1]; long maxb=0; int topn=8192; const char* savebpe=0; int bpevocab=1024, bpeonly=0;
    for(int i=2;i<argc;i++){ if(!strcmp(argv[i],"--max-bytes")&&i+1<argc) maxb=atol(argv[++i]);
        else if(!strcmp(argv[i],"--word-topn")&&i+1<argc) topn=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--save-bpe")&&i+1<argc) savebpe=argv[++i];
        else if(!strcmp(argv[i],"--bpe-vocab")&&i+1<argc) bpevocab=atoi(argv[++i]);
        else if(!strcmp(argv[i],"--bpe-only")) bpeonly=1; }
    FILE* f=fopen(path,"rb"); if(!f){ fprintf(stderr,"open fail %s\n",path); return 1; }
    fseek(f,0,SEEK_END); long fsz=ftell(f); fseek(f,0,SEEK_SET);
    if(maxb>0 && maxb<fsz) fsz=maxb;
    unsigned char* buf=(unsigned char*)malloc(fsz); if(fread(buf,1,fsz,f)!=(size_t)fsz){ fprintf(stderr,"read fail\n"); return 1; } fclose(f);
    size_t trn=(size_t)(fsz*0.90); size_t hen=(size_t)fsz; double held_bytes=(double)(hen-trn);
    fprintf(stderr,"corpus=%ld bytes  train=%zu  held=%zu  word-topn=%d\n",fsz,trn,hen-trn,topn);

    // --save-bpe: train BPE on the SAME train split and dump the merges (BPE is a nested prefix,
    // so the first (bpevocab-256) merges define the vocab). Deterministic -> reuse in 50.A
    // trainer+generator. Format: u32 magic 0x42504531 ('BPE1'), u32 vocab, u32 nmerge, then
    // nmerge*(u32 a, u32 b). --bpe-only skips the n-gram map (fast).
    if(savebpe){
        fprintf(stderr,"BPE save: collecting words...\n"); bpe_collect_words(buf,trn);
        int nm = bpevocab-256; if(nm<0) nm=0;
        fprintf(stderr,"BPE save: training to vocab %d (%d merges, nwords=%zu)...\n",bpevocab,nm,g_nwords);
        bpe_train(nm);
        FILE* bf=fopen(savebpe,"wb"); if(!bf){ fprintf(stderr,"save-bpe open fail %s\n",savebpe); return 1; }
        uint32_t bm=0x42504531u, bv=(uint32_t)bpevocab, bn=(uint32_t)g_nmerge;
        fwrite(&bm,4,1,bf); fwrite(&bv,4,1,bf); fwrite(&bn,4,1,bf);
        for(int r=0;r<g_nmerge;r++){ fwrite(&g_mA[r],4,1,bf); fwrite(&g_mB[r],4,1,bf); }
        fclose(bf);
        fprintf(stderr,"BPE save: wrote %d merges (vocab %d) to %s\n",g_nmerge,bpevocab,savebpe);
        if(bpeonly){ fprintf(stderr,"--bpe-only: done.\n"); return 0; }
    }

    // scratch (worst case: 1 unit per byte)
    uint32_t* tr=(uint32_t*)malloc(trn*sizeof(uint32_t));
    uint32_t* he=(uint32_t*)malloc((hen-trn)*sizeof(uint32_t));
    if(!tr||!he){ fprintf(stderr,"scratch OOM\n"); return 1; }

    Row rows[16]; int nr=0;
    #define ADD(NM,VOC,NTOK,COVER,LOSS) do{ Row* R=&rows[nr++]; R->name=NM; R->V=(VOC); R->ntok=(NTOK); \
        R->bytes_per_unit=held_bytes/(double)(NTOK); R->repeat_mass=repeat_mass(he,(NTOK)); R->cover=(COVER); R->lossless=(LOSS); \
        eval_ngram(tr,ntr,he,(NTOK),(VOC),held_bytes,&R->bpb1,&R->bpb2,&R->bpu2,&R->p90); }while(0)

    size_t ntr, nhe; double cb;

    // --- byte
    ntr=tok_byte(buf,0,trn,tr); nhe=tok_byte(buf,trn,hen,he);
    ADD("byte",256,nhe,0.0,1);

    // --- word
    fprintf(stderr,"building word vocab...\n");
    word_build(buf,trn,topn);
    ntr=tok_word(buf,0,trn,tr,&cb); nhe=tok_word(buf,trn,hen,he,&cb);
    ADD("word",g_word_nvocab,nhe,cb/held_bytes,1);

    // --- BPE (train once to 4096, eval at 512/1024/4096 cutoffs)
    fprintf(stderr,"collecting BPE words...\n"); bpe_collect_words(buf,trn);
    fprintf(stderr,"training BPE to 4096 (nwords=%zu)...\n",g_nwords); bpe_train(3840);
    fprintf(stderr,"BPE trained: %d merges\n",g_nmerge);
    // token byte-lengths: toklen[id<256]=1; toklen[256+r]=toklen[a]+toklen[b]
    int* toklen=(int*)malloc((size_t)(256+g_nmerge)*sizeof(int));
    for(int t=0;t<256;t++) toklen[t]=1;
    for(int r=0;r<g_nmerge;r++) toklen[256+r]=toklen[g_mA[r]]+toklen[g_mB[r]];
    struct { const char* nm; int cut; } bpecfg[3]={{"bpe512",256},{"bpe1024",768},{"bpe4096",3840}};
    for(int c=0;c<3;c++){ int cut=bpecfg[c].cut; if(cut>g_nmerge)cut=g_nmerge;
        ntr=tok_bpe(buf,0,trn,cut,tr,&cb); nhe=tok_bpe(buf,trn,hen,cut,he,&cb);
        double mb=0; for(size_t t=0;t<nhe;t++){ int L=(he[t]<256)?1:toklen[he[t]]; if(L>1) mb+=L; }
        ADD(bpecfg[c].nm,256+cut,nhe,mb/held_bytes,1);
    }
    free(toklen);

    // --- bytepair
    ntr=tok_bytepair(buf,0,trn,tr); nhe=tok_bytepair(buf,trn,hen,he);
    ADD("bytepair",65536+256,nhe,1.0,1);

    // --- hash3gram (LOSSY)
    int H3CAP=65536;
    ntr=tok_hash3(buf,0,trn,H3CAP,tr); nhe=tok_hash3(buf,trn,hen,H3CAP,he);
    ADD("hash3gram",H3CAP+256,nhe,1.0,0);

    // ----------------------------------------------------------------- print table
    printf("\n");
    printf("== PHASE 50.0 UNIT-CHOICE MAP (train=90%% held=10%%, KN-lite n-gram, bits per BYTE) ==\n");
    printf("%-11s %8s %9s %9s %8s %8s %8s %7s %7s %6s\n",
           "unit","vocab","bpb-o1","bpb-o2","b/unit","repeat","bpu-o2","p90","cover%","loss");
    for(int i=0;i<nr;i++){ Row* R=&rows[i];
        printf("%-11s %8d %9.4f %9.4f %8.3f %8.4f %8.3f %7.2f %7.1f %6s\n",
            R->name,R->V,R->bpb1,R->bpb2,R->bytes_per_unit,R->repeat_mass,R->bpu2,R->p90,
            R->cover*100.0, R->lossless?"yes":"LOSSY");
    }
    printf("\nread: bpb-o2 = headline (unit-invariant, lower=better, byte is baseline); repeat = P(next==cur)\n");
    printf("      flood propensity (byte high, word/BPE should collapse); b/unit = sequence shortening;\n");
    printf("      LOSSY rows (hash3gram) understate bpb (info discarded) -> not directly promotable. ADVISORY.\n");
    return 0;
}
