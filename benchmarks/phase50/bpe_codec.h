// Shared BPE-1024 codec for Phase 50.A (trainer + generator). Header-only.
//
// Loads the merge list saved by phase50_0_map (--save-bpe, magic 0x42504531), rebuilds the
// rank map and per-token byte expansions, and provides deterministic encode (byte region ->
// token ids, space/non-space chunked, greedy lowest-rank merge) and decode (token id -> bytes).
// BYTE-level / lossless: every token id<256 is a single byte; id>=256 is a learned merge.
//
// The merges are embedded in the 50.A checkpoint so the generator reconstructs the tokenizer
// from the checkpoint alone (no external file at generation time).

#ifndef BPE_CODEC_H
#define BPE_CODEC_H
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

#define BPE_MAXVOCAB 4096

typedef struct {
    int vocab;            // 256 + nmerge
    int nmerge;
    uint32_t mA[BPE_MAXVOCAB], mB[BPE_MAXVOCAB];   // merge pairs (rank order)
    // rank map: open-addressed, key = a*8192+b + 1
    uint64_t* rk_key; int* rk_rank; size_t rk_cap, rk_mask;
    // token byte expansions
    unsigned char* exp_buf;   // concatenated bytes
    int exp_off[BPE_MAXVOCAB+256];
    int exp_len[BPE_MAXVOCAB+256];
} Bpe;

static int bpe_isspace(unsigned char c){ return c==' '||c=='\t'||c=='\n'||c=='\r'; }

static void bpe_build_internal(Bpe* B){
    // rank map
    size_t cap=1; while(cap < (size_t)(B->nmerge*2+16)) cap<<=1; if(cap<16) cap=16;
    B->rk_cap=cap; B->rk_mask=cap-1;
    B->rk_key=(uint64_t*)calloc(cap,sizeof(uint64_t));
    B->rk_rank=(int*)malloc(cap*sizeof(int));
    for(int r=0;r<B->nmerge;r++){
        uint64_t k=(uint64_t)B->mA[r]*8192ULL+B->mB[r]+1ULL;
        size_t i=(size_t)(k*11400714819323198485ULL)&B->rk_mask;
        while(B->rk_key[i]){ if(B->rk_key[i]==k) break; i=(i+1)&B->rk_mask; }
        if(!B->rk_key[i]){ B->rk_key[i]=k; B->rk_rank[i]=r; }
    }
    // expansions: id<256 -> single byte; id 256+r -> exp(a)+exp(b)
    int V=B->vocab; size_t bufcap=1<<20; B->exp_buf=(unsigned char*)malloc(bufcap); size_t bn=0;
    for(int t=0;t<256;t++){ B->exp_off[t]=(int)bn; B->exp_len[t]=1; if(bn+1>bufcap){bufcap*=2;B->exp_buf=(unsigned char*)realloc(B->exp_buf,bufcap);} B->exp_buf[bn++]=(unsigned char)t; }
    for(int r=0;r<B->nmerge;r++){ int id=256+r; uint32_t a=B->mA[r],b=B->mB[r];
        int la=B->exp_len[a], lb=B->exp_len[b];
        if(bn+la+lb>bufcap){ bufcap*=2; B->exp_buf=(unsigned char*)realloc(B->exp_buf,bufcap); }
        B->exp_off[id]=(int)bn;
        memcpy(B->exp_buf+bn,B->exp_buf+B->exp_off[a],la); bn+=la;
        memcpy(B->exp_buf+bn,B->exp_buf+B->exp_off[b],lb); bn+=lb;
        B->exp_len[id]=la+lb;
    }
    (void)V;
}

// load from a merges file (phase50_0 --save-bpe)
static int bpe_load_file(Bpe* B, const char* path){
    FILE* f=fopen(path,"rb"); if(!f){ fprintf(stderr,"bpe open fail %s\n",path); return 0; }
    uint32_t m,v,n; if(fread(&m,4,1,f)!=1||m!=0x42504531u){ fprintf(stderr,"bpe bad magic\n"); fclose(f); return 0; }
    fread(&v,4,1,f); fread(&n,4,1,f);
    B->vocab=(int)v; B->nmerge=(int)n;
    if(B->nmerge>BPE_MAXVOCAB){ fprintf(stderr,"bpe nmerge too big\n"); fclose(f); return 0; }
    for(int r=0;r<B->nmerge;r++){ fread(&B->mA[r],4,1,f); fread(&B->mB[r],4,1,f); }
    fclose(f);
    bpe_build_internal(B);
    return 1;
}
// load merges directly from arrays (generator: read from checkpoint)
static void bpe_load_arrays(Bpe* B, int vocab, int nmerge, const uint32_t* mA, const uint32_t* mB){
    B->vocab=vocab; B->nmerge=nmerge;
    memcpy(B->mA,mA,(size_t)nmerge*4); memcpy(B->mB,mB,(size_t)nmerge*4);
    bpe_build_internal(B);
}
static inline int bpe_rank(const Bpe* B, uint32_t a, uint32_t b){
    uint64_t k=(uint64_t)a*8192ULL+b+1ULL;
    size_t i=(size_t)(k*11400714819323198485ULL)&B->rk_mask;
    while(B->rk_key[i]){ if(B->rk_key[i]==k) return B->rk_rank[i]; i=(i+1)&B->rk_mask; }
    return -1;
}
// encode one chunk [a,b) (no internal space/non-space mixing) greedily; out must hold >= (b-a)
static int bpe_encode_chunk(const Bpe* B, const unsigned char* buf, size_t a, size_t b, uint32_t* out){
    int L=(int)(b-a); if(L<=0) return 0;
    static __thread uint32_t sb[8192]; uint32_t* s=(L<=8192)?sb:(uint32_t*)malloc(L*sizeof(uint32_t));
    for(int t=0;t<L;t++) s[t]=buf[a+t]; int n=L;
    for(;;){ int best=1<<30, bp=-1;
        for(int t=0;t+1<n;t++){ int r=bpe_rank(B,s[t],s[t+1]); if(r>=0 && r<best){ best=r; bp=t; } }
        if(bp<0) break;
        uint32_t nid=(uint32_t)(256+best);
        int o=0; for(int t=0;t<n;){ if(t==bp){ s[o++]=nid; t+=2; } else { s[o++]=s[t]; t++; } } n=o;
    }
    for(int t=0;t<n;t++) out[t]=s[t];
    if(s!=sb) free(s);
    return n;
}
// encode a byte region [a,b) into token ids (space/non-space chunked). out must hold >= (b-a).
static size_t bpe_encode_region(const Bpe* B, const unsigned char* buf, size_t a, size_t b, uint32_t* out){
    size_t n=0,i=a;
    while(i<b){ size_t j=i; int sp=bpe_isspace(buf[i]); while(j<b && (bpe_isspace(buf[j])?1:0)==sp) j++;
        size_t need=j-i; uint32_t* tmp=(need<=8192)?out+n:(uint32_t*)malloc(need*sizeof(uint32_t));
        int m=bpe_encode_chunk(B,buf,i,j, (need<=8192)?out+n:tmp);
        if(need<=8192){ n+=m; } else { for(int t=0;t<m;t++) out[n++]=tmp[t]; free(tmp); }
        i=j;
    }
    return n;
}
static inline int bpe_tok_len(const Bpe* B, uint32_t id){ return B->exp_len[id]; }
static inline const unsigned char* bpe_tok_bytes(const Bpe* B, uint32_t id){ return B->exp_buf+B->exp_off[id]; }
#endif
