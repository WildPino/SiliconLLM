// Phase 55 helper - tokenize the corpus with BPE-1024 and dump token ids + byte-length table.
// So the PyTorch baseline can train/eval on the EXACT same tokenization (no Python re-encode).
//
// Output:
//   <out_ids>  : raw uint16 token-id stream (whole corpus, in order)
//   <out_meta> : magic 0x54444D50, uint32 VTOK, uint32 ntok, then uint8 exp_len[VTOK] (bytes per token)
//
// Build: gcc -O3 -march=native benchmarks/phase55/phase55_tokdump.c -o bin/phase55_tokdump.exe
// Run:   bin/phase55_tokdump.exe <corpus> <bpe1024.bin> <out_ids> <out_meta> [--max-bytes N]
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include "benchmarks/phase50/bpe_codec.h"

int main(int argc,char** argv){
    if(argc<5){ fprintf(stderr,"usage: %s <corpus> <bpe> <out_ids> <out_meta> [--max-bytes N]\n",argv[0]); return 1; }
    long maxb=0; for(int i=5;i<argc;i++){ if(!strcmp(argv[i],"--max-bytes")&&i+1<argc) maxb=atol(argv[++i]); }
    FILE* fd=fopen(argv[1],"rb"); if(!fd){ fprintf(stderr,"open corpus\n"); return 1; }
    fseek(fd,0,SEEK_END); long fsz=ftell(fd); fseek(fd,0,SEEK_SET);
    if(maxb>0 && maxb<fsz) fsz=maxb;
    unsigned char* buf=malloc(fsz); fread(buf,1,fsz,fd); fclose(fd);
    Bpe B; if(!bpe_load_file(&B,argv[2])) return 1;
    fprintf(stderr,"bpe vocab=%d nmerge=%d; tokenizing %ld bytes...\n",B.vocab,B.nmerge,fsz);
    uint32_t* ids=malloc((size_t)fsz*sizeof(uint32_t));
    size_t n=bpe_encode_region(&B,buf,0,(size_t)fsz,ids);
    fprintf(stderr,"ntok=%zu  bytes/tok=%.3f\n",n,(double)fsz/n);
    // write ids as uint16 (VTOK<=4096 fits)
    FILE* fi=fopen(argv[3],"wb"); uint16_t* u16=malloc(n*2);
    for(size_t i=0;i<n;i++) u16[i]=(uint16_t)ids[i];
    fwrite(u16,2,n,fi); fclose(fi);
    // meta
    FILE* fmF=fopen(argv[4],"wb"); uint32_t mg=0x54444D50,V=(uint32_t)B.vocab,nt=(uint32_t)n;
    fwrite(&mg,4,1,fmF); fwrite(&V,4,1,fmF); fwrite(&nt,4,1,fmF);
    uint8_t* el=malloc(B.vocab); for(int t=0;t<B.vocab;t++) el[t]=(uint8_t)bpe_tok_len(&B,(uint32_t)t);
    fwrite(el,1,B.vocab,fmF);
    // byte expansions per token (concatenated; lengths = el[]) so Python can decode tokens->text
    for(int t=0;t<B.vocab;t++){ const unsigned char* bb=bpe_tok_bytes(&B,(uint32_t)t); fwrite(bb,1,el[t],fmF); }
    fclose(fmF);
    fprintf(stderr,"wrote %s (%zu u16) + %s (VTOK=%d)\n",argv[3],n,argv[4],B.vocab);
    return 0;
}
