/* Original phase60 exact scan body, supplied with real Falcon mapped operands.
   Standalone operator qualification; no full-model or speed admission. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <io.h>
#include <sys/stat.h>
#define DN 768
#define N 64
#define OMP_PFOR
typedef struct { float *A, *Dskip; } SSML;
static inline float silu(float x){ return x/(1.0f+expf(-x)); }

static void original_scan(SSML* s,float (*hl)[N],const float* dt,const float* xx,
                          const float* Bm,const float* Cm,const float* z,float* y){
    OMP_PFOR for(int c=0;c<DN;c++){ const float* Ac=s->A+(size_t)c*N; float* hc=hl[c]; float dtc=dt[c],xc=xx[c],acc=0;
        for(int j=0;j<N;j++){ hc[j]=expf(dtc*Ac[j])*hc[j]+dtc*Bm[j]*xc; acc+=hc[j]*Cm[j]; } y[c]=acc+s->Dskip[c]*xc; }
    for(int c=0;c<DN;c++) y[c]*=silu(z[c]);
}

static void read_floats(FILE* f,float* p,size_t count){
    if(fread(p,sizeof(float),count,f)!=count){fputs("short input\n",stderr);exit(2);}
    for(size_t i=0;i<count;i++) if(!isfinite(p[i])){fputs("nonfinite input\n",stderr);exit(3);}
}
int main(int argc,char** argv){
    if(argc!=3)return 1;
    FILE* in=fopen(argv[1],"rb"); if(!in)return 2;
    uint32_t header[4];
    if(fread(header,sizeof(uint32_t),4,in)!=4 || header[0]!=0x53434e31 || header[1]!=DN || header[2]!=N || header[3]<1 || header[3]>256)return 3;
    size_t T=header[3];
    float* A=malloc(DN*N*sizeof(float));float* Dskip=malloc(DN*sizeof(float));
    float* state=malloc(DN*N*sizeof(float));float* ys=malloc(T*DN*sizeof(float));
    if(!A||!Dskip||!state||!ys)return 4;
    read_floats(in,A,DN*N);read_floats(in,Dskip,DN);read_floats(in,state,DN*N);
    SSML s={A,Dskip};float dt[DN],xx[DN],B[N],C[N],z[DN];
    for(size_t t=0;t<T;t++){
        read_floats(in,dt,DN);read_floats(in,xx,DN);read_floats(in,B,N);read_floats(in,C,N);read_floats(in,z,DN);
        original_scan(&s,(float (*)[N])state,dt,xx,B,C,z,ys+t*DN);
    }
    if(fgetc(in)!=EOF)return 5;fclose(in);
    int fd=_open(argv[2],_O_BINARY|_O_WRONLY|_O_CREAT|_O_EXCL,_S_IREAD|_S_IWRITE);if(fd<0)return 6;
    FILE* out=_fdopen(fd,"wb");if(!out)return 6;
    if(fwrite(state,sizeof(float),DN*N,out)!=DN*N || fwrite(ys,sizeof(float),T*DN,out)!=T*DN)return 7;
    if(fclose(out))return 8;
    printf("original_exact_scan DN=%d N=%d T=%zu\n",DN,N,T);
    free(A);free(Dskip);free(state);free(ys);return 0;
}
