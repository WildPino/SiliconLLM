/* Independent integer arithmetic. Does NOT evaluate source/candidate FFNs. */
#include <stdint.h>
#include <string.h>
#include <fenv.h>
#include <float.h>
#include <math.h>
#define API __declspec(dllexport)
#define L 20

static void add(uint32_t *out, const uint32_t *term, int n) {
    uint64_t carry=0;
    for(int j=0;j<L;j++) {
        uint64_t v=(uint64_t)out[j]+(j<n?term[j]:0)+carry;
        out[j]=(uint32_t)v; carry=v>>32;
    }
}

static int decode(uint32_t bits, uint32_t *a) {
    memset(a,0,9*sizeof(uint32_t));
    unsigned exponent=(bits>>23)&255, mantissa=bits&0x7fffff;
    if(exponent==255) return -1;
    if(exponent)mantissa|=0x800000;
    unsigned shift=exponent?exponent-1:0, at=shift/32, off=shift%32;
    uint64_t v=(uint64_t)mantissa<<off;
    a[at]=(uint32_t)v; if(at+1<9)a[at+1]=(uint32_t)(v>>32);
    return 0;
}

static void difference(uint32_t x, uint32_t y, uint32_t *a, const uint32_t *b) {
    if((x^y)>>31) {
        uint64_t carry=0;
        for(int j=0;j<9;j++){uint64_t v=(uint64_t)a[j]+b[j]+carry; a[j]=(uint32_t)v; carry=v>>32;}
    } else {
        int swap=0;
        for(int j=8;j>=0;j--)if(a[j]!=b[j]){swap=a[j]<b[j];break;}
        uint64_t borrow=0;
        for(int j=0;j<9;j++) {
            uint64_t left=swap?b[j]:a[j], right=(uint64_t)(swap?a[j]:b[j])+borrow;
            uint32_t v=(uint32_t)(left-right);borrow=left<right;a[j]=v;
        }
    }
}

static void square(const uint32_t *a, uint32_t *out) {
    memset(out,0,L*sizeof(uint32_t));
    int n=9;while(n>0 && !a[n-1])n--;
    for(int i=0;i<n;i++) {
        uint64_t carry=0;
        for(int j=0;j<n;j++) {
            uint64_t v=(uint64_t)a[i]*a[j]+out[i+j]+carry;
            out[i+j]=(uint32_t)v;carry=v>>32;
        }
        int at=i+n;
        while(carry){uint64_t v=(uint64_t)out[at]+carry;out[at++]=(uint32_t)v;carry=v>>32;}
    }
}

API int prefix_energy(int rows,int width,const uint32_t *truth,const uint32_t *guess,uint64_t *out) {
    if(rows<1 || rows>17540 || width<1 || width>768 || !truth || !out)return -1;
    for(int i=0;i<rows;i++) {
        uint32_t total[L]={0},a[9],b[9],sq[L];
        for(int d=0;d<width;d++) {
            size_t at=(size_t)i*width+d;
            if(decode(truth[at],a))return -2;
            if(guess){if(decode(guess[at],b))return -2;difference(truth[at],guess[at],a,b);}
            square(a,sq);add(total,sq,L);
        }
        for(int j=0;j<10;j++)out[(size_t)i*10+j]=(uint64_t)total[2*j]|((uint64_t)total[2*j+1]<<32);
    }
    return 0;
}

API int prefix_weighted(int rows,int width,const float *x,const float *mass,float *out) {
    if(rows<1 || rows>17540 || width<1 || width>768 || fesetround(FE_TONEAREST) || FLT_RADIX!=2 || sizeof(float)!=4 || sizeof(double)!=8 || FLT_EVAL_METHOD!=0)return -1;
    for(int i=0;i<rows;i++)for(int d=0;d<width;d++) {
        size_t at=(size_t)i*width+d;
        if(!isfinite(x[at]) || !isfinite(mass[i]))return -2;
        volatile double value=(double)x[at]*(double)mass[i];out[at]=(float)value;
        if(!isfinite(out[at]))return -3;
    }
    return 0;
}
