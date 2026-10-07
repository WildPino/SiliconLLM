/* METH515: literal365 missing encoder batches, qualified374 physical workers. */
#define encode meth515_original_encode
#include "meth374_switch_physical_workers.c"
#undef encode

static void integer_dot4(const int8_t *a,const int16_t *b,int stride,int n,int64_t sums[4]){
    require(n>0&&n<=4096&&stride>=n,"batch_integer_bound");
    __m256i a0=_mm256_setzero_si256(),a1=a0,a2=a0,a3=a0;int i=0;
    for(;i+16<=n;i+=16){
        __m256i w=_mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i *)(a+i)));
        a0=_mm256_add_epi32(a0,_mm256_madd_epi16(w,_mm256_loadu_si256((const __m256i *)(b+i))));
        a1=_mm256_add_epi32(a1,_mm256_madd_epi16(w,_mm256_loadu_si256((const __m256i *)(b+stride+i))));
        a2=_mm256_add_epi32(a2,_mm256_madd_epi16(w,_mm256_loadu_si256((const __m256i *)(b+2*stride+i))));
        a3=_mm256_add_epi32(a3,_mm256_madd_epi16(w,_mm256_loadu_si256((const __m256i *)(b+3*stride+i))));
    }
    int32_t lanes[4][8];_mm256_storeu_si256((__m256i *)lanes[0],a0);_mm256_storeu_si256((__m256i *)lanes[1],a1);
    _mm256_storeu_si256((__m256i *)lanes[2],a2);_mm256_storeu_si256((__m256i *)lanes[3],a3);
    for(int t=0;t<4;t++){int64_t sum=0;for(int j=0;j<8;j++)sum+=(int64_t)lanes[t][j];
        for(int j=i;j<n;j++)sum+=(int64_t)a[j]*(int64_t)b[t*stride+j];sums[t]=sum;}
}

static void encoder_reuse_mv_batch(Matrix w,const float *x,float *y,int rows,int cols,int tokens){
    require((w.q||w.f)&&w.rows==(uint32_t)rows&&w.cols==(uint32_t)cols&&tokens>0&&tokens<=CTX,"batch_matrix_contract");
    double cost_start=cost_enabled&&cost_profile?cost_clock():0.;
    if(cost_enabled){CostCounter *counter=&cost_counts[cost_phase][cost_kind];counter->calls+=tokens;
        if(w.q){counter->codes+=(uint64_t)rows*cols*tokens;counter->scales+=(uint64_t)rows*4*tokens;}
        else counter->f32+=(uint64_t)rows*cols*4*tokens;}
    if(w.q){
        int16_t *codes=alloc((size_t)tokens*cols,2);float scales[CTX];
        for(int token=0;token<tokens;token++)scales[token]=head_activation_codes(x+(size_t)token*cols,codes+(size_t)token*cols,cols);
        #pragma omp parallel for schedule(static) if(rows>64)
        for(int row=0;row<rows;row++){
            const int8_t *wrow=w.q+(size_t)row*cols;int token=0;
            for(;token+4<=tokens;token+=4){int64_t sums[4];integer_dot4(wrow,codes+(size_t)token*cols,cols,cols,sums);
                for(int t=0;t<4;t++)y[(size_t)(token+t)*rows+row]=(float)(((double)sums[t]*(double)w.scales[row])*(double)scales[token+t]);}
            for(;token<tokens;token++){int64_t sum=head_integer_dot(wrow,codes+(size_t)token*cols,cols);
                y[(size_t)token*rows+row]=(float)(((double)sum*(double)w.scales[row])*(double)scales[token]);}
        }
        free(codes);
    }else{
        #pragma omp parallel for schedule(static) if(rows>64)
        for(int row=0;row<rows;row++)for(int token=0;token<tokens;token++)
            y[(size_t)token*rows+row]=(float)dot64(w.f+(size_t)row*cols,x+(size_t)token*cols,cols);
    }
    if(cost_enabled&&cost_profile)cost_counts[cost_phase][cost_kind].matrix_seconds+=cost_clock()-cost_start;
}

static void encoder_reuse_feed(Config c,Block *b,float *hidden,int tokens){
    float *x=alloc((size_t)tokens*c.d,4),*up=alloc((size_t)tokens*c.ff,4),*down=alloc((size_t)tokens*c.d,4);
    for(int token=0;token<tokens;token++)norm(hidden+(size_t)token*c.d,b->ffnorm,x+(size_t)token*c.d,c.d,c.epsilon);
    if(!b->sparse){
        cost_kind=0;encoder_reuse_mv_batch(b->wi,x,up,c.ff,c.d,tokens);
        for(size_t j=0;j<(size_t)tokens*c.ff;j++)if(up[j]<0.f)up[j]=0.f;
        encoder_reuse_mv_batch(b->wo,up,down,c.d,c.ff,tokens);
        for(size_t j=0;j<(size_t)tokens*c.d;j++)hidden[j]+=1.f*down[j];
    }else{
        int *counts=alloc(c.n,sizeof(int));float *logits=alloc((size_t)tokens*c.n,4);
        cost_kind=2;encoder_reuse_mv_batch(b->router,x,logits,c.n,c.d,tokens);
        for(int token=0;token<tokens;token++){
            float *scores=logits+(size_t)token*c.n;int chosen=0;
            for(int e=1;e<c.n;e++)if(scores[e]>scores[chosen])chosen=e;
            double sum=0.;for(int e=0;e<c.n;e++)sum+=(double)(float)exp((double)(scores[e]-scores[chosen]));
            float probability=(float)(1./sum);counts[chosen]++;int accepted=counts[chosen]<=c.capacity||fault==4;
            require(route_count<route_bound,"route_buffer");routes[route_count++]=(Route){chosen,accepted,probability};
            if(fault==3)probability=1.f;
            if(accepted){cost_kind=1;mv(b->expertwi[chosen],x+(size_t)token*c.d,up,c.ff,c.d);
                for(int j=0;j<c.ff;j++)if(up[j]<0.f)up[j]=0.f;
                mv(b->expertwo[chosen],up,down,c.d,c.ff);
                for(int j=0;j<c.d;j++)hidden[(size_t)token*c.d+j]+=probability*down[j];}
        }
        free(logits);free(counts);
    }
    free(x);free(up);free(down);cost_kind=0;
}

static float *encode(Model *m,const int *ids,int tokens,float *snap){
    Config c=m->c;size_t width=(size_t)tokens*c.d;float *h=alloc(width,sizeof(float)),*x=alloc(width,sizeof(float)),*q=alloc(width,sizeof(float)),*k=alloc(width,sizeof(float)),*v=alloc(width,sizeof(float)),*att=alloc(width,sizeof(float)),*out=alloc(width,sizeof(float));
    for(int t=0;t<tokens;t++)memcpy(h+(size_t)t*c.d,m->embed+(size_t)ids[t]*c.d,c.d*4);memcpy(snap,h,width*4);
    for(int l=0;l<c.enc;l++){Block *b=m->enc+l;for(int t=0;t<tokens;t++){size_t off=(size_t)t*c.d;norm(h+off,b->selfnorm,x+off,c.d,c.epsilon);}mv_batch(b->self.q,x,q,c.d,c.d,tokens);mv_batch(b->self.k,x,k,c.d,c.d,tokens);mv_batch(b->self.v,x,v,c.d,c.d,tokens);
        for(int t=0;t<tokens;t++){size_t off=(size_t)t*c.d;attend(c,q+off,k,v,tokens,t,1,0,m->encbias,att+off);}
        encoder_reuse_mv_batch(b->self.o,att,out,c.d,c.d,tokens);for(size_t j=0;j<width;j++)h[j]+=out[j];
        encoder_reuse_feed(c,b,h,tokens);memcpy(snap+(size_t)(l+1)*width,h,width*4);
    }for(int t=0;t<tokens;t++)norm(h+(size_t)t*c.d,m->encnorm,x+(size_t)t*c.d,c.d,c.epsilon);memcpy(snap+(size_t)(c.enc+1)*width,x,width*4);memcpy(h,x,width*4);
    free(x);free(q);free(k);free(v);free(att);free(out);return h;
}
