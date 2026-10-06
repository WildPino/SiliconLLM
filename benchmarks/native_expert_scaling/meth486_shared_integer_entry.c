#include "meth486_shared_integer_generate_entry.c"

static void g486_write(FILE *f,const void *data,size_t count,size_t width){require(fwrite(data,width,count,f)==count&&fflush(f)==0,"control_raw_write");}
static int g486_controls(const char *path){
    const int shapes[11][4]={{1,1,1,0},{17,15,1,1},{8,16,1,2},{9,17,1,3},{8,4096,1,4},{768,768,1,0},{768,768,29,1},{768,768,256,3},{3072,768,1,4},{768,3072,1,2},{32128,768,1,3}};
    FILE *f=fopen(path,"wb");require(f!=NULL,"control_output");g486_write(f,"M486CTL1",8,1);uint32_t count=11;g486_write(f,&count,1,4);
    omp_set_num_threads(1);omp_set_dynamic(0);worker_setup(1);LARGE_INTEGER frequency;require(QueryPerformanceFrequency(&frequency),"control_clock");cost_frequency=(double)frequency.QuadPart;
    for(int ci=0;ci<11;ci++){
        int rows=shapes[ci][0],cols=shapes[ci][1],queries=shapes[ci][2],pattern=shapes[ci][3];
        int8_t *weights=alloc((size_t)rows*cols,1);float *scales=alloc(rows,4),*x=alloc((size_t)queries*cols,4),*y=alloc((size_t)queries*rows,4),*ref=alloc((size_t)queries*rows,4);
        int16_t *codes=alloc((size_t)queries*cols,2);float *alpha=alloc(queries,4);int64_t *reconstructed=alloc((size_t)queries*rows,8);
        const float scaleset[4]={.001f,1.f,1000.f,.000001f};
        for(int r=0;r<rows;r++){scales[r]=scaleset[r%4];for(int j=0;j<cols;j++){int code=pattern==4?-128:((r*17+j*31+ci*13)%256)-128;weights[(size_t)r*cols+j]=(int8_t)code;}}
        for(int t=0;t<queries;t++)for(int j=0;j<cols;j++){
            float value=0.f;if(pattern==1)value=(j+t)%2?-32767.f:32767.f;
            if(pattern==2)value=j==0?32767.f:(float)(((j+t)%31)-15)+.5f;
            if(pattern==3)value=(float)(((j*23+t*11)%1009)-504)*.03125f;
            if(pattern==4)value=32767.f;x[(size_t)t*cols+j]=value;
        }
        for(int t=0;t<queries;t++)alpha[t]=head_activation_codes(x+(size_t)t*cols,codes+(size_t)t*cols,cols);
        uint32_t h[7]={(uint32_t)rows,(uint32_t)cols,(uint32_t)queries,(uint32_t)g486_pad(rows),(uint32_t)g486_pad(cols),(uint32_t)pattern,0};
        g486_write(f,h,7,4);g486_write(f,weights,(size_t)rows*cols,1);g486_write(f,scales,rows,4);g486_write(f,x,(size_t)queries*cols,4);g486_write(f,codes,(size_t)queries*cols,2);g486_write(f,alpha,queries,4);
        Matrix w={NULL,scales,weights,(uint32_t)rows,(uint32_t)cols};double started=cost_clock();g486_runtime();g486_add(w,queries);g486_allocate();g486_setup_seconds=cost_clock()-started;g486_reset();
        require(g486_eval(w,x,y,rows,cols,queries),"control_GPU_dispatch");g486_write(f,g486_hout,(size_t)h[3]*4*queries,4);
        int mismatch=0;for(int t=0;t<queries;t++)for(int r=0;r<rows;r++){
            size_t at=(size_t)t*4*h[3]+r;int64_t fromgpu=(int64_t)g486_hout[at]+128LL*g486_hout[at+h[3]]+16384LL*g486_hout[at+2*h[3]],scalar=0;
            for(int j=0;j<cols;j++)scalar+=(int64_t)weights[(size_t)r*cols+j]*codes[(size_t)t*cols+j];
            reconstructed[(size_t)t*rows+r]=fromgpu;ref[(size_t)t*rows+r]=(float)(((double)scalar*(double)scales[r])*(double)alpha[t]);
            if(fromgpu!=scalar||head_integer_dot(weights+(size_t)r*cols,codes+(size_t)t*cols,cols)!=scalar)mismatch++;
        }
        g486_write(f,reconstructed,(size_t)queries*rows,8);g486_write(f,y,(size_t)queries*rows,4);g486_write(f,ref,(size_t)queries*rows,4);
        printf("{\"control\":%d,\"rows\":%d,\"cols\":%d,\"queries\":%d,\"integer_mismatches\":%d,\"output_byte_exact\":%s",ci,rows,cols,queries,mismatch,memcmp(y,ref,(size_t)queries*rows*4)?"false":"true");g486_json();printf("}\n");fflush(stdout);
        require(!mismatch&&!memcmp(y,ref,(size_t)queries*rows*4),"control_exact_integer_and_output");g486_teardown();
        free(weights);free(scales);free(x);free(y);free(ref);free(codes);free(alpha);free(reconstructed);
    }
    require(fclose(f)==0,"control_close");return 0;
}
int main(int argc,char **argv){
    require(fegetround()==FE_TONEAREST,"backend_rounding");
    if(argc==3&&!strcmp(argv[1],"--controls")){g486_enabled=1;return g486_controls(argv[2]);}
    require(argc>=4&&!strcmp(argv[1],"--backend")&&(!strcmp(argv[2],"0")||!strcmp(argv[2],"1")),"backend 0_or_1 original_generation_or_cost_arguments");
    g486_enabled=atoi(argv[2]);return meth486_generate_entry(argc-2,argv+2);
}
