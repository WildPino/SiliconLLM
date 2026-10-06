/* METH485: one fixed CUDA integer layout; minimal Win64 ABI, no CUDA SDK dependency. */
_Static_assert(sizeof(void *)==8 && sizeof(int)==4 && sizeof(size_t)==8,"Win64 CUDA ABI");
_Static_assert(sizeof(float)==4 && sizeof(int16_t)==2 && sizeof(int32_t)==4,"wire ABI");
typedef int (__stdcall *G485Malloc)(void **,size_t);
typedef int (__stdcall *G485Free)(void *);
typedef int (__stdcall *G485HostAlloc)(void **,size_t,unsigned int);
typedef int (__stdcall *G485Memcpy)(void *,const void *,size_t,int);
typedef int (__stdcall *G485IntOut)(int *);
typedef int (__stdcall *G485IntIn)(int);
typedef int (__stdcall *G485Void)(void);
typedef int (__stdcall *G485MemInfo)(size_t *,size_t *);
typedef int (__stdcall *G485CuInit)(unsigned int);
typedef int (__stdcall *G485Name)(char *,int,int);
typedef int (__stdcall *G485Attr)(int *,int,int);
typedef int (__stdcall *G485Create)(void **);
typedef int (__stdcall *G485Handle)(void *);
typedef int (__stdcall *G485HandleOut)(void *,int *);
typedef int (__stdcall *G485HandleIn)(void *,int);
typedef int (__stdcall *G485StreamIn)(void *,void *);
typedef int (__stdcall *G485StreamOut)(void *,void **);
typedef int (__stdcall *G485Workspace)(void *,void *,size_t);
typedef int (__stdcall *G485Gemm)(void *,int,int,int,int,int,const void *,const void *,int,int,const void *,int,int,const void *,void *,int,int,int,int);
typedef struct {Matrix w;int mp,dp,qmax;size_t offset;} G485Slot;
static G485Slot g485_slots[169];static int g485_count,g485_enabled;
static G485Malloc g485_malloc;static G485Free g485_free,g485_hostfree;
static G485HostAlloc g485_hostalloc;static G485Memcpy g485_copy;
static G485IntOut g485_devicecount,g485_runtimeversion;static G485IntIn g485_setdevice;
static G485Void g485_sync,g485_last;static G485MemInfo g485_meminfo;
static G485CuInit g485_cuinit;static G485Name g485_name;static G485Attr g485_attr;
static G485Create g485_create;static G485Handle g485_destroy;
static G485HandleOut g485_version,g485_getpointer,g485_getmath;
static G485HandleIn g485_setpointer,g485_setmath;
static G485StreamIn g485_setstream;static G485StreamOut g485_getstream;
static G485Workspace g485_setworkspace;static G485Gemm g485_gemm;
static HMODULE g485_modules[4];static char g485_paths[4][MAX_PATH],g485_device[256];
static void *g485_handle,*g485_weights,*g485_in,*g485_out,*g485_work;
static int8_t *g485_hin;static int32_t *g485_hout;static int16_t *g485_codes;
static float *g485_alphas;static size_t g485_wbytes,g485_ibytes,g485_obytes,g485_cbytes,g485_abytes;
static size_t g485_free_before,g485_free_after,g485_total;static int g485_rt,g485_blas,g485_major,g485_minor;
static double g485_setup_seconds;static uint64_t g485_calls,g485_queries,g485_h2d,g485_d2h;
static void g485_status(int code,const char *op){if(code){fprintf(stderr,"meth485_api_error:%s:%d\n",op,code);fflush(stderr);exit(3);}}
#define G485_OK(call) g485_status((call),#call)
static FARPROC g485_symbol(HMODULE module,const char *name){FARPROC p=GetProcAddress(module,name);if(!p){fprintf(stderr,"meth485_missing_symbol:%s:%lu\n",name,(unsigned long)GetLastError());exit(3);}return p;}
#define G485_BIND(var,type,module,name) var=(type)g485_symbol(module,name)
static int g485_pad(int n){require(n>0&&n<=65536,"gpu_pad");return (n+7)&~7;}
static void g485_reset(void){g485_calls=g485_queries=g485_h2d=g485_d2h=0;}
static void g485_json_string(const char *p){putchar('"');for(;*p;p++){if(*p=='\\'||*p=='"')putchar('\\');putchar(*p);}putchar('"');}
static void g485_json(void){
    printf(",\"gpu\":{\"enabled\":%s,\"calls\":%llu,\"queries\":%llu,\"h2d_bytes\":%llu,\"d2h_bytes\":%llu",g485_enabled?"true":"false",(unsigned long long)g485_calls,(unsigned long long)g485_queries,(unsigned long long)g485_h2d,(unsigned long long)g485_d2h);
    if(g485_enabled){printf(",\"matrix_count\":%d,\"weight_bytes\":%llu,\"device_explicit_bytes\":%llu,\"pinned_host_bytes\":%llu,\"cpu_query_bytes\":%llu,\"workspace_bytes\":8388608,\"setup_seconds\":%.9f,\"runtime_version\":%d,\"cublas_version\":%d,\"compute_major\":%d,\"compute_minor\":%d,\"free_before_setup\":%llu,\"free_after_setup\":%llu,\"device_total\":%llu,\"device_name\":",g485_count,(unsigned long long)g485_wbytes,(unsigned long long)(g485_wbytes+g485_ibytes+g485_obytes+(8<<20)),(unsigned long long)(g485_ibytes+g485_obytes),(unsigned long long)(g485_cbytes+g485_abytes),g485_setup_seconds,g485_rt,g485_blas,g485_major,g485_minor,(unsigned long long)g485_free_before,(unsigned long long)g485_free_after,(unsigned long long)g485_total);g485_json_string(g485_device);printf(",\"module_paths\":[");for(int i=0;i<4;i++){if(i)putchar(',');g485_json_string(g485_paths[i]);}printf("]");}
    printf("}");
}
static HMODULE g485_load(int i,const char *path){
    HMODULE h=LoadLibraryExA(path,NULL,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if(!h){fprintf(stderr,"meth485_dll_error:%s:%lu\n",path,(unsigned long)GetLastError());exit(3);}
    require(GetModuleFileNameA(h,g485_paths[i],MAX_PATH)>0&&!_stricmp(path,g485_paths[i]),"bound_module_path");return h;
}
static void g485_runtime(void){
    const char *dir=getenv("SILICON485_LIBDIR");require(dir&&strlen(dir)<MAX_PATH-40,"bound_CUDA_library_directory");
    const char *names[]={"cudart64_12.dll","cublasLt64_12.dll","cublas64_12.dll"};char path[MAX_PATH];
    for(int i=0;i<3;i++){snprintf(path,sizeof(path),"%s\\%s",dir,names[i]);g485_modules[i]=g485_load(i,path);}
    g485_modules[3]=g485_load(3,"C:\\Windows\\System32\\nvcuda.dll");
    HMODULE rt=g485_modules[0],bl=g485_modules[2],dr=g485_modules[3];
    G485_BIND(g485_malloc,G485Malloc,rt,"cudaMalloc");G485_BIND(g485_free,G485Free,rt,"cudaFree");
    G485_BIND(g485_hostalloc,G485HostAlloc,rt,"cudaHostAlloc");G485_BIND(g485_hostfree,G485Free,rt,"cudaFreeHost");
    G485_BIND(g485_copy,G485Memcpy,rt,"cudaMemcpy");G485_BIND(g485_sync,G485Void,rt,"cudaDeviceSynchronize");
    G485_BIND(g485_last,G485Void,rt,"cudaGetLastError");G485_BIND(g485_meminfo,G485MemInfo,rt,"cudaMemGetInfo");
    G485_BIND(g485_devicecount,G485IntOut,rt,"cudaGetDeviceCount");G485_BIND(g485_runtimeversion,G485IntOut,rt,"cudaRuntimeGetVersion");
    G485_BIND(g485_setdevice,G485IntIn,rt,"cudaSetDevice");
    G485_BIND(g485_cuinit,G485CuInit,dr,"cuInit");G485_BIND(g485_name,G485Name,dr,"cuDeviceGetName");G485_BIND(g485_attr,G485Attr,dr,"cuDeviceGetAttribute");
    G485_BIND(g485_create,G485Create,bl,"cublasCreate_v2");G485_BIND(g485_destroy,G485Handle,bl,"cublasDestroy_v2");
    G485_BIND(g485_version,G485HandleOut,bl,"cublasGetVersion_v2");
    G485_BIND(g485_setpointer,G485HandleIn,bl,"cublasSetPointerMode_v2");G485_BIND(g485_getpointer,G485HandleOut,bl,"cublasGetPointerMode_v2");
    G485_BIND(g485_setmath,G485HandleIn,bl,"cublasSetMathMode");G485_BIND(g485_getmath,G485HandleOut,bl,"cublasGetMathMode");
    G485_BIND(g485_setstream,G485StreamIn,bl,"cublasSetStream_v2");G485_BIND(g485_getstream,G485StreamOut,bl,"cublasGetStream_v2");
    G485_BIND(g485_setworkspace,G485Workspace,bl,"cublasSetWorkspace");G485_BIND(g485_gemm,G485Gemm,bl,"cublasGemmEx");
    G485_OK(g485_cuinit(0));int count=0;G485_OK(g485_devicecount(&count));require(count==1,"one_bound_GPU");
    G485_OK(g485_setdevice(0));G485_OK(g485_name(g485_device,sizeof(g485_device),0));
    G485_OK(g485_attr(&g485_major,75,0));G485_OK(g485_attr(&g485_minor,76,0));
    require(strstr(g485_device,"RTX 3060")&&g485_major==8&&g485_minor==6,"bound_RTX3060_CC86");
    G485_OK(g485_runtimeversion(&g485_rt));require(g485_rt==12040,"bound_CUDA124_runtime");
    G485_OK(g485_meminfo(&g485_free_before,&g485_total));require(g485_total>=((size_t)11<<30),"device_capacity");
    G485_OK(g485_create(&g485_handle));G485_OK(g485_version(g485_handle,&g485_blas));
    G485_OK(g485_setpointer(g485_handle,0));G485_OK(g485_setmath(g485_handle,0));G485_OK(g485_setstream(g485_handle,NULL));
    int pointer=-1,math=-1;void *stream=(void *)1;G485_OK(g485_getpointer(g485_handle,&pointer));G485_OK(g485_getmath(g485_handle,&math));G485_OK(g485_getstream(g485_handle,&stream));require(pointer==0&&math==0&&stream==NULL,"cublas_mode_readback");
    G485_OK(g485_malloc(&g485_work,8<<20));G485_OK(g485_setworkspace(g485_handle,g485_work,8<<20));
}
static void g485_add(Matrix w,int qmax){
    require(w.q&&w.scales&&w.cols<=4096&&g485_count<169&&qmax>0&&qmax<=CTX,"shared_GPU_registration");
    for(int i=0;i<g485_count;i++)require(g485_slots[i].w.q!=w.q,"distinct_shared_codes");
    G485Slot *s=g485_slots+g485_count++;s->w=w;s->mp=g485_pad(w.rows);s->dp=g485_pad(w.cols);s->qmax=qmax;s->offset=g485_wbytes;
    g485_wbytes+=(size_t)s->mp*s->dp;
    size_t in=(size_t)s->dp*4*qmax,out=(size_t)s->mp*4*qmax*4,codes=(size_t)w.cols*qmax*2,alphas=(size_t)qmax*4;
    if(in>g485_ibytes)g485_ibytes=in;if(out>g485_obytes)g485_obytes=out;if(codes>g485_cbytes)g485_cbytes=codes;if(alphas>g485_abytes)g485_abytes=alphas;
}
static void g485_allocate(void){
    require(g485_wbytes+g485_ibytes+g485_obytes+(8<<20)<=(2ULL<<30),"explicit_device_budget");
    G485_OK(g485_malloc(&g485_weights,g485_wbytes));G485_OK(g485_malloc(&g485_in,g485_ibytes));G485_OK(g485_malloc(&g485_out,g485_obytes));
    G485_OK(g485_hostalloc((void **)&g485_hin,g485_ibytes,0));G485_OK(g485_hostalloc((void **)&g485_hout,g485_obytes,0));
    g485_codes=alloc(g485_cbytes,1);g485_alphas=alloc(g485_abytes,1);
    for(int i=0;i<g485_count;i++){G485Slot *s=g485_slots+i;size_t bytes=(size_t)s->mp*s->dp;int8_t *p=NULL;const int8_t *source=s->w.q;
        for(uint32_t row=0;row<s->w.rows;row++)require(isfinite(s->w.scales[row])&&s->w.scales[row]>0.f,"original_row_scale");
        if(s->mp!=(int)s->w.rows||s->dp!=(int)s->w.cols){p=alloc(bytes,1);for(uint32_t r=0;r<s->w.rows;r++)memcpy(p+(size_t)r*s->dp,source+(size_t)r*s->w.cols,s->w.cols);source=p;}
        G485_OK(g485_copy((char *)g485_weights+s->offset,source,bytes,1));free(p);
    }
    G485_OK(g485_sync());size_t total;G485_OK(g485_meminfo(&g485_free_after,&total));require(total==g485_total,"device_total_stable");
}
static void g485_attention(Attention a,int batch){g485_add(a.q,batch);g485_add(a.k,batch);g485_add(a.v,batch);g485_add(a.o,1);}
static void g485_model_setup(Model *m){
    if(!g485_enabled)return;double begin=cost_clock();g485_runtime();
    for(int l=0;l<m->c.enc;l++){Block *b=m->enc+l;g485_attention(b->self,CTX);if(!b->sparse){g485_add(b->wi,1);g485_add(b->wo,1);}}
    for(int l=0;l<m->c.dec;l++){Block *b=m->dec+l;g485_attention(b->self,1);g485_add(b->cross.q,1);g485_add(b->cross.k,CTX);g485_add(b->cross.v,CTX);g485_add(b->cross.o,1);if(!b->sparse){g485_add(b->wi,1);g485_add(b->wo,1);}}
    g485_add(m->head,1);require(g485_count==169&&g485_wbytes==166232064&&g485_ibytes+g485_obytes==3932160,"complete_484_shared_geometry");g485_allocate();g485_setup_seconds=cost_clock()-begin;g485_reset();
}
static void g485_teardown(void){
    if(!g485_enabled)return;G485_OK(g485_sync());G485_OK(g485_destroy(g485_handle));G485_OK(g485_free(g485_weights));G485_OK(g485_free(g485_in));G485_OK(g485_free(g485_out));G485_OK(g485_free(g485_work));G485_OK(g485_hostfree(g485_hin));G485_OK(g485_hostfree(g485_hout));free(g485_codes);free(g485_alphas);
    for(int i=2;i>=0;i--)require(FreeLibrary(g485_modules[i]),"CUDA_module_release");require(FreeLibrary(g485_modules[3]),"driver_module_release");
    g485_count=0;g485_wbytes=g485_ibytes=g485_obytes=g485_cbytes=g485_abytes=0;
}
static int g485_eval(Matrix w,const float *x,float *y,int rows,int cols,int queries){
    if(!g485_enabled||!w.q)return 0;G485Slot *s=NULL;for(int i=0;i<g485_count;i++)if(g485_slots[i].w.q==w.q){s=g485_slots+i;break;}if(!s)return 0;
    require(rows==(int)s->w.rows&&cols==(int)s->w.cols&&queries>0&&queries<=s->qmax,"GPU_operator_shape");
    size_t input=(size_t)s->dp*4*queries,output=(size_t)s->mp*4*queries*4;require(input<=g485_ibytes&&output<=g485_obytes,"GPU_operator_buffer");memset(g485_hin,0,input);
    for(int t=0;t<queries;t++){g485_alphas[t]=head_activation_codes(x+(size_t)t*cols,g485_codes+(size_t)t*cols,cols);
        for(int j=0;j<cols;j++){uint16_t u=(uint16_t)g485_codes[(size_t)t*cols+j];size_t at=(size_t)t*4*s->dp+j;g485_hin[at]=u&127;g485_hin[at+s->dp]=(u>>7)&127;g485_hin[at+2*s->dp]=(int8_t)((u>>14)-4*(u>=32768));}}
    G485_OK(g485_copy(g485_in,g485_hin,input,1));int32_t one=1,zero=0;
    /* Column major op(W)=T, B=N. Types R_8I=3/R_32I=10, COMPUTE_32I=72, DEFAULT=-1. */
    G485_OK(g485_gemm(g485_handle,1,0,s->mp,4*queries,s->dp,&one,(char *)g485_weights+s->offset,3,s->dp,g485_in,3,s->dp,&zero,g485_out,10,s->mp,72,-1));
    /* Blocking default-stream D2H plus explicit synchronization: all work charged before return. */
    G485_OK(g485_copy(g485_hout,g485_out,output,2));G485_OK(g485_sync());G485_OK(g485_last());
    for(int t=0;t<queries;t++)for(int r=0;r<rows;r++){size_t at=(size_t)t*4*s->mp+r;int64_t value=(int64_t)g485_hout[at]+128LL*g485_hout[at+s->mp]+16384LL*g485_hout[at+2*s->mp];y[(size_t)t*rows+r]=(float)(((double)value*(double)w.scales[r])*(double)g485_alphas[t]);}
    g485_calls++;g485_queries+=queries;g485_h2d+=input;g485_d2h+=output;return 1;
}
