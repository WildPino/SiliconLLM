/* METH486: one fixed CUDA integer layout; minimal Win64 ABI, no CUDA SDK dependency. */
_Static_assert(sizeof(void *)==8 && sizeof(int)==4 && sizeof(size_t)==8,"Win64 CUDA ABI");
_Static_assert(sizeof(float)==4 && sizeof(int16_t)==2 && sizeof(int32_t)==4,"wire ABI");
typedef int (__stdcall *G486Malloc)(void **,size_t);
typedef int (__stdcall *G486Free)(void *);
typedef int (__stdcall *G486HostAlloc)(void **,size_t,unsigned int);
typedef int (__stdcall *G486Memcpy)(void *,const void *,size_t,int);
typedef int (__stdcall *G486IntOut)(int *);
typedef int (__stdcall *G486IntIn)(int);
typedef int (__stdcall *G486Void)(void);
typedef int (__stdcall *G486MemInfo)(size_t *,size_t *);
typedef int (__stdcall *G486CuInit)(unsigned int);
typedef int (__stdcall *G486Name)(char *,int,int);
typedef int (__stdcall *G486Attr)(int *,int,int);
typedef int (__stdcall *G486Create)(void **);
typedef int (__stdcall *G486Handle)(void *);
typedef int (__stdcall *G486HandleOut)(void *,int *);
typedef int (__stdcall *G486HandleIn)(void *,int);
typedef int (__stdcall *G486StreamIn)(void *,void *);
typedef int (__stdcall *G486StreamOut)(void *,void **);
typedef int (__stdcall *G486Workspace)(void *,void *,size_t);
typedef int (__stdcall *G486Gemm)(void *,int,int,int,int,int,const void *,const void *,int,int,const void *,int,int,const void *,void *,int,int,int,int);
typedef struct {Matrix w;int mp,dp,qmax;size_t offset;} G486Slot;
static G486Slot g486_slots[169];static int g486_count,g486_enabled;
static G486Malloc g486_malloc;static G486Free g486_free,g486_hostfree;
static G486HostAlloc g486_hostalloc;static G486Memcpy g486_copy;
static G486IntOut g486_devicecount,g486_runtimeversion;static G486IntIn g486_setdevice;
static G486Void g486_sync,g486_last;static G486MemInfo g486_meminfo;
static G486CuInit g486_cuinit;static G486Name g486_name;static G486Attr g486_attr;
static G486Create g486_create;static G486Handle g486_destroy;
static G486HandleOut g486_version,g486_getpointer,g486_getmath;
static G486HandleIn g486_setpointer,g486_setmath;
static G486StreamIn g486_setstream;static G486StreamOut g486_getstream;
static G486Workspace g486_setworkspace;static G486Gemm g486_gemm;
static HMODULE g486_modules[4];static char g486_paths[4][MAX_PATH],g486_device[256];
static void *g486_handle,*g486_weights,*g486_in,*g486_out,*g486_work;
static int8_t *g486_hin;static int32_t *g486_hout;static int16_t *g486_codes;
static float *g486_alphas;static size_t g486_wbytes,g486_ibytes,g486_obytes,g486_cbytes,g486_abytes;
static size_t g486_free_before,g486_free_after,g486_total;static int g486_rt,g486_blas,g486_major,g486_minor;
static double g486_setup_seconds;static uint64_t g486_calls,g486_queries,g486_h2d,g486_d2h;
static void g486_status(int code,const char *op){if(code){fprintf(stderr,"meth486_api_error:%s:%d\n",op,code);fflush(stderr);exit(3);}}
#define G486_OK(call) g486_status((call),#call)
static FARPROC g486_symbol(HMODULE module,const char *name){FARPROC p=GetProcAddress(module,name);if(!p){fprintf(stderr,"meth486_missing_symbol:%s:%lu\n",name,(unsigned long)GetLastError());exit(3);}return p;}
#define G486_BIND(var,type,module,name) var=(type)g486_symbol(module,name)
static int g486_pad(int n){require(n>0&&n<=65536,"gpu_pad");return (n+7)&~7;}
static void g486_reset(void){g486_calls=g486_queries=g486_h2d=g486_d2h=0;}
static void g486_json_string(const char *p){putchar('"');for(;*p;p++){if(*p=='\\'||*p=='"')putchar('\\');putchar(*p);}putchar('"');}
static void g486_json(void){
    printf(",\"gpu\":{\"enabled\":%s,\"calls\":%llu,\"queries\":%llu,\"h2d_bytes\":%llu,\"d2h_bytes\":%llu",g486_enabled?"true":"false",(unsigned long long)g486_calls,(unsigned long long)g486_queries,(unsigned long long)g486_h2d,(unsigned long long)g486_d2h);
    if(g486_enabled){printf(",\"matrix_count\":%d,\"weight_bytes\":%llu,\"device_explicit_bytes\":%llu,\"pinned_host_bytes\":%llu,\"cpu_query_bytes\":%llu,\"workspace_bytes\":8388608,\"setup_seconds\":%.9f,\"runtime_version\":%d,\"cublas_version\":%d,\"compute_major\":%d,\"compute_minor\":%d,\"free_before_setup\":%llu,\"free_after_setup\":%llu,\"device_total\":%llu,\"device_name\":",g486_count,(unsigned long long)g486_wbytes,(unsigned long long)(g486_wbytes+g486_ibytes+g486_obytes+(8<<20)),(unsigned long long)(g486_ibytes+g486_obytes),(unsigned long long)(g486_cbytes+g486_abytes),g486_setup_seconds,g486_rt,g486_blas,g486_major,g486_minor,(unsigned long long)g486_free_before,(unsigned long long)g486_free_after,(unsigned long long)g486_total);g486_json_string(g486_device);printf(",\"module_paths\":[");for(int i=0;i<4;i++){if(i)putchar(',');g486_json_string(g486_paths[i]);}printf("]");}
    printf("}");
}
static HMODULE g486_load(int i,const char *path){
    HMODULE h=LoadLibraryExA(path,NULL,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if(!h){fprintf(stderr,"meth486_dll_error:%s:%lu\n",path,(unsigned long)GetLastError());exit(3);}
    require(GetModuleFileNameA(h,g486_paths[i],MAX_PATH)>0&&!_stricmp(path,g486_paths[i]),"bound_module_path");return h;
}
static void g486_runtime(void){
    const char *dir=getenv("SILICON486_LIBDIR");require(dir&&strlen(dir)<MAX_PATH-40,"bound_CUDA_library_directory");
    const char *names[]={"cudart64_12.dll","cublasLt64_12.dll","cublas64_12.dll"};char path[MAX_PATH];
    for(int i=0;i<3;i++){snprintf(path,sizeof(path),"%s\\%s",dir,names[i]);g486_modules[i]=g486_load(i,path);}
    g486_modules[3]=g486_load(3,"C:\\Windows\\System32\\nvcuda.dll");
    HMODULE rt=g486_modules[0],bl=g486_modules[2],dr=g486_modules[3];
    G486_BIND(g486_malloc,G486Malloc,rt,"cudaMalloc");G486_BIND(g486_free,G486Free,rt,"cudaFree");
    G486_BIND(g486_hostalloc,G486HostAlloc,rt,"cudaHostAlloc");G486_BIND(g486_hostfree,G486Free,rt,"cudaFreeHost");
    G486_BIND(g486_copy,G486Memcpy,rt,"cudaMemcpy");G486_BIND(g486_sync,G486Void,rt,"cudaDeviceSynchronize");
    G486_BIND(g486_last,G486Void,rt,"cudaGetLastError");G486_BIND(g486_meminfo,G486MemInfo,rt,"cudaMemGetInfo");
    G486_BIND(g486_devicecount,G486IntOut,rt,"cudaGetDeviceCount");G486_BIND(g486_runtimeversion,G486IntOut,rt,"cudaRuntimeGetVersion");
    G486_BIND(g486_setdevice,G486IntIn,rt,"cudaSetDevice");
    G486_BIND(g486_cuinit,G486CuInit,dr,"cuInit");G486_BIND(g486_name,G486Name,dr,"cuDeviceGetName");G486_BIND(g486_attr,G486Attr,dr,"cuDeviceGetAttribute");
    G486_BIND(g486_create,G486Create,bl,"cublasCreate_v2");G486_BIND(g486_destroy,G486Handle,bl,"cublasDestroy_v2");
    G486_BIND(g486_version,G486HandleOut,bl,"cublasGetVersion_v2");
    G486_BIND(g486_setpointer,G486HandleIn,bl,"cublasSetPointerMode_v2");G486_BIND(g486_getpointer,G486HandleOut,bl,"cublasGetPointerMode_v2");
    G486_BIND(g486_setmath,G486HandleIn,bl,"cublasSetMathMode");G486_BIND(g486_getmath,G486HandleOut,bl,"cublasGetMathMode");
    G486_BIND(g486_setstream,G486StreamIn,bl,"cublasSetStream_v2");G486_BIND(g486_getstream,G486StreamOut,bl,"cublasGetStream_v2");
    G486_BIND(g486_setworkspace,G486Workspace,bl,"cublasSetWorkspace_v2");G486_BIND(g486_gemm,G486Gemm,bl,"cublasGemmEx");
    G486_OK(g486_cuinit(0));int count=0;G486_OK(g486_devicecount(&count));require(count==1,"one_bound_GPU");
    G486_OK(g486_setdevice(0));G486_OK(g486_name(g486_device,sizeof(g486_device),0));
    G486_OK(g486_attr(&g486_major,75,0));G486_OK(g486_attr(&g486_minor,76,0));
    require(strstr(g486_device,"RTX 3060")&&g486_major==8&&g486_minor==6,"bound_RTX3060_CC86");
    G486_OK(g486_runtimeversion(&g486_rt));require(g486_rt==12040,"bound_CUDA124_runtime");
    G486_OK(g486_meminfo(&g486_free_before,&g486_total));require(g486_total>=((size_t)11<<30),"device_capacity");
    G486_OK(g486_create(&g486_handle));G486_OK(g486_version(g486_handle,&g486_blas));
    G486_OK(g486_setpointer(g486_handle,0));G486_OK(g486_setmath(g486_handle,0));G486_OK(g486_setstream(g486_handle,NULL));
    int pointer=-1,math=-1;void *stream=(void *)1;G486_OK(g486_getpointer(g486_handle,&pointer));G486_OK(g486_getmath(g486_handle,&math));G486_OK(g486_getstream(g486_handle,&stream));require(pointer==0&&math==0&&stream==NULL,"cublas_mode_readback");
    G486_OK(g486_malloc(&g486_work,8<<20));G486_OK(g486_setworkspace(g486_handle,g486_work,8<<20));
}
static void g486_add(Matrix w,int qmax){
    require(w.q&&w.scales&&w.cols<=4096&&g486_count<169&&qmax>0&&qmax<=CTX,"shared_GPU_registration");
    for(int i=0;i<g486_count;i++)require(g486_slots[i].w.q!=w.q,"distinct_shared_codes");
    G486Slot *s=g486_slots+g486_count++;s->w=w;s->mp=g486_pad(w.rows);s->dp=g486_pad(w.cols);s->qmax=qmax;s->offset=g486_wbytes;
    g486_wbytes+=(size_t)s->mp*s->dp;
    size_t in=(size_t)s->dp*4*qmax,out=(size_t)s->mp*4*qmax*4,codes=(size_t)w.cols*qmax*2,alphas=(size_t)qmax*4;
    if(in>g486_ibytes)g486_ibytes=in;if(out>g486_obytes)g486_obytes=out;if(codes>g486_cbytes)g486_cbytes=codes;if(alphas>g486_abytes)g486_abytes=alphas;
}
static void g486_allocate(void){
    require(g486_wbytes+g486_ibytes+g486_obytes+(8<<20)<=(2ULL<<30),"explicit_device_budget");
    G486_OK(g486_malloc(&g486_weights,g486_wbytes));G486_OK(g486_malloc(&g486_in,g486_ibytes));G486_OK(g486_malloc(&g486_out,g486_obytes));
    G486_OK(g486_hostalloc((void **)&g486_hin,g486_ibytes,0));G486_OK(g486_hostalloc((void **)&g486_hout,g486_obytes,0));
    g486_codes=alloc(g486_cbytes,1);g486_alphas=alloc(g486_abytes,1);
    for(int i=0;i<g486_count;i++){G486Slot *s=g486_slots+i;size_t bytes=(size_t)s->mp*s->dp;int8_t *p=NULL;const int8_t *source=s->w.q;
        for(uint32_t row=0;row<s->w.rows;row++)require(isfinite(s->w.scales[row])&&s->w.scales[row]>0.f,"original_row_scale");
        if(s->mp!=(int)s->w.rows||s->dp!=(int)s->w.cols){p=alloc(bytes,1);for(uint32_t r=0;r<s->w.rows;r++)memcpy(p+(size_t)r*s->dp,source+(size_t)r*s->w.cols,s->w.cols);source=p;}
        G486_OK(g486_copy((char *)g486_weights+s->offset,source,bytes,1));free(p);
    }
    G486_OK(g486_sync());size_t total;G486_OK(g486_meminfo(&g486_free_after,&total));require(total==g486_total,"device_total_stable");
}
static void g486_attention(Attention a,int batch){g486_add(a.q,batch);g486_add(a.k,batch);g486_add(a.v,batch);g486_add(a.o,1);}
static void g486_model_setup(Model *m){
    if(!g486_enabled)return;double begin=cost_clock();g486_runtime();
    for(int l=0;l<m->c.enc;l++){Block *b=m->enc+l;g486_attention(b->self,CTX);if(!b->sparse){g486_add(b->wi,1);g486_add(b->wo,1);}}
    for(int l=0;l<m->c.dec;l++){Block *b=m->dec+l;g486_attention(b->self,1);g486_add(b->cross.q,1);g486_add(b->cross.k,CTX);g486_add(b->cross.v,CTX);g486_add(b->cross.o,1);if(!b->sparse){g486_add(b->wi,1);g486_add(b->wo,1);}}
    g486_add(m->head,1);require(g486_count==169&&g486_wbytes==166232064&&g486_ibytes+g486_obytes==3932160,"complete_484_shared_geometry");g486_allocate();g486_setup_seconds=cost_clock()-begin;g486_reset();
}
static void g486_teardown(void){
    if(!g486_enabled)return;G486_OK(g486_sync());G486_OK(g486_destroy(g486_handle));G486_OK(g486_free(g486_weights));G486_OK(g486_free(g486_in));G486_OK(g486_free(g486_out));G486_OK(g486_free(g486_work));G486_OK(g486_hostfree(g486_hin));G486_OK(g486_hostfree(g486_hout));free(g486_codes);free(g486_alphas);
    for(int i=2;i>=0;i--)require(FreeLibrary(g486_modules[i]),"CUDA_module_release");require(FreeLibrary(g486_modules[3]),"driver_module_release");
    g486_count=0;g486_wbytes=g486_ibytes=g486_obytes=g486_cbytes=g486_abytes=0;
}
static int g486_eval(Matrix w,const float *x,float *y,int rows,int cols,int queries){
    if(!g486_enabled||!w.q)return 0;G486Slot *s=NULL;for(int i=0;i<g486_count;i++)if(g486_slots[i].w.q==w.q){s=g486_slots+i;break;}if(!s)return 0;
    require(rows==(int)s->w.rows&&cols==(int)s->w.cols&&queries>0&&queries<=s->qmax,"GPU_operator_shape");
    size_t input=(size_t)s->dp*4*queries,output=(size_t)s->mp*4*queries*4;require(input<=g486_ibytes&&output<=g486_obytes,"GPU_operator_buffer");memset(g486_hin,0,input);
    for(int t=0;t<queries;t++){g486_alphas[t]=head_activation_codes(x+(size_t)t*cols,g486_codes+(size_t)t*cols,cols);
        for(int j=0;j<cols;j++){uint16_t u=(uint16_t)g486_codes[(size_t)t*cols+j];size_t at=(size_t)t*4*s->dp+j;g486_hin[at]=u&127;g486_hin[at+s->dp]=(u>>7)&127;g486_hin[at+2*s->dp]=(int8_t)((u>>14)-4*(u>=32768));}}
    G486_OK(g486_copy(g486_in,g486_hin,input,1));int32_t one=1,zero=0;
    /* Column major op(W)=T, B=N. Types R_8I=3/R_32I=10, COMPUTE_32I=72, DEFAULT=-1. */
    G486_OK(g486_gemm(g486_handle,1,0,s->mp,4*queries,s->dp,&one,(char *)g486_weights+s->offset,3,s->dp,g486_in,3,s->dp,&zero,g486_out,10,s->mp,72,-1));
    /* Blocking default-stream D2H plus explicit synchronization: all work charged before return. */
    G486_OK(g486_copy(g486_hout,g486_out,output,2));G486_OK(g486_sync());G486_OK(g486_last());
    for(int t=0;t<queries;t++)for(int r=0;r<rows;r++){size_t at=(size_t)t*4*s->mp+r;int64_t value=(int64_t)g486_hout[at]+128LL*g486_hout[at+s->mp]+16384LL*g486_hout[at+2*s->mp];y[(size_t)t*rows+r]=(float)(((double)value*(double)w.scales[r])*(double)g486_alphas[t]);}
    g486_calls++;g486_queries+=queries;g486_h2d+=input;g486_d2h+=output;return 1;
}
