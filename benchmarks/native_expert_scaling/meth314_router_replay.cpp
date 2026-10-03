#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <vector>
#include <cmath>
static void require(bool b){if(!b){std::fprintf(stderr,"M314 router apparatus failure\n");std::exit(91);}}
static void read(FILE*f,void*p,size_t n){require(std::fread(p,1,n,f)==n);}
struct Coordinate{uint32_t chunk,batch,token;};
struct Output{int32_t expert[4];float gate[4];};
int main(int argc,char**argv){
    require(argc==3);FILE*f=std::fopen(argv[1],"rb");require(f);char magic[8];read(f,magic,8);require(!std::memcmp(magic,"M314RTR1",8));
    uint32_t n;read(f,&n,4);require(n>0&&n<=64000);std::vector<float>w(1536*64),bias(64),x(size_t(n)*1536);
    std::vector<Coordinate>coordinates(n);read(f,w.data(),w.size()*4);read(f,bias.data(),bias.size()*4);read(f,coordinates.data(),n*sizeof(Coordinate));read(f,x.data(),x.size()*4);require(std::fgetc(f)==EOF);std::fclose(f);
    for(uint32_t i=0;i<n;i++){auto c=coordinates[i];require(c.chunk<125&&c.batch<4&&c.token<128);if(i){auto p=coordinates[i-1];require(c.chunk>p.chunk||(c.chunk==p.chunk&&(c.batch>p.batch||(c.batch==p.batch&&c.token>p.token))));}}
    for(float v:x)require(std::isfinite(v));
    ggml_init_params params={32u<<20,nullptr,true};ggml_context*ctx=ggml_init(params);require(ctx);
    auto weights=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,1536,64);
    auto correction=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,64,1);
    auto input=ggml_new_tensor_2d(ctx,GGML_TYPE_F32,1536,128);
    auto logits=ggml_mul_mat(ctx,weights,input);auto probability=ggml_sigmoid(ctx,logits);
    auto rank=ggml_add(ctx,probability,correction);auto selected=ggml_argsort_top_k(ctx,rank,4);
    auto compact_ids=ggml_cont(ctx,selected);
    auto probs3=ggml_reshape_3d(ctx,probability,1,64,128);
    auto picked=ggml_get_rows(ctx,probs3,selected);auto picked2=ggml_reshape_2d(ctx,picked,4,128);
    auto denominator=ggml_clamp(ctx,ggml_sum_rows(ctx,picked2),6.103515625e-5f,INFINITY);
    auto gates=ggml_div(ctx,picked2,denominator);
    auto graph=ggml_new_graph(ctx);ggml_build_forward_expand(graph,compact_ids);ggml_build_forward_expand(graph,gates);
    auto backend=ggml_backend_cpu_init();require(backend);ggml_backend_cpu_set_n_threads(backend,6);
    auto buffer=ggml_backend_alloc_ctx_tensors(ctx,backend);require(buffer);
    ggml_backend_tensor_set(weights,w.data(),0,w.size()*4);ggml_backend_tensor_set(correction,bias.data(),0,bias.size()*4);
    std::vector<float>batch(1536*128),values(4*128);std::vector<int32_t>ids(4*128);std::vector<Output>out(n);uint32_t groups=0;
    for(uint32_t begin=0;begin<n;){
        uint32_t end=begin+1;while(end<n&&coordinates[end].chunk==coordinates[begin].chunk&&coordinates[end].batch==coordinates[begin].batch)end++;
        std::fill(batch.begin(),batch.end(),0.0f);
        for(uint32_t i=begin;i<end;i++)std::memcpy(batch.data()+size_t(coordinates[i].token)*1536,x.data()+size_t(i)*1536,1536*4);
        ggml_backend_tensor_set(input,batch.data(),0,batch.size()*4);require(ggml_backend_graph_compute(backend,graph)==GGML_STATUS_SUCCESS);
        ggml_backend_tensor_get(compact_ids,ids.data(),0,ids.size()*4);ggml_backend_tensor_get(gates,values.data(),0,values.size()*4);
        for(uint32_t i=begin;i<end;i++)for(unsigned k=0;k<4;k++){unsigned index=coordinates[i].token*4+k;require(ids[index]>=0&&ids[index]<64&&std::isfinite(values[index])&&values[index]>0);out[i].expert[k]=ids[index];out[i].gate[k]=values[index];}
        begin=end;groups++;
    }
    static_assert(sizeof(Output)==32);FILE*output=std::fopen(argv[2],"wb");require(output);require(std::fwrite("MR314OU1",1,8,output)==8);require(std::fwrite(&n,1,4,output)==4);require(std::fwrite(out.data(),sizeof(Output),n,output)==n);require(std::fclose(output)==0);
    std::printf("{\"event\":\"native_router\",\"rows\":%u,\"batch_groups\":%u,\"batch_width\":128,\"threads\":6}\n",n,groups);
    ggml_backend_buffer_free(buffer);ggml_backend_free(backend);ggml_free(ctx);return 0;
}
