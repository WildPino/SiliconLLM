/* Actual full source bridge: reuse unchanged328 arithmetic, add every-tensor audit. */
#define main meth328_entry
#include "meth328_switch_f32_reference.c"
#undef main
#include <bcrypt.h>

static void native_audit(const char *spec,const char *output){
    Model *model=load(spec);FILE *stream=fopen(output,"wb");require(stream!=NULL,"audit_open");
    BCRYPT_ALG_HANDLE algorithm=NULL;require(BCryptOpenAlgorithmProvider(&algorithm,BCRYPT_SHA256_ALGORITHM,NULL,0)>=0,"sha_algorithm");
    for(int i=0;i<model->ntensors;i++){
        Tensor *t=model->tensors+i;BCRYPT_HASH_HANDLE hash=NULL;unsigned char digest[32];
        require(BCryptCreateHash(algorithm,&hash,NULL,0,NULL,0,0)>=0,"sha_create");
        uint64_t bytes=(uint64_t)t->shape[0]*t->shape[1]*4,position=0;
        while(position<bytes){ULONG count=(ULONG)((bytes-position)>(4u<<20)?(4u<<20):(bytes-position));
            require(BCryptHashData(hash,(PUCHAR)((const unsigned char *)t->value+position),count,0)>=0,"sha_update");position+=count;}
        require(BCryptFinishHash(hash,digest,32,0)>=0,"sha_finish");BCryptDestroyHash(hash);
        char hex[65];for(int j=0;j<32;j++)snprintf(hex+2*j,3,"%02x",digest[j]);hex[64]=0;
        fprintf(stream,"%s\t%u\t%u\t%u\t%llu\t%s\n",t->name,t->dims,t->shape[0],t->shape[1],(unsigned long long)bytes,hex);
        if(i%64==63)SetProcessWorkingSetSize(GetCurrentProcess(),(SIZE_T)-1,(SIZE_T)-1);
    }
    require(fclose(stream)==0,"audit_close");BCryptCloseAlgorithmProvider(algorithm,0);
    printf("{\"actual_native_tensor_hashes\":%d,\"source_experts\":%d}\n",model->ntensors,model->c.n);unload(model);
}
int main(int argc,char **argv){if(argc==4&&!strcmp(argv[1],"--audit")){native_audit(argv[2],argv[3]);return 0;}return meth328_entry(argc,argv);}
