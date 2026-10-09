/* Calls the same frozen original MoE body, with F32 inputs independent of core. */
#define main original_capacity_main
#include "original_packed_capacity.c"
#undef main
int main(int argc,char**argv){
    if(argc!=5)fail("usage: model inputs bank_output routes");
    if(fesetround(FE_TONEAREST))fail("round mode");
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_OFF);_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_OFF);
    load_model(argv[1]);FILE*f=fopen(argv[2],"rb");uint32_t n;
    if(!f||fread(&n,4,1,f)!=1||n<1||n>1024)fail("bank input header");
    FILE*yf=output(argv[3]),*rf=output(argv[4]);float x[D],y[D];
    for(uint32_t t=0;t<n;t++){
        if(fread(x,4,D,f)!=D)fail("bank input extent");
        for(int i=0;i<D;i++)if(!isfinite(x[i]))fail("bank input finite");
        for(int l=0;l<L;l++){
            mlp_moe(l,x,y,1,NULL);put(yf,y,sizeof(y));
            put(rf,actual_ids[l],sizeof(actual_ids[l]));put(rf,actual_mass[l],sizeof(actual_mass[l]));
        }
    }
    if(fgetc(f)!=EOF)fail("bank input tail");fclose(f);fclose(yf);fclose(rf);return 0;
}
