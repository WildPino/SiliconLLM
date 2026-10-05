/* Arithmetic apparatus only: never a synthetic expert quality/capacity test. */
static int fanout_alias_control(void){
    int8_t weights[34]={0};float scales[2]={1.f,1.f},x[17]={0},other[1]={0};
    Matrix matrices[2]={{NULL,scales,weights,1,17},{NULL,scales+1,weights+17,1,17}};
    float *outputs[2]={x,other};mv_fanout(matrices,outputs,2,x,1,17,1);
    require(0,"fanout_alias_not_rejected");return 0;
}
static int fanout_controls(const char *input,const char *output,int threads){
    require(threads==3||threads==6,"fanout_control_threads");worker_setup(threads);
    FILE *f=fopen(input,"rb"),*out=fopen(output,"wb");require(f&&out,"fanout_control_files");
    char magic[8];require(fread(magic,1,8,f)==8&&!memcmp(magic,"MF461I01",8),"fanout_control_magic");
    uint32_t cases=u32(f);require(cases==5,"fanout_control_count");fwrite("MF461O01",1,8,out);fwrite(&cases,4,1,out);
    for(uint32_t ci=0;ci<cases;ci++){
        uint32_t maps=u32(f),rows=u32(f),cols=u32(f),tokens=u32(f);
        require((maps==2||maps==3)&&(rows==2||rows==65)&&(cols==17||cols==768||cols==4096)&&(tokens==1||tokens==3),"fanout_control_shape");
        Matrix matrices[3];float *original[3],*candidate[3];int8_t *weights[3];float *row_scales[3];
        float *x=alloc((size_t)tokens*cols,4);require(fread(x,4,(size_t)tokens*cols,f)==(size_t)tokens*cols,"fanout_control_x");
        for(uint32_t map=0;map<maps;map++){
            weights[map]=alloc((size_t)rows*cols,1);row_scales[map]=alloc(rows,4);original[map]=alloc((size_t)tokens*rows,4);candidate[map]=alloc((size_t)tokens*rows,4);
            require(fread(weights[map],1,(size_t)rows*cols,f)==(size_t)rows*cols&&fread(row_scales[map],4,rows,f)==rows,"fanout_control_matrix");
            matrices[map]=(Matrix){NULL,row_scales[map],weights[map],rows,cols};
            if(tokens==1)mv(matrices[map],x,original[map],rows,cols);else mv_batch(matrices[map],x,original[map],rows,cols,tokens);
        }
        mv_fanout(matrices,candidate,maps,x,rows,cols,tokens);
        uint32_t shape[]={maps,rows,cols,tokens};fwrite(shape,4,4,out);
        for(uint32_t map=0;map<maps;map++){
            require(!memcmp(original[map],candidate[map],(size_t)tokens*rows*4),"fanout_original_candidate_bytes");
            fwrite(candidate[map],4,(size_t)tokens*rows,out);free(weights[map]);free(row_scales[map]);free(original[map]);free(candidate[map]);
        }free(x);
    }
    require(fgetc(f)==EOF&&fclose(f)==0&&fclose(out)==0,"fanout_control_close");
    printf("{\"cases\":5,\"threads\":%d,\"original_candidate_bytes_exact\":true",threads);worker_json();printf("}\n");return 0;
}
