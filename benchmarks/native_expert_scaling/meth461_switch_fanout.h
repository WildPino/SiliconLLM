/* ONE exact A16 common-input fanout; original integer dot and scale order. */
static int fanout_disjoint(const void *a,size_t an,const void *b,size_t bn){
    uintptr_t av=(uintptr_t)a,bv=(uintptr_t)b;
    require(an<=UINTPTR_MAX-av&&bn<=UINTPTR_MAX-bv,"fanout_address_bounds");
    return av+an<=bv||bv+bn<=av;
}
static void mv_fanout(const Matrix *matrices,float *const *outputs,int maps,const float *x,int rows,int cols,int tokens){
    require((maps==2||maps==3)&&rows>0&&rows<=65536&&cols>0&&cols<=4096&&tokens>0&&tokens<=CTX,"fanout_dimensions");
    size_t input_bytes=(size_t)tokens*cols*4,output_bytes=(size_t)tokens*rows*4;
    for(int map=0;map<maps;map++){
        require(matrices[map].q&&matrices[map].scales&&matrices[map].rows==(uint32_t)rows&&matrices[map].cols==(uint32_t)cols,"fanout_I8_matrix_dimensions");
        require(fanout_disjoint(outputs[map],output_bytes,x,input_bytes),"fanout_input_output_alias");
        for(int earlier=0;earlier<map;earlier++)require(fanout_disjoint(outputs[map],output_bytes,outputs[earlier],output_bytes),"fanout_output_output_alias");
    }
    double cost_start=cost_enabled&&cost_profile?cost_clock():0.;
    if(cost_enabled){CostCounter *counter=&cost_counts[cost_phase][cost_kind];counter->calls+=(uint64_t)maps*tokens;
        counter->codes+=(uint64_t)maps*rows*cols*tokens;counter->scales+=(uint64_t)maps*rows*4*tokens;}
    int16_t single_codes[4096];int16_t *codes=tokens==1?single_codes:alloc((size_t)tokens*cols,2);float scales[CTX];
    for(int token=0;token<tokens;token++)scales[token]=head_activation_codes(x+(size_t)token*cols,codes+(size_t)token*cols,cols);
    #pragma omp parallel for schedule(static) if(rows>64)
    for(int index=0;index<maps*rows;index++){
        int map=index/rows,row=index%rows;Matrix w=matrices[map];float *y=outputs[map];
        for(int token=0;token<tokens;token++){
            int64_t value=head_integer_dot(w.q+(size_t)row*cols,codes+(size_t)token*cols,cols);
            y[(size_t)token*rows+row]=(float)(((double)value*(double)w.scales[row])*(double)scales[token]);
        }
    }
    if(tokens!=1)free(codes);
    if(cost_enabled&&cost_profile)cost_counts[cost_phase][cost_kind].matrix_seconds+=cost_clock()-cost_start;
}
