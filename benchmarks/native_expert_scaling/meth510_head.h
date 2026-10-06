/* Fixed k=8 source-row replacement. Original I8 tail remains eligible. */
#define HYBRID_K510 8
typedef struct {
    uint64_t calls,visited,comparisons,source_rows,source_MACs,f32_bytes;
    double scan_seconds,refine_seconds;
} HybridCounter510;
static HybridCounter510 hybrid_counts510;
static int hybrid_ids510[HYBRID_K510];
static double hybrid_dots510[HYBRID_K510];

static void hybrid_refine510(Matrix w,const float *source,const float *x,float *y,int rows,int cols){
    require(w.q&&!w.column&&source&&rows>=HYBRID_K510&&cols>0&&cols<=4096&&
            w.rows==(uint32_t)rows&&w.cols==(uint32_t)cols&&fegetround()==FE_TONEAREST,"hybrid_contract510");
    double start=cost_enabled&&cost_profile?cost_clock():0.;
    int selected[HYBRID_K510],used=0;uint64_t comparisons=0;
    /* Ascending row traversal makes equal scores retain ascending original IDs. */
    for(int row=0;row<rows;row++){
        require(isfinite(y[row]),"hybrid_I8_finite510");int pos=used;
        for(int j=0;j<used;j++){comparisons++;if(y[row]>y[selected[j]]){pos=j;break;}}
        if(pos<HYBRID_K510){
            int last=used<HYBRID_K510?used:HYBRID_K510-1;
            for(int j=last;j>pos;j--)selected[j]=selected[j-1];selected[pos]=row;
            if(used<HYBRID_K510)used++;
        }
    }
    require(used==HYBRID_K510,"hybrid_top8_count510");
    double scanned=cost_enabled&&cost_profile?cost_clock():0.;
    for(int j=0;j<HYBRID_K510;j++){
        int row=selected[j];double value=dot64(source+(size_t)row*cols,x,cols);
        float rounded=(float)value;require(isfinite(value)&&isfinite(rounded),"hybrid_source_finite510");
        hybrid_ids510[j]=row;hybrid_dots510[j]=value;y[row]=rounded;
    }
    if(cost_enabled){
        hybrid_counts510.calls++;hybrid_counts510.visited+=(uint64_t)rows;
        hybrid_counts510.comparisons+=comparisons;hybrid_counts510.source_rows+=HYBRID_K510;
        hybrid_counts510.source_MACs+=(uint64_t)HYBRID_K510*cols;
        hybrid_counts510.f32_bytes+=(uint64_t)HYBRID_K510*cols*4;
        cost_counts[cost_phase][cost_kind].f32+=(uint64_t)HYBRID_K510*cols*4;
        if(cost_profile){hybrid_counts510.scan_seconds+=scanned-start;hybrid_counts510.refine_seconds+=cost_clock()-scanned;}
    }
}
static void hybrid_head510(Matrix w,const float *source,const float *x,float *y,int rows,int cols){
    require(w.q&&!w.column&&source&&rows>=HYBRID_K510&&cols>0&&cols<=4096&&
            w.rows==(uint32_t)rows&&w.cols==(uint32_t)cols&&fegetround()==FE_TONEAREST,"hybrid_contract510");
    head_mv(w,x,y,rows,cols);hybrid_refine510(w,source,x,y,rows,cols);
}
static void hybrid_json510(void){
    HybridCounter510 c=hybrid_counts510;
    printf(",\"hybrid_head\":{\"calls\":%llu,\"visited\":%llu,\"comparisons\":%llu,\"source_rows\":%llu,\"source_MACs\":%llu,\"f32_bytes\":%llu,\"scan_seconds\":%.9f,\"refine_seconds\":%.9f}",
        (unsigned long long)c.calls,(unsigned long long)c.visited,(unsigned long long)c.comparisons,
        (unsigned long long)c.source_rows,(unsigned long long)c.source_MACs,(unsigned long long)c.f32_bytes,c.scan_seconds,c.refine_seconds);
}
