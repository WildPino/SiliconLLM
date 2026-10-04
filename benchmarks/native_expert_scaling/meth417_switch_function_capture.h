/* Read-only final decoder bank observer; no model arithmetic or routing change. */
static FILE *function_capture_file;
static int function_capture_active;
static void function_capture_close(void){
    if(function_capture_file){FILE *f=function_capture_file;function_capture_file=NULL;require(fclose(f)==0,"function_capture_close");}
}
static void function_capture_write(const void *data,size_t size,size_t count){
    require(function_capture_file!=NULL&&fwrite(data,size,count,function_capture_file)==count,"function_capture_write");
}
static void function_capture_float(const float *data,int count){function_capture_write(data,4,(size_t)count);}
static void function_capture_quant(const float *data,int count){
    int16_t codes[4096];float scale=head_activation_codes(data,codes,count);
    function_capture_float(&scale,1);function_capture_write(codes,2,(size_t)count);
}
static void function_capture_begin(Config c,const float *hidden,int position,int id,int source_tokens){
    require(!function_capture_active&&c.d==768&&c.ff==3072&&c.dec==12&&c.vocab==32128&&(c.n==128||c.n==256),"function_capture_geometry");
    if(!function_capture_file){
        const char *path=getenv("SILICON_FUNCTION_CAPTURE_PATH");require(path&&path[0],"function_capture_path");
        function_capture_file=fopen(path,"wb");require(function_capture_file!=NULL,"function_capture_open");
        function_capture_write("SWFUN001",1,8);uint32_t dims[]={c.d,c.ff,c.vocab,c.n,c.dec-1,1};
        function_capture_write(dims,4,6);require(atexit(function_capture_close)==0,"function_capture_atexit");
    }
    function_capture_active=1;uint32_t record[]={position,id,source_tokens,route_count};
    function_capture_write(record,4,4);function_capture_float(hidden,c.d);
}
static void function_capture_input(Config c,const float *input){
    if(function_capture_active){function_capture_float(input,c.d);function_capture_quant(input,c.d);}
}
static void function_capture_up(Config c,const float *up){if(function_capture_active)function_capture_float(up,c.ff);}
static void function_capture_relu(Config c,const float *up){
    if(function_capture_active){function_capture_float(up,c.ff);function_capture_quant(up,c.ff);}
}
static void function_capture_down(Config c,const float *down){if(function_capture_active)function_capture_float(down,c.d);}
static void function_capture_post(Config c,const float *hidden){
    if(function_capture_active){
        Route r=routes[route_count-1];require(r.accepted==1,"function_capture_accepted");
        function_capture_write(&r.expert,4,1);function_capture_write(&r.accepted,4,1);function_capture_float(&r.probability,1);
        function_capture_float(hidden,c.d);
    }
}
static void function_capture_final(Config c,const float *normalized){
    require(function_capture_active,"function_capture_final_active");function_capture_float(normalized,c.d);
}
static void function_capture_finish(Config c,const float *head_input){
    require(function_capture_active,"function_capture_finish_active");function_capture_float(head_input,c.d);function_capture_quant(head_input,c.d);
    function_capture_active=0;
}
