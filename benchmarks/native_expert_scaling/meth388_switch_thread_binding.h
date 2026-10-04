/* Local single-group Windows worker placement; all model arithmetic unchanged. */
typedef struct {DWORD tid;DWORD_PTR mask;WORD group;int slot;} WorkerBinding;
static DWORD_PTR worker_masks[6];
static WorkerBinding worker_latest[6],worker_events[1024];
static volatile LONG worker_event_count;
static int worker_enabled,worker_requested,worker_physical_cores,worker_affinity_fault;
static _Thread_local int worker_bound_slot=-1;

static void worker_bind(void){
    if(!worker_enabled)return;
    int slot=omp_get_thread_num();require(slot>=0&&slot<worker_requested,"worker_slot");
    if(worker_bound_slot==slot)return;
    DWORD_PTR mask=worker_masks[slot];
    DWORD_PTR applied=worker_affinity_fault&&slot==1?worker_masks[0]:mask;
    require(SetThreadAffinityMask(GetCurrentThread(),applied)!=0,"worker_set_affinity");
    GROUP_AFFINITY actual;memset(&actual,0,sizeof(actual));
    require(GetThreadGroupAffinity(GetCurrentThread(),&actual)&&actual.Group==0&&actual.Mask==mask,"worker_affinity_readback");
    WorkerBinding binding={GetCurrentThreadId(),actual.Mask,actual.Group,slot};
    LONG index=InterlockedIncrement(&worker_event_count)-1;
    require(index>=0&&index<1024,"worker_event_bound");
    worker_events[index]=binding;worker_latest[slot]=binding;worker_bound_slot=slot;
}

static void worker_setup(int threads){
    require(threads==1||threads==3||threads==6,"worker_threads");
    DWORD_PTR permitted,system;require(GetProcessAffinityMask(GetCurrentProcess(),&permitted,&system),"worker_process_mask");
    DWORD bytes=0;GetLogicalProcessorInformationEx(RelationProcessorCore,NULL,&bytes);
    require(bytes>0&&GetLastError()==ERROR_INSUFFICIENT_BUFFER,"worker_topology_size");
    unsigned char *buffer=alloc(bytes,1);
    require(GetLogicalProcessorInformationEx(RelationProcessorCore,(PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)buffer,&bytes),"worker_topology");
    DWORD_PTR candidates[64];int count=0;DWORD offset=0;
    while(offset<bytes){
        PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX item=(PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX)(buffer+offset);
        require(item->Size>0&&offset+item->Size<=bytes&&item->Relationship==RelationProcessorCore,"worker_topology_record");
        require(item->Processor.GroupCount==1&&item->Processor.GroupMask[0].Group==0,"worker_single_group_only");
        worker_physical_cores++;DWORD_PTR allowed=item->Processor.GroupMask[0].Mask&permitted;
        if(allowed){require(count<64,"worker_core_bound");candidates[count++]=allowed&((DWORD_PTR)0-allowed);}
        offset+=item->Size;
    }
    free(buffer);require(offset==bytes&&count>=threads,"worker_allowed_physical_cores");
    for(int i=0;i<count;i++)for(int j=i+1;j<count;j++)if(candidates[j]<candidates[i]){DWORD_PTR old=candidates[i];candidates[i]=candidates[j];candidates[j]=old;}
    DWORD_PTR pool=0;for(int i=0;i<threads;i++){worker_masks[i]=candidates[i];pool|=candidates[i];}
    require(SetProcessAffinityMask(GetCurrentProcess(),pool),"worker_set_process_pool");
    DWORD_PTR readback;require(GetProcessAffinityMask(GetCurrentProcess(),&readback,&system)&&readback==pool,"worker_process_pool_readback");
    worker_requested=threads;worker_enabled=1;omp_set_dynamic(0);omp_set_num_threads(threads);
    const char *fault_value=getenv("SILICON_WORKER_BINDING_FAULT");
    require(!fault_value||!strcmp(fault_value,"1"),"worker_fault_value");worker_affinity_fault=fault_value!=NULL;
    #pragma omp parallel num_threads(threads)
    {require(omp_get_num_threads()==threads,"worker_team_size");worker_bind();}
}

static void worker_json(void){
    require(worker_enabled&&worker_event_count>=worker_requested,"worker_initialized");
    printf(",\"worker_physical_cores\":%d,\"worker_affinity\":[",worker_physical_cores);
    for(int slot=0;slot<worker_requested;slot++){
        WorkerBinding binding=worker_latest[slot];require(binding.tid&&binding.slot==slot&&binding.mask==worker_masks[slot],"worker_latest_identity");
        HANDLE handle=OpenThread(THREAD_QUERY_LIMITED_INFORMATION,FALSE,binding.tid);require(handle!=NULL,"worker_open_thread");
        GROUP_AFFINITY actual;memset(&actual,0,sizeof(actual));int ok=GetThreadGroupAffinity(handle,&actual);CloseHandle(handle);
        require(ok&&actual.Group==binding.group&&actual.Mask==binding.mask,"worker_end_affinity_readback");
        printf("%s{\"slot\":%d,\"windows_thread_id\":%lu,\"group\":%u,\"actual_mask\":%llu}",slot?",":"",slot,(unsigned long)binding.tid,(unsigned)actual.Group,(unsigned long long)actual.Mask);
    }
    printf("],\"worker_binding_events\":[");
    for(LONG index=0;index<worker_event_count;index++){
        WorkerBinding binding=worker_events[index];require(binding.mask==worker_masks[binding.slot]&&binding.group==0,"worker_event_identity");
        printf("%s{\"slot\":%d,\"windows_thread_id\":%lu,\"group\":%u,\"actual_mask\":%llu}",index?",":"",binding.slot,(unsigned long)binding.tid,(unsigned)binding.group,(unsigned long long)binding.mask);
    }
    printf("]");
}
