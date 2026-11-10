// kp920: full packed A[MC*KC] and B[KC*NC], jr -> ir, active C[MC*NC].
// PMU raw events match /sys/bus/event_source/devices/armv8_pmuv3_0/events.
// The reused C block is counted separately; no packing inside measurement.
#include "../src/gemm.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <linux/perf_event.h>
#include <sched.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>

struct Counts { uint64_t nr, enabled, running, cycles, l1refill, l2access, l2refill; };

static int event(int type, int config, int leader) {
    perf_event_attr a{};
    a.size=sizeof(a); a.type=type; a.config=config;
    a.disabled=leader<0; a.pinned=leader<0;
    a.exclude_kernel=1; a.exclude_hv=1;
    a.read_format=PERF_FORMAT_GROUP|PERF_FORMAT_TOTAL_TIME_ENABLED|PERF_FORMAT_TOTAL_TIME_RUNNING;
    return syscall(SYS_perf_event_open,&a,0,-1,leader,0);
}

static bool scratch_c = false;

static void run(int mc,int nc,int kc,int passes,const float *A,const float *B,float *C) {
    for(int p=0;p<passes;++p)
        for(int jr=0;jr<nc;jr+=NR)
            for(int ir=0;ir<mc/MR;++ir)
                neon_4x16(kc,A+ir*MR*kc,B+jr*kc,scratch_c ? C : C+ir*MR*nc+jr, scratch_c ? NR : nc);
}

int main(int argc,char **argv) {
    if(argc!=6) return 1;
    int mc=std::atoi(argv[1]),nc=std::atoi(argv[2]),kc=std::atoi(argv[3]),gap=std::atoi(argv[4]);
    if(mc<=0 || mc%MR || nc<=0 || nc%NR || kc<=0 || gap<0 || gap%64) return 1;
    scratch_c=std::atoi(argv[5])!=0;
    cpu_set_t cpus; CPU_ZERO(&cpus); CPU_SET(0,&cpus);
    if(sched_setaffinity(0,sizeof(cpus),&cpus)) {std::perror("affinity");return 1;}
    size_t bsize=nc*kc,asize=mc*kc,csize=scratch_c ? MR*NR : mc*nc;
    size_t bytes=(bsize+asize+csize)*4+gap+128;
    float *memory=(float*)std::aligned_alloc(64,(bytes+63)/64*64);
    if(!memory) return 1;
    float *B=memory,*A=B+bsize+gap/4,*C=A+asize+32;
    std::srand(1);
    for(size_t i=0;i<bsize;++i) B[i]=(std::rand()%17-8)/32.0f;
    for(size_t i=0;i<asize;++i) A[i]=(std::rand()%17-8)/32.0f;
    std::fill_n(C,csize,0.0f);
    if (scratch_c) neon_4x16(kc,A,B,C,NR);
    else run(mc,nc,kc,1,A,B,C);
    for(int sample=0;sample<128;++sample) {
        int i=scratch_c ? sample%MR : sample*(mc-1)/127;
        int j=scratch_c ? (sample/MR)%NR : (sample*37)%nc;
        double ref=0;
        for(int k=0;k<kc;++k)
            ref+=(double)A[(i/MR)*MR*kc+k*MR+i%MR]*B[(j/NR)*kc*NR+k*NR+j%NR];
        if(!(std::abs(C[i*(scratch_c ? NR : nc)+j]-ref)<=1e-4*(1+std::abs(ref)))) {
            std::fprintf(stderr,"correctness failed\n");return 1;
        }
    }
    int fd=event(PERF_TYPE_HARDWARE,PERF_COUNT_HW_CPU_CYCLES,-1);
    if(fd<0) {std::perror("cycles");return 1;}
    int members[3]; int codes[3]={0x03,0x16,0x17};
    for(int i=0;i<3;++i) {
        members[i]=event(PERF_TYPE_RAW,codes[i],fd);
        if(members[i]<0) {std::perror("cache event");return 1;}
    }
    const int passes=std::max(2,1048576/((mc/MR)*(nc/NR)*kc));
    const double ks=(double)passes*(mc/MR)*(nc/NR)*kc;
    Counts samples[7];
    for(int rep=0;rep<7;++rep) {
        std::fill_n(C,csize,0.0f);
        run(mc,nc,kc,4,A,B,C);
        if(ioctl(fd,PERF_EVENT_IOC_RESET,PERF_IOC_FLAG_GROUP) ||
           ioctl(fd,PERF_EVENT_IOC_ENABLE,PERF_IOC_FLAG_GROUP)) return 1;
        run(mc,nc,kc,passes,A,B,C);
        if(ioctl(fd,PERF_EVENT_IOC_DISABLE,PERF_IOC_FLAG_GROUP)) return 1;
        Counts &v=samples[rep];
        if(read(fd,&v,sizeof(v))!=sizeof(v)||v.nr!=4||!v.running||v.enabled!=v.running) {
            std::fprintf(stderr,"invalid PMU group\n");return 1;
        }
        std::printf("sample mc=%d nc=%d kc=%d gap=%d rep=%d cycles/K=%.5f L1refill/K=%.6f L2access/K=%.6f L2refill/K=%.6f\n",
                    mc,nc,kc,gap,rep,v.cycles/ks,v.l1refill/ks,v.l2access/ks,v.l2refill/ks);
    }
    // Report separate medians, not counters chosen from the fastest run.
    auto median=[&](int field) {
        double values[7];
        for(int i=0;i<7;++i) {
            const Counts &v=samples[i];
            values[i]=(field==0?v.cycles:field==1?v.l1refill:field==2?v.l2access:v.l2refill)/ks;
        }
        std::sort(values,values+7);return values[3];
    };
    std::printf("SUMMARY scratch=%d mc=%d nc=%d kc=%d gap=%d input_KiB=%.3f C_KiB=%.3f cycles/K=%.5f L1refill/K=%.6f L2access/K=%.6f L2refill/K=%.6f Check=PASS\n",
                scratch_c,mc,nc,kc,gap,(bsize+asize)/256.0,csize/256.0,median(0),median(1),median(2),median(3));
    for(int e:members) close(e);
    close(fd);std::free(memory);
}
