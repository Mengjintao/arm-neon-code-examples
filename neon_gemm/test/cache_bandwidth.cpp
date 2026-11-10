#include <arm_neon.h>
#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>
#include <random>
#include <linux/perf_event.h>
#include <sched.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>

extern "C" void load_ldp_kernel(float*, int, int64_t);
static volatile float sink;
static double gettime() {
    timespec t; clock_gettime(CLOCK_MONOTONIC_RAW,&t);
    return t.tv_sec+t.tv_nsec*1e-9;
}
// Explicit NEON copy: report useful source reads + destination writes, not bus traffic.
__attribute__((noinline)) static void copy_stream(float* dst,const float* src,size_t n,int64_t passes) {
    for (int64_t p=0;p<passes;++p)
        for (size_t i=0;i<n;i+=32) {
            float32x4_t v0=vld1q_f32(src+i),v1=vld1q_f32(src+i+4);
            float32x4_t v2=vld1q_f32(src+i+8),v3=vld1q_f32(src+i+12);
            float32x4_t v4=vld1q_f32(src+i+16),v5=vld1q_f32(src+i+20);
            float32x4_t v6=vld1q_f32(src+i+24),v7=vld1q_f32(src+i+28);
            vst1q_f32(dst+i,v0); vst1q_f32(dst+i+4,v1);
            vst1q_f32(dst+i+8,v2); vst1q_f32(dst+i+12,v3);
            vst1q_f32(dst+i+16,v4); vst1q_f32(dst+i+20,v5);
            vst1q_f32(dst+i+24,v6); vst1q_f32(dst+i+28,v7);
        }
}
struct Counts {uint64_t nr,enabled,running,v[5];};
static int event(int code,int leader) {
    perf_event_attr a{};a.size=sizeof(a);a.type=PERF_TYPE_RAW;a.config=code;
    a.disabled=leader<0;a.pinned=leader<0;a.exclude_kernel=1;a.exclude_hv=1;
    a.read_format=PERF_FORMAT_GROUP|PERF_FORMAT_TOTAL_TIME_ENABLED|PERF_FORMAT_TOTAL_TIME_RUNNING;
    return syscall(SYS_perf_event_open,&a,0,-1,leader,0);
}
struct Sample {double gb,bpc,ghz,seconds,l1,l2,ll,miss;};
struct Case {size_t kib;bool copy;int64_t passes;std::vector<Sample> samples;};
int main(int argc, char**) {
    const bool confirm=argc>1;
    const int rounds=confirm?9:7;
    const double target=confirm?0.25:0.1;
    cpu_set_t set;CPU_ZERO(&set);CPU_SET(0,&set);
    if(sched_setaffinity(0,sizeof(set),&set)) {perror("affinity");return 1;}
    int fd[5];const int codes[]={0x11,0x03,0x17,0x36,0x37};
    for(int i=0;i<5;++i) {fd[i]=event(codes[i],i?fd[0]:-1);if(fd[i]<0){perror("PMU");return 1;}}
    constexpr size_t maxbytes=64ULL*1024*1024;
    float* data=(float*)std::aligned_alloc(4096,maxbytes);
    if(!data)return 1;
    for(size_t i=0;i<maxbytes/4;++i)data[i]=(i%251)/256.0f;
    std::vector<Case> cases;
    std::vector<size_t> sizes=confirm ? std::vector<size_t>{32,96,128,192,224,256,4096,8192,12288,16384,65536}
        : std::vector<size_t>{16,32,48,64,128,256,384,512,1024,2048,4096,8192,16384,32768,65536};
    for(size_t kib : sizes)
        for(bool copy : {false,true})cases.push_back({kib,copy,1,{}});
    auto run=[&](const Case& c,int64_t passes) {
        size_t bytes=c.kib*1024;
        if(c.copy)copy_stream(data+bytes/8,data,bytes/8,passes);
        else load_ldp_kernel(data,bytes/512,passes);
    };
    // Adapt repetitions to ~100 ms; no nominal clock is used for bytes/cycle.
    for(auto& c:cases) {
        run(c,4);
        if(c.copy && std::memcmp(data,data+c.kib*1024/8,c.kib*1024/2)) {
            fprintf(stderr,"copy check failed\n");return 1;
        }
        double dt;
        do {double t=gettime();run(c,c.passes);dt=gettime()-t;
            if(dt<0.01)c.passes*=2;
        }while(dt<0.01);
        c.passes=std::max<int64_t>(1,c.passes*target/dt);
    }
    std::mt19937 rng(123);
    std::vector<size_t> order(cases.size());for(size_t i=0;i<order.size();++i)order[i]=i;
    puts("CPU0; sequential warm streams; footprint=source+destination for copy; GB/s=decimal useful bytes; PMU excludes kernel; shuffled rounds");
    for(int round=0;round<rounds;++round) {
        std::shuffle(order.begin(),order.end(),rng);
        for(size_t index:order) {
            auto& c=cases[index];run(c,4);
            if(ioctl(fd[0],PERF_EVENT_IOC_RESET,PERF_IOC_FLAG_GROUP))return 1;
            const double start=gettime();
            if(ioctl(fd[0],PERF_EVENT_IOC_ENABLE,PERF_IOC_FLAG_GROUP))return 1;
            run(c,c.passes);
            if(ioctl(fd[0],PERF_EVENT_IOC_DISABLE,PERF_IOC_FLAG_GROUP))return 1;
            double sec=gettime()-start;
            Counts v{};
            if(read(fd[0],&v,sizeof(v))!=sizeof(v)||v.nr!=5||!v.running||v.enabled!=v.running) {
                fprintf(stderr,"PMU group invalid/multiplexed enabled=%lu running=%lu\n",v.enabled,v.running);return 1;
            }
            double bytes=(double)c.kib*1024*c.passes;
            Sample s{bytes/sec/1e9,bytes/v.v[0],v.v[0]/sec/1e9,sec,
                     v.v[1]/(bytes/1048576),v.v[2]/(bytes/1048576),v.v[3]/(bytes/1048576),v.v[4]/(bytes/1048576)};
            c.samples.push_back(s);
            printf("RAW %s KiB=%zu round=%d passes=%ld seconds=%.6f GB/s=%.3f B/cycle=%.3f GHz=%.4f L1refill/MiB=%.2f L2refill/MiB=%.2f LLread/MiB=%.2f LLmiss/MiB=%.2f\n",
                   c.copy?"copy":"read",c.kib,round,c.passes,sec,s.gb,s.bpc,s.ghz,s.l1,s.l2,s.ll,s.miss);
            fflush(stdout);
        }
    }
    for(auto& c:cases) {
        std::sort(c.samples.begin(),c.samples.end(),[](auto a,auto b){return a.gb<b.gb;});
        auto s=c.samples[rounds/2];
        printf("SUMMARY %s KiB=%zu GB/s=%.3f min=%.3f max=%.3f B/cycle=%.3f GHz=%.4f L1refill/MiB=%.2f L2refill/MiB=%.2f LLread/MiB=%.2f LLmiss/MiB=%.2f\n",
               c.copy?"copy":"read",c.kib,s.gb,c.samples[0].gb,c.samples.back().gb,s.bpc,s.ghz,s.l1,s.l2,s.ll,s.miss);
    }
    sink=data[0];std::free(data);for(int f:fd)close(f);
}
