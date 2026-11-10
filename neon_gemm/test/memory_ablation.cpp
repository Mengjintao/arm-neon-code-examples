#include "../src/gemm.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <cstdint>
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
using Fn=void(*)(int,int,int,const float*,int,const float*,int,float*,int,float*,float*);
extern void gemm0(int,int,int,const float*,int,const float*,int,float*,int,float*,float*);
extern void gemm1(int,int,int,const float*,int,const float*,int,float*,int,float*,float*);
extern void gemm2(int,int,int,const float*,int,const float*,int,float*,int,float*,float*);
static double now(){timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec+t.tv_nsec*1e-9;}
struct Counts{uint64_t nr,enabled,running,cycles,instructions;};
static int event(uint64_t config,int leader){
    perf_event_attr a{};a.size=sizeof(a);a.type=PERF_TYPE_HARDWARE;a.config=config;
    a.disabled=leader<0;a.pinned=leader<0;a.exclude_kernel=1;a.exclude_hv=1;
    a.read_format=PERF_FORMAT_GROUP|PERF_FORMAT_TOTAL_TIME_ENABLED|PERF_FORMAT_TOTAL_TIME_RUNNING;
    int fd=syscall(SYS_perf_event_open,&a,0,-1,leader,0);
    if(fd<0){perror("PMU");exit(1);}return fd;
}
int main(){
    Fn fn[]={gemm0,gemm1,gemm2};
    const char* names[]={"baseline","no_AB_load","no_C_store"};
    constexpr int N=1024;
    auto alloc=[](size_t n){return (float*)aligned_alloc(64,n*4);};
    float *A=alloc(N*N),*B=alloc(N*N),*C=alloc(N*N),*pa=alloc(MC*KC),*pb=alloc(KC*N);
    if(!A||!B||!C||!pa||!pb)return 1;
    srand(1);for(int i=0;i<N*N;++i)A[i]=(float)rand()/RAND_MAX;
    for(int i=0;i<N*N;++i)B[i]=(float)rand()/RAND_MAX;
    int fd=event(PERF_COUNT_HW_CPU_CYCLES,-1),ins=event(PERF_COUNT_HW_INSTRUCTIONS,fd);
    double best[3]={1e9,1e9,1e9};Counts bestc[3]{};
    puts("CPU0 1024^3 all packing/prefetch retained. Ablations are NOT valid GEMM outputs. Time and PMU measured together.");
    for(int round=0;round<6;++round)for(int pos=0;pos<3;++pos){
        int v=(round%2 ? round+2-pos:round+pos)%3;
        for(int w=0;w<5;++w)fn[v](N,N,N,A,N,B,N,C,N,pa,pb);
        double local=1e9;
        for(int rep=0;rep<15;++rep){
            if(ioctl(fd,PERF_EVENT_IOC_RESET,PERF_IOC_FLAG_GROUP)||ioctl(fd,PERF_EVENT_IOC_ENABLE,PERF_IOC_FLAG_GROUP))return 1;
            double start=now();fn[v](N,N,N,A,N,B,N,C,N,pa,pb);double dt=now()-start;
            if(ioctl(fd,PERF_EVENT_IOC_DISABLE,PERF_IOC_FLAG_GROUP))return 1;
            Counts c{};if(read(fd,&c,sizeof(c))!=sizeof(c)||c.nr!=2||!c.running||c.enabled!=c.running)return 1;
            local=std::min(local,dt);if(dt<best[v]){best[v]=dt;bestc[v]=c;}
            printf("SAMPLE round=%d variant=%s rep=%d ms=%.6f cycles=%lu instructions=%lu IPC=%.6f\n",round+1,names[v],rep,dt*1e3,c.cycles,c.instructions,(double)c.instructions/c.cycles);
        }
        printf("ROUND round=%d variant=%s best_ms=%.6f\n",round+1,names[v],local*1e3);fflush(stdout);
    }
    fn[0](N,N,N,A,N,B,N,C,N,pa,pb);
    std::vector<double> ref(N*N,0);
    for(int i=0;i<N;++i)for(int k=0;k<N;++k)for(int j=0;j<N;++j)ref[i*N+j]+=(double)A[i*N+k]*B[k*N+j];
    double err=0;for(int i=0;i<N*N;++i){double e=fabs(C[i]-ref[i]);err=std::max(err,e);if(!(e<=1e-4*(1+fabs(ref[i]))))return 1;}
    printf("Baseline Check=PASS max_error=%.9g\n",err);
    fn[1](N,N,N,A,N,B,N,C,N,pa,pb);
    for(int i=0;i<N*N;++i)if(!std::isfinite(C[i])||C[i]==0)return 1;
    puts("No-load output sanity=PASS (not GEMM correctness)");
    fn[2](N,N,N,A,N,B,N,C,N,pa,pb);
    for(int i=0;i<N*N;++i)if(C[i]!=0)return 1;
    puts("No-store output remains caller-zeroed=PASS (not GEMM correctness)");
    for(int v=0;v<3;++v)printf("FINAL variant=%s best_ms=%.6f delta_percent=%.4f cycles=%lu instructions=%lu IPC=%.6f\n",names[v],best[v]*1e3,100*(best[0]-best[v])/best[0],bestc[v].cycles,bestc[v].instructions,(double)bestc[v].instructions/bestc[v].cycles);
    close(ins);close(fd);free(A);free(B);free(C);free(pa);free(pb);
}
