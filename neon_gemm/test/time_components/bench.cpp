#include "../../src/gemm.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <random>
#include <cstdint>
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
using Fn=void(*)(int,int,int,const float*,int,const float*,int,float*,int,float*,float*);
#define DECL(n) extern void gemm##n(int,int,int,const float*,int,const float*,int,float*,int,float*,float*);
DECL(0) DECL(1) DECL(2) DECL(3) DECL(4) DECL(5) DECL(6) DECL(7)
struct Counts{uint64_t nr,enabled,running,cycles,instructions;};
static int event(uint64_t config,int leader){perf_event_attr a{};a.size=sizeof(a);a.type=PERF_TYPE_HARDWARE;a.config=config;a.disabled=leader<0;a.pinned=leader<0;a.exclude_kernel=1;a.exclude_hv=1;a.read_format=PERF_FORMAT_GROUP|PERF_FORMAT_TOTAL_TIME_ENABLED|PERF_FORMAT_TOTAL_TIME_RUNNING;int fd=syscall(SYS_perf_event_open,&a,0,-1,leader,0);if(fd<0){perror("PMU");exit(1);}return fd;}
static double now(){timespec t;clock_gettime(CLOCK_MONOTONIC_RAW,&t);return t.tv_sec+t.tv_nsec*1e-9;}
struct Sample{double ms;uint64_t cycles,instructions;};
int main(int argc,char**argv){
 const int seed=argc>1?atoi(argv[1]):1;
 const int variants=argc>2?atoi(argv[2]):8;
 Fn fn[]={gemm0,gemm1,gemm2,gemm3,gemm4,gemm5,gemm6,gemm7};
 const char* names[]={"baseline","full_slots_control","offline_A","offline_B","offline_AB","no_C_zero","no_C_store","no_AB_load"};
 constexpr int N=1024;
 auto alloc=[](size_t n){return (float*)aligned_alloc(64,(n*4+63)/64*64);};
 float *a=alloc(N*N),*b=alloc(N*N),*c=alloc(N*N),*pa=alloc((size_t)((N+KC-1)/KC)*((N+MC-1)/MC)*MC*KC),*pb=alloc((size_t)((N+KC-1)/KC)*KC*N);
 srand(1);for(int i=0;i<N*N;++i)a[i]=(float)rand()/RAND_MAX;for(int i=0;i<N*N;++i)b[i]=(float)rand()/RAND_MAX;
 auto prepare=[&](){for(int pc=0;pc<N;pc+=KC){int kc=std::min(KC,N-pc);pack_B(kc,N,b+(size_t)pc*N,N,pb+(size_t)(pc/KC)*KC*N);for(int ic=0;ic<N;ic+=MC)pack_A(std::min(MC,N-ic),kc,a+(size_t)ic*N+pc,N,pa+((size_t)(pc/KC)*((N+MC-1)/MC)+ic/MC)*MC*KC);}};
 // All packs outside timing for cached variants, and before every case for equal setup.
 std::vector<double> ref(N*N,0);
 for(int i=0;i<N;++i)for(int k=0;k<N;++k)for(int j=0;j<N;++j)ref[i*N+j]+=(double)a[i*N+k]*b[k*N+j];
 for(int v=0;v<5;++v){prepare();fn[v](N,N,N,a,N,b,N,c,N,pa,pb);double err=0;for(int i=0;i<N*N;++i){double e=std::abs(c[i]-ref[i]);err=std::max(err,e);if(!(e<=1e-4*(1+std::abs(ref[i])))){printf("FAIL %s\n",names[v]);return 1;}}printf("Check PASS %s max_error=%g\n",names[v],err);}
 puts("no_C_zero/no_C_store/no_AB_load: mathematical correctness intentionally not checked.");
 int fd=event(PERF_COUNT_HW_CPU_CYCLES,-1),ins=event(PERF_COUNT_HW_INSTRUCTIONS,fd);
 std::mt19937 rng(seed);std::vector<int> order;for(int v=0;v<variants;++v)order.push_back(v);
 std::vector<Sample> samples[8];
 for(int r=0;r<15;++r){std::shuffle(order.begin(),order.end(),rng);for(int v:order){prepare();std::fill_n(c,N*N,0.0f); // finite starting values outside timing, never per-K clearing for mode 5
  for(int w=0;w<2;++w)fn[v](N,N,N,a,N,b,N,c,N,pa,pb);
  if(ioctl(fd,PERF_EVENT_IOC_RESET,PERF_IOC_FLAG_GROUP)||ioctl(fd,PERF_EVENT_IOC_ENABLE,PERF_IOC_FLAG_GROUP))return 2;
  double t=now();fn[v](N,N,N,a,N,b,N,c,N,pa,pb);double ms=(now()-t)*1000;
  if(ioctl(fd,PERF_EVENT_IOC_DISABLE,PERF_IOC_FLAG_GROUP))return 2;
  Counts ct{};if(read(fd,&ct,sizeof(ct))!=sizeof(ct)||ct.nr!=2||!ct.running||ct.running!=ct.enabled){puts("PMU multiplex/error");return 3;}
  samples[v].push_back({ms,ct.cycles,ct.instructions});
  printf("SAMPLE seed=%d round=%d variant=%s ms=%.6f cycles=%lu instructions=%lu\n",seed,r+1,names[v],ms,ct.cycles,ct.instructions);
 }fflush(stdout);}
 for(int v=0;v<variants;++v){auto t=samples[v];std::sort(t.begin(),t.end(),[](auto x,auto y){return x.ms<y.ms;});std::vector<uint64_t> cyc;for(auto x:t)cyc.push_back(x.cycles);std::sort(cyc.begin(),cyc.end());printf("RESULT seed=%d variant=%s median_ms=%.6f median_cycles=%lu time_median_sample_cycles=%lu instructions=%lu min_ms=%.6f max_ms=%.6f\n",seed,names[v],t[7].ms,cyc[7],t[7].cycles,t[7].instructions,t[0].ms,t[14].ms);}
 close(ins);close(fd);free(a);free(b);free(c);free(pa);free(pb);
}
