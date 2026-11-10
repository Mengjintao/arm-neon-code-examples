#include "../src/gemm.h"
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

__attribute__((noinline)) static void run_gemm(int block_m, int M, int N, int K, const float* A, int lda,
          const float* B, int ldb, float* C, int ldc,
          float* packed_A, float* packed_B)
{
    for (int pc = 0; pc < K; pc += KC) {
        const int kc = std::min(KC, K - pc);
        pack_B(kc, N, B + (size_t)pc * ldb, ldb, packed_B);
        for (int ic = 0; ic < M; ic += block_m) {
            const int mc = std::min(block_m, M - ic);
            pack_A(mc, kc, A + (size_t)ic * lda + pc, lda, packed_A);
            for (int jc = 0; jc < N; jc += NC) {
                const int nc = std::min(NC, N - jc);
                for (int jr = 0; jr < nc; jr += NR) {
                    const int nr = std::min(NR, nc - jr);
                    const float* bp = packed_B + (size_t)(jc + jr) * kc;
                    for (int ir = 0; ir < mc; ir += MR) {
                        const int mr = std::min(MR, mc - ir);
                        const float* ap = packed_A + ir * kc;
                        float* cp = C + (size_t)(ic + ir) * ldc + jc + jr;
                        int next_ir = ir + MR;
                        int next_jr = jr;
                        if (next_ir >= mc) {
                            next_ir = 0;
                            next_jr += NR;
                        }
                        if (next_jr < nc) {
                            const int next_rows = std::min(MR, mc - next_ir);
                            for (int r = 0; r < next_rows; ++r)
                                __builtin_prefetch(C + (size_t)(ic + next_ir + r) * ldc
                                                   + jc + next_jr, 0, 3);
                        }
                        if (mr == MR && nr == NR) {
                            if (pc == 0) {
                                for (int r = 0; r < MR; ++r)
                                    std::fill_n(cp + (size_t)r * ldc, NR, 0.0f);
                            }
                            neon_4x16(kc, ap, bp, cp, ldc);
                        } else {
                            // Edge tiles use the same kernel; copy only valid C entries.
                            alignas(64) float edge[MR * NR]{};
                            if (pc != 0) {
                                for (int r = 0; r < mr; ++r)
                                    std::copy_n(cp + (size_t)(r) * ldc, nr, edge + r * NR);
                            }
                            neon_4x16(kc, ap, bp, edge, NR);
                            for (int r = 0; r < mr; ++r)
                                std::copy_n(edge + r * NR, nr, cp + (size_t)(r) * ldc);
                        }
                    }
                }
            }
        }
    }
}
static double gettime() {
    timespec ts;clock_gettime(CLOCK_MONOTONIC_RAW,&ts);return ts.tv_sec+ts.tv_nsec*1e-9;
}
static int event(int config,int leader) {
    perf_event_attr a{};a.size=sizeof(a);a.type=PERF_TYPE_RAW;a.config=config;
    a.disabled=leader<0;a.pinned=leader<0;a.exclude_kernel=1;a.exclude_hv=1;
    a.read_format=PERF_FORMAT_GROUP|PERF_FORMAT_TOTAL_TIME_ENABLED|PERF_FORMAT_TOTAL_TIME_RUNNING;
    return syscall(SYS_perf_event_open,&a,0,-1,leader,0);
}
struct Counts {uint64_t nr,enabled,running,v[4];};
struct Sample {double seconds;Counts counts;};
int main(int argc,char**argv) {
    constexpr int N=1024;
    const int seed=argc>1?std::atoi(argv[1]):1;
    const int choices[]={88,96,120,128,136,144,160,176};
    auto alloc=[](size_t n){return (float*)std::aligned_alloc(64,n*sizeof(float));};
    float *A=alloc(N*N),*B=alloc(N*N),*C=alloc(N*N),*pa=alloc(176*KC),*pb=alloc(N*KC);
    if(!A||!B||!C||!pa||!pb)return 1;
    std::srand(1);
    for(int i=0;i<N*N;++i){A[i]=(float)std::rand()/RAND_MAX;B[i]=(float)std::rand()/RAND_MAX;}
    std::vector<double> ref(N*N,0);
    for(int i=0;i<N;++i)for(int k=0;k<N;++k)for(int j=0;j<N;++j)
        ref[i*N+j]+=(double)A[i*N+k]*B[k*N+j];
    auto run=[&](int mc){run_gemm(mc,N,N,N,A,N,B,N,C,N,pa,pb);};
    for(int mc:choices) {
        std::fill_n(C,N*N,123.0f);run(mc);double error=0;
        for(int i=0;i<N*N;++i) {
            const double e=std::abs(C[i]-ref[i]);error=std::max(error,e);
            if(!std::isfinite(e)||e>1e-4*(1+std::abs(ref[i]))){printf("FAIL MC=%d\n",mc);return 1;}
        }
        printf("CHECK MC=%d PASS max_error=%.8g\n",mc,error);
        for(int w=0;w<3;++w)run(mc);
    }
    int fd[4];const int codes[]={0x11,0x03,0x16,0x17};
    for(int i=0;i<4;++i) {fd[i]=event(codes[i],i?fd[0]:-1);if(fd[i]<0){perror("perf");return 1;}}
    std::vector<Sample> samples[8];std::vector<int> order={0,1,2,3,4,5,6,7};std::mt19937 rng(seed);
    printf("1024^3 NC48 KC256; same runtime-MC function, same buffers, includes all pack/C init; order_seed=%d\n",seed);
    for(int r=0;r<15;++r) {
        std::shuffle(order.begin(),order.end(),rng);
        for(int index:order) {
            run(choices[index]); // Same immediate warmup for every measured call.
            if(ioctl(fd[0],PERF_EVENT_IOC_RESET,PERF_IOC_FLAG_GROUP))return 1;
            double start=gettime();
            if(ioctl(fd[0],PERF_EVENT_IOC_ENABLE,PERF_IOC_FLAG_GROUP))return 1;
            run(choices[index]);
            if(ioctl(fd[0],PERF_EVENT_IOC_DISABLE,PERF_IOC_FLAG_GROUP))return 1;
            double seconds=gettime()-start;Counts c{};
            if(read(fd[0],&c,sizeof(c))!=sizeof(c)||c.nr!=4||!c.running||c.enabled!=c.running)return 1;
            samples[index].push_back({seconds,c});
            printf("RAW MC=%d round=%d ms=%.5f cycles=%lu L1refill=%lu L2access=%lu L2refill=%lu\n",
                   choices[index],r,seconds*1e3,c.v[0],c.v[1],c.v[2],c.v[3]);
        }
    }
    for(int i=0;i<8;++i) {
        auto& v=samples[i];std::sort(v.begin(),v.end(),[](auto a,auto b){return a.seconds<b.seconds;});
        auto s=v[7];printf("SUMMARY MC=%d ms=%.5f GFLOPS=%.4f min_ms=%.5f max_ms=%.5f GHz=%.4f L1refill=%lu L2access=%lu L2refill=%lu\n",
            choices[i],s.seconds*1e3,2.0*N*N*N/s.seconds/1e9,v.front().seconds*1e3,v.back().seconds*1e3,
            s.counts.v[0]/s.seconds/1e9,s.counts.v[1],s.counts.v[2],s.counts.v[3]);
    }
    for(int f:fd)close(f);std::free(A);std::free(B);std::free(C);std::free(pa);std::free(pb);
}
