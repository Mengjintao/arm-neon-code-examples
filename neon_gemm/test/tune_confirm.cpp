// Final 1024^3 comparison: constant blocks, production-sized malloc workspaces.
// Build with src/pack_v.cpp, src/neon_4x16_2unroll_pipeline_2buffer.cpp (neon_4x16)
// and src/neon_4x16_2unroll_pipeline_2buffer_prfm.cpp (neon_4x16_next_AB, called by the included main.cpp).
// Include existing implementation only to retain the exact baseline.
#define main original_example_main
#include "../src/main.cpp"
#undef main
#include <vector>
#include <sched.h>

template<int BM, int BN, int BK>
void blocked(int M, int N, int K, const float* A, int lda,
          const float* B, int ldb, float* C, int ldc,
          float* packed_A, float* packed_B)
{
    for (int jc = 0; jc < N; jc += BN) {
        const int nc = std::min(BN, N - jc);
        for (int pc = 0; pc < K; pc += BK) {
            const int kc = std::min(BK, K - pc);
            pack_B(kc, nc, B + (size_t)(pc) * ldb + jc, ldb, packed_B);
            for (int ic = 0; ic < M; ic += BM) {
                const int mc = std::min(BM, M - ic);
                pack_A(mc, kc, A + (size_t)(ic) * lda + pc, lda, packed_A);
                for (int jr = 0; jr < nc; jr += NR) {
                    const int nr = std::min(NR, nc - jr);
                    const float* bp = packed_B + jr * kc;
                    for (int ir = 0; ir < mc; ir += MR) {
                        const int mr = std::min(MR, mc - ir);
                        const float* ap = packed_A + ir * kc;
                        float* cp = C + (size_t)(ic + ir) * ldc + jc + jr;
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

using Gemm = decltype(&gemm);
struct Config { const char *name; int mc,nc,kc; Gemm fn; };
static Config configs[] = {
    {"jr_ir",20,352,256,blocked<20,352,256>},
    {"jr_ir",20,256,256,blocked<20,256,256>},
    {"jr_ir",24,256,224,blocked<24,256,224>},
    {"jr_ir",24,256,256,blocked<24,256,256>},
    {"jr_ir",16,256,256,blocked<16,256,256>},
    {"jr_ir",64,1024,512,blocked<64,1024,512>},
    {"jr_ir",64,1024,384,blocked<64,1024,384>},
    {"jr_ir",64,160,256,blocked<64,160,256>},
    {"jr_ir",96,128,256,blocked<96,128,256>},
    {"jr_ir",128,256,256,blocked<128,256,256>},
};
constexpr int CONFIG_COUNT = sizeof(configs) / sizeof(configs[0]);

static bool check(const Config &cfg, float *pa, float *pb) {
    const int M=529,N=273,K=517,lda=520,ldb=280,ldc=280;
    std::vector<float> A(M*lda),B(K*ldb),C(M*ldc,-999),ref(M*N,0);
    std::srand(1);
    for(float &v:A) v=(std::rand()%17-8)/32.0f;
    for(float &v:B) v=(std::rand()%17-8)/32.0f;
    for(int i=0;i<M;++i)
        for(int k=0;k<K;++k)
            for(int j=0;j<N;++j) ref[i*N+j]+=A[i*lda+k]*B[k*ldb+j];
    // Existing baseline clears only valid C entries too.
    cfg.fn(M,N,K,A.data(),lda,B.data(),ldb,C.data(),ldc,pa,pb);
    for(int i=0;i<M;++i) {
        for(int j=0;j<N;++j) if(C[i*ldc+j]!=ref[i*N+j]) return false;
        for(int j=N;j<ldc;++j) if(C[i*ldc+j]!=-999) return false;
    }
    return true;
}

int main() {
    cpu_set_t cpus; CPU_ZERO(&cpus);CPU_SET(0,&cpus);
    if(sched_setaffinity(0,sizeof(cpus),&cpus)) {std::perror("affinity");return 1;}
    setvbuf(stdout,nullptr,_IOLBF,0);
    std::vector<float> pa(512*384),pb(1024*512);
    for(const Config &cfg:configs) {
        bool ok=check(cfg,pa.data(),pb.data());
        std::printf("Check=%s MC=%d NC=%d KC=%d order=%s\n",ok?"PASS":"FAIL",cfg.mc,cfg.nc,cfg.kc,cfg.name);
        if(!ok) return 1;
    }
    int shapes[][3]={{1024,1024,1024}};
    std::printf("CPU=0; 1 warmup before each sample, 9 timed rounds, rotated config order; exact malloc workspace per config; time includes pack+C initialization+compute\n");
    for(auto &shape:shapes) {
        int M=shape[0],N=shape[1],K=shape[2];
        std::vector<float> A((size_t)M*K),B((size_t)K*N),C((size_t)M*N);
        std::srand(123);
        for(float &v:A) v=(float)std::rand()/RAND_MAX-0.5f;
        for(float &v:B) v=(float)std::rand()/RAND_MAX-0.5f;
        double times[CONFIG_COUNT][9]{};
        for(int round=0;round<9;++round)
            for(int pos=0;pos<CONFIG_COUNT;++pos) {
                int ci=(pos+round)%CONFIG_COUNT;
                const Config &cfg=configs[ci];
                float *wa=(float*)std::malloc(cfg.mc*cfg.kc*sizeof(float));
                float *wb=(float*)std::malloc(cfg.nc*cfg.kc*sizeof(float));
                if(!wa || !wb) return 1;
                cfg.fn(M,N,K,A.data(),K,B.data(),N,C.data(),N,wa,wb);
                double start=gettime();
                cfg.fn(M,N,K,A.data(),K,B.data(),N,C.data(),N,wa,wb);
                times[ci][round]=gettime()-start;
                std::free(wa);std::free(wb);
                std::printf("sample M=%d N=%d K=%d MC=%d NC=%d KC=%d order=%s round=%d seconds=%.9f GFLOPS=%.4f\n",
                            M,N,K,cfg.mc,cfg.nc,cfg.kc,cfg.name,round,times[ci][round],2.0*M*N*K/times[ci][round]/1e9);
                if(round==8) {
                    for(int p=0;p<64;++p) {
                        int i=(int)((size_t)p*(M-1)/63),j=(p*37)%N;
                        double ref=0;
                        for(int k=0;k<K;++k) ref+=(double)A[(size_t)i*K+k]*B[(size_t)k*N+j];
                        if(!(std::abs(C[(size_t)i*N+j]-ref)<=1e-4*(1+std::abs(ref)))) {
                            std::fprintf(stderr,"sample check failed\n");return 1;
                        }
                    }
                }
            }
        for(int ci=0;ci<CONFIG_COUNT;++ci) {
            auto &cfg=configs[ci];std::sort(times[ci],times[ci]+9);
            std::printf("SUMMARY M=%d N=%d K=%d MC=%d NC=%d KC=%d order=%s median_GFLOPS=%.4f min_GFLOPS=%.4f max_GFLOPS=%.4f Check=PASS\n",
                        M,N,K,cfg.mc,cfg.nc,cfg.kc,cfg.name,2.0*M*N*K/times[ci][4]/1e9,
                        2.0*M*N*K/times[ci][8]/1e9,2.0*M*N*K/times[ci][0]/1e9);
        }
    }
}
