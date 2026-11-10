// kp920 tuning: hot packed-panel PMU probes and full 1024^3 GEMM screening.
// Build with -O3 -std=c++17, src/pack_v.cpp, src/neon_4x16_2unroll_pipeline_2buffer.cpp (neon_4x16)
// and src/neon_4x16_2unroll_pipeline_2buffer_prfm.cpp (neon_4x16_next_AB, called by the included main.cpp).
// micro: K scan; probe MC NC KC: packed reuse; [rounds]: stdin MC NC KC triples.
// Full GEMM includes packing and C initialization, jc -> pc -> ic -> jr -> ir.
// Runtime parameters screen candidates; confirm finalists with constant blocks.
#define main original_example_main
#include "../src/main.cpp"
#undef main
#include <vector>
#include <sched.h>

void blocked(int BM, int BN, int BK, int M, int N, int K, const float* A, int lda,
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

#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <cstdint>
#include <random>
#include <string>
struct Counts { uint64_t nr, enabled, running, cycles, l1refill, l2access, l2refill; };

static int event(int type, int config, int leader) {
    perf_event_attr a{};
    a.size=sizeof(a); a.type=type; a.config=config;
    a.disabled=leader<0; a.pinned=leader<0;
    a.exclude_kernel=1; a.exclude_hv=1;
    a.read_format=PERF_FORMAT_GROUP|PERF_FORMAT_TOTAL_TIME_ENABLED|PERF_FORMAT_TOTAL_TIME_RUNNING;
    return syscall(SYS_perf_event_open,&a,0,-1,leader,0);
}


struct PMU {
    int fd, members[3];
    PMU() {
        fd=event(PERF_TYPE_HARDWARE,PERF_COUNT_HW_CPU_CYCLES,-1);
        int codes[]={0x03,0x16,0x17};
        if(fd<0) {perror("cycles");exit(1);}
        for(int i=0;i<3;++i) {
            members[i]=event(PERF_TYPE_RAW,codes[i],fd);
            if(members[i]<0) {perror("PMU");exit(1);}
        }
    }
    void start() {
        if(ioctl(fd,PERF_EVENT_IOC_RESET,PERF_IOC_FLAG_GROUP)||ioctl(fd,PERF_EVENT_IOC_ENABLE,PERF_IOC_FLAG_GROUP)) exit(1);
    }
    Counts stop() {
        if(ioctl(fd,PERF_EVENT_IOC_DISABLE,PERF_IOC_FLAG_GROUP)) exit(1);
        Counts c{};
        if(read(fd,&c,sizeof(c))!=sizeof(c)||c.nr!=4||!c.running||c.running!=c.enabled) {fprintf(stderr,"invalid PMU group\n");exit(1);}
        return c;
    }
    ~PMU() {for(int f:members) close(f);close(fd);}
};
static double median(std::vector<double> v) {std::sort(v.begin(),v.end());return v[v.size()/2];}
static void packed(int m,int n,int k,int passes,const float *a,const float *b,float *c) {
    for(int p=0;p<passes;++p)
        for(int j=0;j<n;j+=NR)
            for(int i=0;i<m;i+=MR)
                neon_4x16(k,a+i*k,b+j*k,c+i*n+j,n);
}
static void probe(int m,int n,int k,PMU &pmu) {
    std::vector<float> a(m*k),b(n*k),c(m*n,0);
    for(size_t i=0;i<a.size();++i) a[i]=(int(i%17)-8)/32.0f;
    for(size_t i=0;i<b.size();++i) b[i]=(int(i%13)-6)/32.0f;
    packed(m,n,k,1,a.data(),b.data(),c.data());
    for(int p=0;p<64;++p) {
        int i=p*(m-1)/63,j=p*37%n;double ref=0;
        for(int l=0;l<k;++l) ref+=double(a[(i/4)*4*k+l*4+i%4])*b[(j/16)*16*k+l*16+j%16];
        if(c[i*n+j]!=ref) {fprintf(stderr,"probe check failed\n");exit(1);}
    }
    int passes=std::max(2,524288/((m/4)*(n/16)*k));
    double ks=double(passes)*(m/4)*(n/16)*k;
    std::vector<double> cy,l1,l2,ac;
    for(int r=0;r<5;++r) {
        std::fill(c.begin(),c.end(),0);
        packed(m,n,k,8,a.data(),b.data(),c.data());
        pmu.start();packed(m,n,k,passes,a.data(),b.data(),c.data());auto v=pmu.stop();
        cy.push_back(v.cycles/ks);l1.push_back(v.l1refill/ks);l2.push_back(v.l2refill/ks);ac.push_back(v.l2access/ks);
    }
    printf("PROBE MC=%d NC=%d KC=%d input_KiB=%.2f cycles/K=%.5f L1refill/K=%.6f L2access/K=%.6f L2refill/K=%.6f Check=PASS\n",m,n,k,(m+n)*k/256.0,median(cy),median(l1),median(ac),median(l2));
}
struct Config {int m,n,k;std::vector<double> times;};
int main(int argc,char **argv) {
    cpu_set_t cpus;CPU_ZERO(&cpus);CPU_SET(0,&cpus);
    if(sched_setaffinity(0,sizeof(cpus),&cpus)) {perror("affinity");return 1;}
    setvbuf(stdout,nullptr,_IOLBF,0);
    if(argc==2 && std::string(argv[1])=="micro") {
        PMU pmu;
        for(int k=128;k<=4096;k+=64) probe(4,16,k,pmu);
        for(int k=4352;k<=8192;k+=256) probe(4,16,k,pmu);
        return 0;
    }
    if(argc==5 && std::string(argv[1])=="probe") {
        PMU pmu;probe(atoi(argv[2]),atoi(argv[3]),atoi(argv[4]),pmu);return 0;
    }
    // Read candidate triples from stdin; all use the same runtime blocking driver.
    int rounds=argc==2?atoi(argv[1]):5;
    std::vector<Config> configs;int m,n,k;
    while(scanf("%d %d %d",&m,&n,&k)==3) configs.push_back({m,n,k,{}});
    const int N=1024;
    std::vector<float> a(N*N),b(N*N),c(N*N),pa(1024*1024),pb(1024*1024);
    srand(123);for(float &x:a)x=float(rand())/RAND_MAX-0.5f;for(float &x:b)x=float(rand())/RAND_MAX-0.5f;
    auto run=[&](Config &cfg) {blocked(cfg.m,cfg.n,cfg.k,N,N,N,a.data(),N,b.data(),N,c.data(),N,pa.data(),pb.data());};
    for(auto &cfg:configs) run(cfg);
    std::mt19937 rng(123);
    std::vector<int> order(configs.size());for(size_t i=0;i<order.size();++i)order[i]=i;
    for(int r=0;r<rounds;++r) {
        std::shuffle(order.begin(),order.end(),rng);
        for(int idx:order) {
            auto &cfg=configs[idx];double t=gettime();run(cfg);t=gettime()-t;cfg.times.push_back(t);
            printf("SAMPLE MC=%d NC=%d KC=%d round=%d seconds=%.9f\n",cfg.m,cfg.n,cfg.k,r,t);
            if(r==0) for(int p=0;p<128;++p) {
                int i=p*101%N,j=p*37%N;double ref=0;
                for(int l=0;l<N;++l) ref+=double(a[i*N+l])*b[l*N+j];
                if(!(std::abs(c[i*N+j]-ref)<=1e-4*(1+std::abs(ref)))) {fprintf(stderr,"GEMM check failed\n");return 1;}
            }
        }
    }
    std::sort(configs.begin(),configs.end(),[](const Config &a,const Config &b){return median(a.times)<median(b.times);});
    for(auto &cfg:configs) printf("GEMM MC=%d NC=%d KC=%d median_GFLOPS=%.5f min_GFLOPS=%.5f max_GFLOPS=%.5f rounds=%d Check=PASS\n",cfg.m,cfg.n,cfg.k,2.0*N*N*N/median(cfg.times)/1e9,2.0*N*N*N/(*std::max_element(cfg.times.begin(),cfg.times.end()))/1e9,2.0*N*N*N/(*std::min_element(cfg.times.begin(),cfg.times.end()))/1e9,rounds);
}
