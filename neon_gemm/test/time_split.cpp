// 时间组成（第 29 页）：最终版 src/main.cpp，1024³，CPU0。
// 同一个程序里链接 5 个版本的 gemm，只换微内核（见 prepare_kernels.py）；
// 每个版本的 pack_A / pack_B 外面都包一层计时，周期数用 PMU（用户态 cycles）。
#include "../src/gemm.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <linux/perf_event.h>
#include <random>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <vector>

using GemmFn = void (*)(int, int, int, const float*, int, const float*, int, float*, int, float*, float*);
#define DECL(v) void gemm_##v(int, int, int, const float*, int, const float*, int, float*, int, float*, float*);
DECL(full) DECL(ref) DECL(noload) DECL(nostore) DECL(pure) DECL(empty)

static double now_s()
{
    timespec t{};
    clock_gettime(CLOCK_MONOTONIC_RAW, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

static double pack_a_s = 0, pack_b_s = 0;
void timed_pack_A(int M, int K, const float* A, int lda, float* packed)
{
    const double b = now_s();
    pack_A(M, K, A, lda, packed);
    pack_a_s += now_s() - b;
}
void timed_pack_B(int K, int N, const float* B, int ldb, float* packed)
{
    const double b = now_s();
    pack_B(K, N, B, ldb, packed);
    pack_b_s += now_s() - b;
}

struct Counts { uint64_t nr, enabled, running, cycles, instructions; };
static int open_event(uint64_t config, int leader)
{
    perf_event_attr a{};
    a.size = sizeof(a);
    a.type = PERF_TYPE_HARDWARE;
    a.config = config;
    a.disabled = leader < 0;
    a.pinned = leader < 0;
    a.exclude_kernel = 1;
    a.exclude_hv = 1;
    a.read_format = PERF_FORMAT_GROUP | PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
    const int fd = syscall(SYS_perf_event_open, &a, 0, -1, leader, 0);
    if (fd < 0) { perror("perf_event_open"); std::exit(1); }
    return fd;
}

static double median(std::vector<double> v)
{
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

int main()
{
    const char* names[] = {"full", "ref", "noload", "nostore", "pure", "empty"};
    GemmFn fns[] = {gemm_full, gemm_ref, gemm_noload, gemm_nostore, gemm_pure, gemm_empty};
    const int V = 6, N = 1024, ROUNDS = 3, WARM = 3, REPS = 15;
    auto alloc = [](size_t n) { return (float*)aligned_alloc(64, (n * 4 + 63) / 64 * 64); };
    float *A = alloc(N * N), *B = alloc(N * N), *C = alloc(N * N), *pa = alloc(MC * KC), *pb = alloc(KC * N);
    std::srand(1);
    for (int i = 0; i < N * N; ++i) A[i] = (float)std::rand() / RAND_MAX;
    for (int i = 0; i < N * N; ++i) B[i] = (float)std::rand() / RAND_MAX;

    // 正确性：只有 full 是正确的 GEMM。
    gemm_full(N, N, N, A, N, B, N, C, N, pa, pb);
    std::vector<double> ref((size_t)N * N, 0.0);
    for (int i = 0; i < N; ++i)
        for (int k = 0; k < N; ++k)
            for (int j = 0; j < N; ++j) ref[(size_t)i * N + j] += (double)A[(size_t)i * N + k] * B[(size_t)k * N + j];
    double err = 0;
    for (size_t i = 0; i < (size_t)N * N; ++i) {
        const double e = std::abs(C[i] - ref[i]);
        err = std::max(err, e);
        if (!(e <= 1e-4 * (1.0 + std::abs(ref[i])))) { std::printf("CHECK full FAIL at %zu\n", i); return 1; }
    }
    std::printf("CHECK full PASS max_error=%.3g (other variants are not valid GEMM)\n", err);

    const int fd = open_event(PERF_COUNT_HW_CPU_CYCLES, -1);
    const int ins = open_event(PERF_COUNT_HW_INSTRUCTIONS, fd);
    // 每个指标：[版本][轮] 的中位数
    std::vector<std::vector<double>> ms(V), cyc(V), instr(V), pam(V), pbm(V);
    std::vector<int> order = {0, 1, 2, 3, 4, 5};
    for (int round = 1; round <= ROUNDS; ++round) {
        std::mt19937 rng(round);
        std::vector<std::vector<double>> t(V), c(V), in(V), a(V), b(V);
        for (int rep = 0; rep < REPS; ++rep) {
            std::shuffle(order.begin(), order.end(), rng);
            for (int v : order) {
                for (int w = 0; w < WARM; ++w) fns[v](N, N, N, A, N, B, N, C, N, pa, pb);
                pack_a_s = pack_b_s = 0;
                ioctl(fd, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP);
                ioctl(fd, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP);
                const double s = now_s();
                fns[v](N, N, N, A, N, B, N, C, N, pa, pb);
                const double dt = now_s() - s;
                ioctl(fd, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP);
                Counts k{};
                if (read(fd, &k, sizeof(k)) != sizeof(k) || k.nr != 2 || !k.running || k.enabled != k.running) {
                    std::fprintf(stderr, "PMU multiplexed or not scheduled\n");
                    return 1;
                }
                t[v].push_back(dt * 1e3); c[v].push_back(k.cycles); in[v].push_back(k.instructions);
                a[v].push_back(pack_a_s * 1e3); b[v].push_back(pack_b_s * 1e3);
                std::printf("SAMPLE round=%d rep=%d variant=%s ms=%.4f cycles=%lu instructions=%lu packA_ms=%.4f packB_ms=%.4f\n",
                            round, rep, names[v], dt * 1e3, k.cycles, k.instructions, pack_a_s * 1e3, pack_b_s * 1e3);
            }
        }
        for (int v = 0; v < V; ++v) {
            ms[v].push_back(median(t[v])); cyc[v].push_back(median(c[v])); instr[v].push_back(median(in[v]));
            pam[v].push_back(median(a[v])); pbm[v].push_back(median(b[v]));
            std::printf("ROUND round=%d variant=%s median_ms=%.4f median_cycles=%.0f packA_ms=%.4f packB_ms=%.4f\n",
                        round, names[v], ms[v].back(), cyc[v].back(), pam[v].back(), pbm[v].back());
        }
        std::fflush(stdout);
    }

    const double ksteps = (double)(N / MR) * (N / NR) * N;  // 一次 GEMM 里微内核走的 k 步总数
    std::printf("\nRESULT (median of %d round medians; ideal = %.0f FMLA / 2 per cycle = %.0f cycles)\n",
                ROUNDS, ksteps * 16, ksteps * 8);
    double M_ms[6], M_c[6], M_a[6], M_b[6];
    for (int v = 0; v < V; ++v) {
        M_ms[v] = median(ms[v]); M_c[v] = median(cyc[v]); M_a[v] = median(pam[v]); M_b[v] = median(pbm[v]);
        std::printf("RESULT variant=%-8s ms=%.4f GFLOPS=%.3f cycles=%.0f instructions=%.0f IPC=%.3f packA_ms=%.4f packB_ms=%.4f GHz=%.3f\n",
                    names[v], M_ms[v], 2.0 * N * N * N / M_ms[v] / 1e6, M_c[v], median(instr[v]),
                    median(instr[v]) / M_c[v], M_a[v], M_b[v], M_c[v] / M_ms[v] / 1e6);
    }
    // 0 full(正式) 1 ref(同源码、防外提选项) 2 noload 3 nostore 4 pure 5 empty
    const double other_ms = M_ms[5] - M_a[5] - M_b[5];
    const double kernel_ms = M_ms[0] - M_a[0] - M_b[0] - other_ms;
    std::printf("E2E total_ms=%.3f packB_ms=%.3f packA_ms=%.3f kernel_ms=%.3f other_ms=%.3f\n",
                M_ms[0], M_b[0], M_a[0], kernel_ms, other_ms);
    std::printf("E2E_PCT packB=%.2f%% packA=%.2f%% kernel=%.2f%% other=%.2f%%\n",
                100 * M_b[0] / M_ms[0], 100 * M_a[0] / M_ms[0], 100 * kernel_ms / M_ms[0], 100 * other_ms / M_ms[0]);
    std::printf("PER_K full=%.3f ref=%.3f pure_fmla=%.3f load(ref-noload)=%.3f store(ref-nostore)=%.3f rest=%.3f (cycles per k-step; ideal 8.000)\n",
                (M_c[0] - M_c[5]) / ksteps, (M_c[1] - M_c[5]) / ksteps, (M_c[4] - M_c[5]) / ksteps,
                (M_c[1] - M_c[2]) / ksteps, (M_c[1] - M_c[3]) / ksteps,
                ((M_c[1] - M_c[5]) - (M_c[4] - M_c[5]) - (M_c[1] - M_c[2]) - (M_c[1] - M_c[3])) / ksteps);
    std::printf("CYCLES total=%.0f non_kernel=%.0f kernel=%.0f ideal=%.0f ideal_share=%.2f%%\n",
                M_c[0], M_c[5], M_c[0] - M_c[5], ksteps * 8, 100 * ksteps * 8 / M_c[0]);
    close(ins); close(fd);
    return 0;
}
