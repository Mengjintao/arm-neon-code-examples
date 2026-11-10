// Reference: cpufb/arm64/asm/_ASIMD_.S, asimd_fmla_vs_f32f32f32.
// Lanes 0..3 alternate across chains, as in GEMM.
// 24 independent accumulators; four rounds per loop, no loads in the hot loop.
// Build from gemm_neon (flags prevent GCC from hoisting lane broadcasts):
// g++ -O2 -fno-tree-pre -fno-tree-fre -fno-gcse -fno-move-loop-invariants -fno-tree-loop-im
//     -std=c++17 -march=armv8-a+simd test/fmla_lane_ipc.cpp -o test/fmla_lane_ipc
// Run: ./test/fmla_lane_ipc 0 > result.out 2> error.err
// Check generated code after changing compiler/flags: 96 lane FMLAs per loop.
// FMLA/cycle = batch*iterations*96 / cycles; IPC = PMU instructions / cycles.
// 4x16 GEMM: effective FMLA/cycle = 16/(cycles/K); FLOPs/cycle = 128/(cycles/K).
#include <arm_neon.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <linux/perf_event.h>
#include <sched.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>

constexpr int CHAINS = 24;
constexpr int ROUNDS = 4;
constexpr int ITERATIONS = 200000;
constexpr int REPEATS = 15;
constexpr int BATCH = 20;

__attribute__((noinline))
void fmla_lane(int iterations, const float *seed, const float *a, const float *b, float *out)
{
    const float32x4_t av = vld1q_f32(a);
    const float32x4_t bv = vld1q_f32(b);
    float32x4_t c0 = vld1q_f32(seed + 0);
    float32x4_t c1 = vld1q_f32(seed + 4);
    float32x4_t c2 = vld1q_f32(seed + 8);
    float32x4_t c3 = vld1q_f32(seed + 12);
    float32x4_t c4 = vld1q_f32(seed + 16);
    float32x4_t c5 = vld1q_f32(seed + 20);
    float32x4_t c6 = vld1q_f32(seed + 24);
    float32x4_t c7 = vld1q_f32(seed + 28);
    float32x4_t c8 = vld1q_f32(seed + 32);
    float32x4_t c9 = vld1q_f32(seed + 36);
    float32x4_t c10 = vld1q_f32(seed + 40);
    float32x4_t c11 = vld1q_f32(seed + 44);
    float32x4_t c12 = vld1q_f32(seed + 48);
    float32x4_t c13 = vld1q_f32(seed + 52);
    float32x4_t c14 = vld1q_f32(seed + 56);
    float32x4_t c15 = vld1q_f32(seed + 60);
    float32x4_t c16 = vld1q_f32(seed + 64);
    float32x4_t c17 = vld1q_f32(seed + 68);
    float32x4_t c18 = vld1q_f32(seed + 72);
    float32x4_t c19 = vld1q_f32(seed + 76);
    float32x4_t c20 = vld1q_f32(seed + 80);
    float32x4_t c21 = vld1q_f32(seed + 84);
    float32x4_t c22 = vld1q_f32(seed + 88);
    float32x4_t c23 = vld1q_f32(seed + 92);

#define FMLA24() \
    c0 = vfmaq_laneq_f32(c0, bv, av, 0); \
    c1 = vfmaq_laneq_f32(c1, bv, av, 1); \
    c2 = vfmaq_laneq_f32(c2, bv, av, 2); \
    c3 = vfmaq_laneq_f32(c3, bv, av, 3); \
    c4 = vfmaq_laneq_f32(c4, bv, av, 0); \
    c5 = vfmaq_laneq_f32(c5, bv, av, 1); \
    c6 = vfmaq_laneq_f32(c6, bv, av, 2); \
    c7 = vfmaq_laneq_f32(c7, bv, av, 3); \
    c8 = vfmaq_laneq_f32(c8, bv, av, 0); \
    c9 = vfmaq_laneq_f32(c9, bv, av, 1); \
    c10 = vfmaq_laneq_f32(c10, bv, av, 2); \
    c11 = vfmaq_laneq_f32(c11, bv, av, 3); \
    c12 = vfmaq_laneq_f32(c12, bv, av, 0); \
    c13 = vfmaq_laneq_f32(c13, bv, av, 1); \
    c14 = vfmaq_laneq_f32(c14, bv, av, 2); \
    c15 = vfmaq_laneq_f32(c15, bv, av, 3); \
    c16 = vfmaq_laneq_f32(c16, bv, av, 0); \
    c17 = vfmaq_laneq_f32(c17, bv, av, 1); \
    c18 = vfmaq_laneq_f32(c18, bv, av, 2); \
    c19 = vfmaq_laneq_f32(c19, bv, av, 3); \
    c20 = vfmaq_laneq_f32(c20, bv, av, 0); \
    c21 = vfmaq_laneq_f32(c21, bv, av, 1); \
    c22 = vfmaq_laneq_f32(c22, bv, av, 2); \
    c23 = vfmaq_laneq_f32(c23, bv, av, 3);
    for (int i = 0; i < iterations; ++i) {
        FMLA24();
        FMLA24();
        FMLA24();
        FMLA24();
    }
#undef FMLA24
    vst1q_f32(out + 0, c0);
    vst1q_f32(out + 4, c1);
    vst1q_f32(out + 8, c2);
    vst1q_f32(out + 12, c3);
    vst1q_f32(out + 16, c4);
    vst1q_f32(out + 20, c5);
    vst1q_f32(out + 24, c6);
    vst1q_f32(out + 28, c7);
    vst1q_f32(out + 32, c8);
    vst1q_f32(out + 36, c9);
    vst1q_f32(out + 40, c10);
    vst1q_f32(out + 44, c11);
    vst1q_f32(out + 48, c12);
    vst1q_f32(out + 52, c13);
    vst1q_f32(out + 56, c14);
    vst1q_f32(out + 60, c15);
    vst1q_f32(out + 64, c16);
    vst1q_f32(out + 68, c17);
    vst1q_f32(out + 72, c18);
    vst1q_f32(out + 76, c19);
    vst1q_f32(out + 80, c20);
    vst1q_f32(out + 84, c21);
    vst1q_f32(out + 88, c22);
    vst1q_f32(out + 92, c23);
}

static int open_event(uint64_t config, int leader)
{
    perf_event_attr attr{};
    attr.size = sizeof(attr);
    attr.type = PERF_TYPE_HARDWARE;
    attr.config = config;
    attr.disabled = leader == -1;
    attr.pinned = leader == -1;
    attr.exclude_kernel = 1;
    attr.exclude_hv = 1;
    attr.read_format = PERF_FORMAT_GROUP | PERF_FORMAT_TOTAL_TIME_ENABLED | PERF_FORMAT_TOTAL_TIME_RUNNING;
    return syscall(SYS_perf_event_open, &attr, 0, -1, leader, 0);
}

struct Count {
    uint64_t nr, enabled, running, cycles, instructions;
};

int main(int argc, char **argv)
{
    int cpu = argc > 1 ? std::atoi(argv[1]) : 0;
    if (argc > 2 || cpu < 0 || cpu >= CPU_SETSIZE) return 1;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(cpu, &set);
    if (sched_setaffinity(0, sizeof(set), &set)) {
        std::perror("sched_setaffinity"); return 1;
    }
    alignas(64) float seed[CHAINS * 4], out[CHAINS * 4];
    alignas(16) float a[4] = {0.03125f, 0.0625f, 0.125f, 0.25f};
    alignas(16) float b[4] = {0.03125f, 0.0625f, 0.09375f, 0.125f};
    for (int i = 0; i < CHAINS * 4; ++i) seed[i] = i / 128.0f;
    fmla_lane(8, seed, a, b, out);
    for (int i = 0; i < CHAINS * 4; ++i) {
        if (out[i] != seed[i] + 8 * ROUNDS * a[(i / 4) % 4] * b[i % 4]) {
            std::fprintf(stderr, "Check failed at %d\n", i); return 1;
        }
    }
    int cycles_fd = open_event(PERF_COUNT_HW_CPU_CYCLES, -1);
    if (cycles_fd < 0) { std::perror("open CPU cycles"); return 1; }
    int instructions_fd = open_event(PERF_COUNT_HW_INSTRUCTIONS, cycles_fd);
    if (instructions_fd < 0) {
        std::perror("open instructions"); close(cycles_fd); return 1;
    }
    std::printf("Check=PASS CPU=%d chains=%d FMLA/loop=%d iterations=%d\n",
                cpu, CHAINS, CHAINS * ROUNDS, ITERATIONS);
    std::printf("PMU counts user-space cycles/instructions together; FMLA count comes from verified loop body.\n");
    const double fmlas = (double)BATCH * ITERATIONS * CHAINS * ROUNDS;
    Count samples[REPEATS]{};
    for (int repeat = 0; repeat < REPEATS; ++repeat) {
        fmla_lane(ITERATIONS, seed, a, b, out);
        if (ioctl(cycles_fd, PERF_EVENT_IOC_RESET, PERF_IOC_FLAG_GROUP) ||
            ioctl(cycles_fd, PERF_EVENT_IOC_ENABLE, PERF_IOC_FLAG_GROUP)) {
            std::perror("enable counters"); return 1;
        }
        for (int batch = 0; batch < BATCH; ++batch)
            fmla_lane(ITERATIONS, seed, a, b, out);
        if (ioctl(cycles_fd, PERF_EVENT_IOC_DISABLE, PERF_IOC_FLAG_GROUP)) {
            std::perror("disable counters"); return 1;
        }
        Count &c = samples[repeat];
        if (read(cycles_fd, &c, sizeof(c)) != sizeof(c) || c.nr != 2 ||
            !c.cycles || !c.running || c.running != c.enabled) {
            std::fprintf(stderr, "PMU group unavailable or not continuously scheduled.\n"); return 1;
        }
        for (int i = 0; i < CHAINS * 4; ++i) {
            if (out[i] != seed[i] + ITERATIONS * ROUNDS * a[(i / 4) % 4] * b[i % 4]) {
                std::fprintf(stderr, "Measured output check failed.\n"); return 1;
            }
        }
        std::printf("sample=%d cycles=%llu instructions=%llu FMLA/cycle=%.4f IPC=%.4f\n",
                    repeat + 1, (unsigned long long)c.cycles, (unsigned long long)c.instructions,
                    fmlas / c.cycles, (double)c.instructions / c.cycles);
    }
    std::sort(samples, samples + REPEATS, [](const Count &x, const Count &y) { return x.cycles < y.cycles; });
    const Count &c = samples[0];
    std::printf("SUMMARY best-cycle sample: FMLA/cycle=%.4f cycles/FMLA=%.4f IPC=%.4f FLOPs/cycle=%.4f\n",
                fmlas / c.cycles, c.cycles / fmlas, (double)c.instructions / c.cycles, 8 * fmlas / c.cycles);
    std::printf("FMLA/cycle range=%.6f..%.6f batches/sample=%d samples=%d\n", fmlas / samples[REPEATS-1].cycles, fmlas / c.cycles, BATCH, REPEATS);
    std::printf("Compute-only equivalent cycles/K for 16 FMLAs: %.4f\n", 16 * c.cycles / fmlas);
    std::printf("cycles/FMLA is reciprocal throughput, NOT dependency-chain latency.\n");
    close(instructions_fd);
    close(cycles_fd);
    return 0;
}
