// Build (AArch64): g++ -O3 -fno-schedule-insns -std=c++17 src/main.cpp src/pack_buffer.cpp src/neon_4x16_next_AB.cpp -o gemm
// Run: ./gemm M N K [lda ldb ldc] > result.out 2> error.err
#include "gemm.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>

static double gettime()
{
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + ts.tv_nsec * 1e-9;
}

// pc -> ic -> jc -> jr -> ir. Pack all N columns of B once per pc.
// Each packed A block is reused across all N columns; prefetch the next C tile.
void gemm(int M, int N, int K, const float* A, int lda,
          const float* B, int ldb, float* C, int ldc,
          float* packed_A, float* packed_B)
{
    for (int pc = 0; pc < K; pc += KC) {
        const int kc = std::min(KC, K - pc);
        pack_B(kc, N, B + (size_t)pc * ldb, ldb, packed_B);
        for (int ic = 0; ic < M; ic += MC) {
            const int mc = std::min(MC, M - ic);
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
                        //算出下一个tail的起始位置 预取C块
                        int next_ir = ir + MR;
                        int next_jr = jr;
                        if (next_ir >= mc) {
                            next_ir = 0; //回到顶行
                            next_jr += NR;  //下一块微面板
                        }
                        if (next_jr < nc) {
                            const int next_rows = std::min(MR, mc - next_ir);
                            for (int r = 0; r < next_rows; ++r) //一块C 4*16*4 四条cache line
                                __builtin_prefetch(C + (size_t)(ic + next_ir + r) * ldc
                                                   + jc + next_jr, 0, 3);
                        }
                        //下一块packed A/B 的指针。
                        const float* next_A = ir + MR < mc ? ap + MR * kc  
                            : (jc + jr + NR < N ? packed_A : nullptr); //已是最后一个行组，但右边还有 B 面板 
                        const float* next_B = ir + MR >= mc && jc + jr + NR < N
                            ? bp + NR * kc : nullptr;
                        if (mr == MR && nr == NR) {
                            if (pc == 0) {
                                for (int r = 0; r < MR; ++r)
                                    std::fill_n(cp + (size_t)r * ldc, NR, 0.0f);
                            }
                            neon_4x16_next_AB(kc, ap, bp, cp, ldc, next_A, next_B);
                        } else {
                            // 申请一块补齐到 4×16 的临时空间，算完再裁掉多余的部分。
                            alignas(64) float edge[MR * NR]{};
                            if (pc != 0) {
                                for (int r = 0; r < mr; ++r)
                                    std::copy_n(cp + (size_t)(r) * ldc, nr, edge + r * NR);
                            }
                            neon_4x16_next_AB(kc, ap, bp, edge, NR, next_A, next_B);
                            for (int r = 0; r < mr; ++r)
                                std::copy_n(edge + r * NR, nr, cp + (size_t)(r) * ldc);
                        }
                    }
                }
            }
        }
    }
}

int main(int argc, char** argv)
{
    if (argc != 4 && argc != 7) {
        std::fprintf(stderr, "Usage: %s M N K [lda ldb ldc]\n", argv[0]);
        return 1;
    }
    const int M = std::atoi(argv[1]);
    const int N = std::atoi(argv[2]);
    const int K = std::atoi(argv[3]);
    const int lda = argc == 7 ? std::atoi(argv[4]) : K;
    const int ldb = argc == 7 ? std::atoi(argv[5]) : N;
    const int ldc = argc == 7 ? std::atoi(argv[6]) : N;
    if (M <= 0 || N <= 0 || K <= 0 || lda < K || ldb < N || ldc < N) {
        std::fprintf(stderr, "Require M,N,K > 0; lda >= K, ldb >= N, ldc >= N.\n");
        return 1;
    }

    float* A = (float*)std::aligned_alloc(64, ((size_t)M * lda * sizeof(float) + 63) / 64 * 64);
    float* B = (float*)std::aligned_alloc(64, ((size_t)K * ldb * sizeof(float) + 63) / 64 * 64);
    // At N=ldc=1024, each 4x16 tile row occupies exactly one cache line.
    const size_t c_bytes = (size_t)M * ldc * sizeof(float);
    float* C = (float*)std::aligned_alloc(64, (c_bytes + 63) / 64 * 64);
    float* packed_A = (float*)std::aligned_alloc(64, MC * KC * sizeof(float));
    float* packed_B = (float*)std::aligned_alloc(64, (size_t)KC * (((size_t)N + NR - 1) / NR * NR) * sizeof(float));
    if (!A || !B || !C || !packed_A || !packed_B) {
        std::fprintf(stderr, "Memory allocation failed.\n");
        std::free(A);
        std::free(B);
        std::free(C);
        std::free(packed_A);
        std::free(packed_B);
        return 1;
    }
    std::srand(1);
    for (size_t i = 0; i < (size_t)M * lda; ++i)
        A[i] = (float)std::rand() / RAND_MAX;
    for (size_t i = 0; i < (size_t)K * ldb; ++i)
        B[i] = (float)std::rand() / RAND_MAX;

    // Warm up, then time 15 complete GEMMs; report the best time.
    for (int i = 0; i < 10; ++i)
        gemm(M, N, K, A, lda, B, ldb, C, ldc, packed_A, packed_B);
        
    double times[15];
    for (double& elapsed : times) {
        const double begin = gettime();
        gemm(M, N, K, A, lda, B, ldb, C, ldc, packed_A, packed_B);
        elapsed = gettime() - begin;
    }
    std::sort(times, times + 15);
    const double elapsed = times[0];

    // Reference check is outside the timed region.
    double max_error = 0.0;
    bool passed = true;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double reference = 0.0;
            for (int k = 0; k < K; ++k)
                reference += (double)A[(size_t)i * lda + k] * B[(size_t)k * ldb + j];
            const double error = std::abs(C[(size_t)i * ldc + j] - reference);
            max_error = std::max(max_error, error);
            if (!(error <= 1e-4 * (1.0 + std::abs(reference))))
                passed = false;
        }
    }
    std::printf("M=%d N=%d K=%d lda=%d ldb=%d ldc=%d\n", M, N, K, lda, ldb, ldc);
    std::printf("Best=%.6f s GFLOPS=%.3f Min=%.6f Max=%.6f Samples=15\n", elapsed, 2.0 * M * N * K / elapsed / 1e9, times[0], times[14]);
    std::printf("Check=%s Max_error=%.8g\n", passed ? "PASS" : "FAIL", max_error);

    std::free(A);
    std::free(B);
    std::free(C);
    std::free(packed_A);
    std::free(packed_B);
    return passed ? 0 : 1;
}
