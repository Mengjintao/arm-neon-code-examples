// 教学代码的循环顺序 Mc → Kc → Nc（ic → pc → jc → jr → ir），分块参数由编译选项给定：
//   -DBMC=.. -DBNC=.. -DBKC=..
// A 块每个 (ic, pc) 打包一次；B 块每个 (ic, pc, jc) 重新打包一次（顺序决定的代价）。
// 打包函数、微内核、C 清零和边界处理都和 src/main.cpp 相同；不做软件预取。
#include "../src/gemm.h"
#include <algorithm>

static void gemm_ipj(int M, int N, int K, const float* A, int lda,
                     const float* B, int ldb, float* C, int ldc,
                     float* packed_A, float* packed_B)
{
    for (int ic = 0; ic < M; ic += BMC) {
        const int mc = std::min(BMC, M - ic);
        for (int pc = 0; pc < K; pc += BKC) {
            const int kc = std::min(BKC, K - pc);
            pack_A(mc, kc, A + (size_t)ic * lda + pc, lda, packed_A);
            for (int jc = 0; jc < N; jc += BNC) {
                const int nc = std::min(BNC, N - jc);
                pack_B(kc, nc, B + (size_t)pc * ldb + jc, ldb, packed_B);
                for (int jr = 0; jr < nc; jr += NR) {
                    const int nr = std::min(NR, nc - jr);
                    const float* bp = packed_B + (size_t)jr * kc;
                    for (int ir = 0; ir < mc; ir += MR) {
                        const int mr = std::min(MR, mc - ir);
                        const float* ap = packed_A + ir * kc;
                        float* cp = C + (size_t)(ic + ir) * ldc + jc + jr;
                        if (mr == MR && nr == NR) {
                            if (pc == 0)
                                for (int r = 0; r < MR; ++r) std::fill_n(cp + (size_t)r * ldc, NR, 0.0f);
                            neon_4x16_next_AB(kc, ap, bp, cp, ldc, nullptr, nullptr);
                        } else {
                            alignas(64) float edge[MR * NR]{};
                            if (pc != 0)
                                for (int r = 0; r < mr; ++r) std::copy_n(cp + (size_t)r * ldc, nr, edge + r * NR);
                            neon_4x16_next_AB(kc, ap, bp, edge, NR, nullptr, nullptr);
                            for (int r = 0; r < mr; ++r) std::copy_n(edge + r * NR, nr, cp + (size_t)r * ldc);
                        }
                    }
                }
            }
        }
    }
}

extern "C" void run(int m, int n, int k, const float* a, int lda, const float* b, int ldb, float* c, int ldc,
                    float* pa, float* pb)
{
    gemm_ipj(m, n, k, a, lda, b, ldb, c, ldc, pa, pb);
}
