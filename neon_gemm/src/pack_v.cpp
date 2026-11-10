#include "gemm.h"
#include <arm_neon.h>
#include <algorithm>
#include <cstddef>

// Four row vectors -> four column vectors; each packed k contains MR rows.
static inline void pack_a_4x4(const float* A, int lda, float* dst)
{
    const float32x4_t r0 = vld1q_f32(A);
    const float32x4_t r1 = vld1q_f32(A + lda);
    const float32x4_t r2 = vld1q_f32(A + 2LL * lda);
    const float32x4_t r3 = vld1q_f32(A + 3LL * lda);
    const float32x4x2_t t0 = vtrnq_f32(r0, r1);
    const float32x4x2_t t1 = vtrnq_f32(r2, r3);
    vst1q_f32(dst,      vcombine_f32(vget_low_f32(t0.val[0]), vget_low_f32(t1.val[0])));
    vst1q_f32(dst + 4,  vcombine_f32(vget_low_f32(t0.val[1]), vget_low_f32(t1.val[1])));
    vst1q_f32(dst + 8,  vcombine_f32(vget_high_f32(t0.val[0]), vget_high_f32(t1.val[0])));
    vst1q_f32(dst + 12, vcombine_f32(vget_high_f32(t0.val[1]), vget_high_f32(t1.val[1])));
}

static inline void pack_b_16(const float* B, float* dst)
{
    vst1q_f32(dst,      vld1q_f32(B));
    vst1q_f32(dst + 4,  vld1q_f32(B + 4));
    vst1q_f32(dst + 8,  vld1q_f32(B + 8));
    vst1q_f32(dst + 12, vld1q_f32(B + 12));
}

// Layout: [ceil(M/MR)][K][MR]. The final panel is zero-padded.
void pack_A(int M, int K, const float* A, int lda, float* packed)
{
    for (int i = 0; i < M; i += MR) {
        float* dst = packed + (size_t)i * K;
        const int rows = std::min(MR, M - i);
        int k = 0;
        if (rows == MR) {
            for (; k + 4 <= K; k += 4)
                pack_a_4x4(A + (size_t)(i) * lda + k, lda, dst + k * MR);
        }
        for (; k < K; ++k)
            for (int r = 0; r < MR; ++r)
                dst[k * MR + r] = r < rows ? A[(size_t)(i + r) * lda + k] : 0.0f;
    }
}

// Layout: [ceil(N/NR)][K][NR], NOT a single row-major K*N block.
void pack_B(int K, int N, const float* B, int ldb, float* packed)
{
    for (int j = 0; j < N; j += NR) {
        float* dst = packed + (size_t)j * K;
        const int cols = std::min(NR, N - j);
        for (int k = 0; k < K; ++k) {
            const float* src = B + (size_t)(k) * ldb + j;
            if (cols == NR) {
                pack_b_16(src, dst);
            } else {
                int c = 0;
                for (; c + 4 <= cols; c += 4)
                    vst1q_f32(dst + c, vld1q_f32(src + c));
                for (; c < NR; ++c)
                    dst[c] = c < cols ? src[c] : 0.0f;
            }
            dst += NR;
        }
    }
}
