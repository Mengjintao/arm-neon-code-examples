#include "gemm.h"
#include <arm_neon.h>
#include <algorithm>
#include <cstddef>

// A 布局：[微面板][K][4]，每个 K 连续存放四行元素。
void pack_A(int M, int K, const float *A, int lda, float *packed) {
    float32x4_t av00, av01, av02, av03;
    float32x4_t av10, av11, av12, av13;
    float32x4x2_t t0, t1;

    for (int i = 0; i < M; i += MR) {
        const float *src = A + (size_t)i * lda;
        float *dst = packed + (size_t)i * K;
        const int rows = std::min(MR, M - i);
        int k = 0;
        if (rows == MR) {
            if (K >= 8) {
                // 两套寄存器：分别保存相邻的两个 4x4 小块。
                av00 = vld1q_f32(src);
                av01 = vld1q_f32(src + lda);
                av02 = vld1q_f32(src + 2LL * lda);
                av03 = vld1q_f32(src + 3LL * lda);
                av10 = vld1q_f32(src + 4);
                av11 = vld1q_f32(src + 4 + lda);
                av12 = vld1q_f32(src + 4 + 2LL * lda);
                av13 = vld1q_f32(src + 4 + 3LL * lda);
                // 写出当前两块，并交错加载后续两块。
                for (; k + 16 <= K; k += 8) {
                    { //将当前4x4小块转置并写出，同时加载下一块4x4小块
                        float *dp = dst + k * MR;
                        const float *next = src + k + 8;
                        t0 = vtrnq_f32(av00, av01); //将两行中的元素交错排列 把同一列的元素先凑成一对
                        t1 = vtrnq_f32(av02, av03);
                        av00 = vld1q_f32(next);
                        //将交错排列的元素组合成列向量并写出
                        vst1q_f32(dp, vcombine_f32(vget_low_f32(t0.val[0]), vget_low_f32(t1.val[0])));
                        av01 = vld1q_f32(next + lda);
                        vst1q_f32(dp + 4, vcombine_f32(vget_low_f32(t0.val[1]), vget_low_f32(t1.val[1])));
                        av02 = vld1q_f32(next + 2LL * lda);
                        vst1q_f32(dp + 8, vcombine_f32(vget_high_f32(t0.val[0]), vget_high_f32(t1.val[0])));
                        av03 = vld1q_f32(next + 3LL * lda);
                        vst1q_f32(dp + 12, vcombine_f32(vget_high_f32(t0.val[1]), vget_high_f32(t1.val[1])));
                    }
                    /*
                    av00 = a00 a01 a02 a03    （第 0 行）
                    av01 = a10 a11 a12 a13    （第 1 行）
                    av02 = a20 a21 a22 a23    （第 2 行）
                    av03 = a30 a31 a32 a33    （第 3 行）
                    vtrnq_f32(x, y) 返回两个寄存器（float32x4x2_t）：
                    - .val[0] = {x0, y0, x2, y2}：两行的偶数通道交错
                    - .val[1] = {x1, y1, x3, y3}：两行的奇数通道交错

                    t0 = vtrnq_f32(av00, av01)
                    t0.val[0] = a00 a10 a02 a12
                    t0.val[1] = a01 a11 a03 a13

                    t1 = vtrnq_f32(av02, av03)
                    t1.val[0] = a20 a30 a22 a32
                    t1.val[1] = a21 a31 a23 a33

                    vget_low_f32(t0.val[0]) = a00 a10
                    vget_low_f32(t1.val[0]) = a20 a30
                    取低半部分并组合成列向量：
                    vcombine_f32(vget_low_f32(t0.val[0]), vget_low_f32(t1.val[0])) = a00 a10 a20 a30

                    */
                    {
                        float *dp = dst + (k + 4) * MR;
                        const float *next = src + k + 12;
                        t0 = vtrnq_f32(av10, av11);
                        t1 = vtrnq_f32(av12, av13);
                        av10 = vld1q_f32(next);
                        vst1q_f32(dp, vcombine_f32(vget_low_f32(t0.val[0]), vget_low_f32(t1.val[0])));
                        av11 = vld1q_f32(next + lda);
                        vst1q_f32(dp + 4, vcombine_f32(vget_low_f32(t0.val[1]), vget_low_f32(t1.val[1])));
                        av12 = vld1q_f32(next + 2LL * lda);
                        vst1q_f32(dp + 8, vcombine_f32(vget_high_f32(t0.val[0]), vget_high_f32(t1.val[0])));
                        av13 = vld1q_f32(next + 3LL * lda);
                        vst1q_f32(dp + 12, vcombine_f32(vget_high_f32(t0.val[1]), vget_high_f32(t1.val[1])));
                    }
                }
                // 收尾：只转置并写出最后两块，不再加载后续数据。
                {
                    float *dp = dst + k * MR;
                    t0 = vtrnq_f32(av00, av01);
                    t1 = vtrnq_f32(av02, av03);
                    vst1q_f32(dp, vcombine_f32(vget_low_f32(t0.val[0]), vget_low_f32(t1.val[0])));
                    vst1q_f32(dp + 4, vcombine_f32(vget_low_f32(t0.val[1]), vget_low_f32(t1.val[1])));
                    vst1q_f32(dp + 8, vcombine_f32(vget_high_f32(t0.val[0]), vget_high_f32(t1.val[0])));
                    vst1q_f32(dp + 12, vcombine_f32(vget_high_f32(t0.val[1]), vget_high_f32(t1.val[1])));
                }
                {
                    float *dp = dst + (k + 4) * MR;
                    t0 = vtrnq_f32(av10, av11);
                    t1 = vtrnq_f32(av12, av13);
                    vst1q_f32(dp, vcombine_f32(vget_low_f32(t0.val[0]), vget_low_f32(t1.val[0])));
                    vst1q_f32(dp + 4, vcombine_f32(vget_low_f32(t0.val[1]), vget_low_f32(t1.val[1])));
                    vst1q_f32(dp + 8, vcombine_f32(vget_high_f32(t0.val[0]), vget_high_f32(t1.val[0])));
                    vst1q_f32(dp + 12, vcombine_f32(vget_high_f32(t0.val[1]), vget_high_f32(t1.val[1])));
                }
                k += 8;
            }
            for (; k + 4 <= K; k += 4) {
                float *dp = dst + k * MR;
                av00 = vld1q_f32(src + k);
                av01 = vld1q_f32(src + k + lda);
                av02 = vld1q_f32(src + k + 2LL * lda);
                av03 = vld1q_f32(src + k + 3LL * lda);
                t0 = vtrnq_f32(av00, av01);
                t1 = vtrnq_f32(av02, av03);
                vst1q_f32(dp, vcombine_f32(vget_low_f32(t0.val[0]), vget_low_f32(t1.val[0])));
                vst1q_f32(dp + 4, vcombine_f32(vget_low_f32(t0.val[1]), vget_low_f32(t1.val[1])));
                vst1q_f32(dp + 8, vcombine_f32(vget_high_f32(t0.val[0]), vget_high_f32(t1.val[0])));
                vst1q_f32(dp + 12, vcombine_f32(vget_high_f32(t0.val[1]), vget_high_f32(t1.val[1])));
            }
        }
        for (; k < K; ++k)
            for (int r = 0; r < MR; ++r)
                dst[k * MR + r] = r < rows ? src[(size_t)r * lda + k] : 0.0f;
    }
}

// B 布局：[微面板][K][16]，每个 K 连续存放十六列元素。
void pack_B(int K, int N, const float *B, int ldb, float *packed) {
    float32x4_t bv00, bv01, bv02, bv03;
    float32x4_t bv10, bv11, bv12, bv13;

    for (int j = 0; j < N; j += NR) {
        const float *src = B + j;
        float *dst = packed + (size_t)j * K;
        const int cols = std::min(NR, N - j);
        int k = 0;
        if (cols == NR) {
            if (K >= 2) {
                // 两套寄存器：分别保存相邻两行的 16 个元素。
                bv00 = vld1q_f32(src);
                bv01 = vld1q_f32(src + 4);
                bv02 = vld1q_f32(src + 8);
                bv03 = vld1q_f32(src + 12);
                bv10 = vld1q_f32(src + ldb);
                bv11 = vld1q_f32(src + ldb + 4);
                bv12 = vld1q_f32(src + ldb + 8);
                bv13 = vld1q_f32(src + ldb + 12);
                // 写出当前两行，并交错加载后续两行。
                for (; k + 4 <= K; k += 2) {
                    {
                        float *dp = dst + k * NR;
                        const float *next = src + (size_t)(k + 2) * ldb;
                        vst1q_f32(dp, bv00);
                        bv00 = vld1q_f32(next);
                        vst1q_f32(dp + 4, bv01);
                        bv01 = vld1q_f32(next + 4);
                        vst1q_f32(dp + 8, bv02);
                        bv02 = vld1q_f32(next + 8);
                        vst1q_f32(dp + 12, bv03);
                        bv03 = vld1q_f32(next + 12);
                    }
                    {
                        float *dp = dst + (k + 1) * NR;
                        const float *next = src + (size_t)(k + 3) * ldb;
                        vst1q_f32(dp, bv10);
                        bv10 = vld1q_f32(next);
                        vst1q_f32(dp + 4, bv11);
                        bv11 = vld1q_f32(next + 4);
                        vst1q_f32(dp + 8, bv12);
                        bv12 = vld1q_f32(next + 8);
                        vst1q_f32(dp + 12, bv13);
                        bv13 = vld1q_f32(next + 12);
                    }
                }
                vst1q_f32(dst + k * NR + 0, bv00);
                vst1q_f32(dst + k * NR + 4, bv01);
                vst1q_f32(dst + k * NR + 8, bv02);
                vst1q_f32(dst + k * NR + 12, bv03);
                vst1q_f32(dst + (k + 1) * NR + 0, bv10);
                vst1q_f32(dst + (k + 1) * NR + 4, bv11);
                vst1q_f32(dst + (k + 1) * NR + 8, bv12);
                vst1q_f32(dst + (k + 1) * NR + 12, bv13);
                k += 2;
            }
            for (; k < K; ++k) {
                const float *sp = src + (size_t)k * ldb;
                float *dp = dst + k * NR;
                vst1q_f32(dp, vld1q_f32(sp + 0));
                vst1q_f32(dp + 4, vld1q_f32(sp + 4));
                vst1q_f32(dp + 8, vld1q_f32(sp + 8));
                vst1q_f32(dp + 12, vld1q_f32(sp + 12));
            }
        } else {
            for (; k < K; ++k) {
                int c = 0;
                for (; c + 4 <= cols; c += 4)
                    vst1q_f32(dst + k * NR + c, vld1q_f32(src + (size_t)k * ldb + c));
                for (; c < NR; ++c)
                    dst[k * NR + c] = c < cols ? src[(size_t)k * ldb + c] : 0.0f;
            }
        }
    }
}
