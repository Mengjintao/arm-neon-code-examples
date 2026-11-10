#include "gemm.h"
#include <algorithm>
#include <cstddef>

// 标量打包：布局和 pack_v.cpp 相同，一次只搬一个 float。
// 编译时加 -fno-tree-vectorize，避免编译器自己把它变成 NEON。

// A 布局：[微面板][K][4]，每个 K 连续存放四行元素；不足 4 行的补 0。
void pack_A(int M, int K, const float *A, int lda, float *packed) {
    for (int i = 0; i < M; i += MR) {
        const float *src = A + (size_t)i * lda;
        float *dst = packed + (size_t)i * K;
        const int rows = std::min(MR, M - i);
        for (int k = 0; k < K; ++k)
            for (int r = 0; r < MR; ++r)
                dst[k * MR + r] = r < rows ? src[(size_t)r * lda + k] : 0.0f;
    }
}

// B 布局：[微面板][K][16]，每个 K 连续存放十六列元素；不足 16 列的补 0。
void pack_B(int K, int N, const float *B, int ldb, float *packed) {
    for (int j = 0; j < N; j += NR) {
        const float *src = B + j;
        float *dst = packed + (size_t)j * K;
        const int cols = std::min(NR, N - j);
        for (int k = 0; k < K; ++k)
            for (int c = 0; c < NR; ++c)
                dst[k * NR + c] = c < cols ? src[(size_t)k * ldb + c] : 0.0f;
    }
}
