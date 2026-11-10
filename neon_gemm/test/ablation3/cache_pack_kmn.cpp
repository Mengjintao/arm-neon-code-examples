// 消融第 5 步：在 step/cache_inline_pack.cpp 上只换循环顺序 Mc→Kc→Nc → Kc→Mc→Nc（第 10 页）。
// B 的 Kc×N 整片每个 pc 打包一次（按 Nc 块存，块内行距 nc）；A 块每个 (pc, ic) 打包一次。
// 分块 128³、块内 ikj、打包函数都和 cache_inline_pack.cpp 相同。
#include <algorithm>
#include <cstring>
#include <vector>

const int Mc = 128;
const int Nc = 128;
const int Kc = 128;

// 把 A 的一个 mc×kc 块拷到连续内存：行距从 lda 变成 kc
void pack_A(int mc, int kc, const float *A, int lda, float *A_packed) {
    for (int i = 0; i < mc; i++)
        for (int k = 0; k < kc; k++)
            A_packed[i * kc + k] = A[i * lda + k];
}

// 把 B 的一个 kc×nc 块拷到连续内存：行距从 ldb 变成 nc
void pack_B(int kc, int nc, const float *B, int ldb, float *B_packed) {
    for (int k = 0; k < kc; k++)
        for (int j = 0; j < nc; j++)
            B_packed[k * nc + j] = B[k * ldb + j];
}

void cache_block(int M, int N, int K, float *A, int lda, float *B, int ldb, float *C, int ldc) {
    for (int i = 0; i < M; i++)
        memset(&C[i * ldc], 0, N * sizeof(float));

    std::vector<float> B_packed((size_t)Kc * N);      // Kc×N 整片
    for (int k = 0; k < K; k += Kc) {
        const int kc = std::min(Kc, K - k);
        for (int j = 0; j < N; j += Nc)                // 每个 pc 只打包一次 B，被所有 A 块复用
            pack_B(kc, std::min(Nc, N - j), &B[k * ldb + j], ldb, &B_packed[(size_t)j * kc]);
        for (int i = 0; i < M; i += Mc) {
            const int mc = std::min(Mc, M - i);
            float A_packed[Mc * Kc];
            pack_A(mc, kc, &A[i * lda + k], lda, A_packed);   // 这个 A 块被 N 方向所有 B 块复用
            const float *Ap = A_packed;
            for (int j = 0; j < N; j += Nc) {
                const int nc = std::min(Nc, N - j);
                const float *Bp = &B_packed[(size_t)j * kc];
                for (int ii = 0; ii < mc; ii++) {
                    for (int kk = 0; kk < kc; kk++) {
                        float a_ik = Ap[ii * kc + kk];
                        for (int jj = 0; jj < nc; jj++)
                            C[(i + ii) * ldc + j + jj] += a_ik * Bp[kk * nc + jj];
                    }
                }
            }
        }
    }
}
