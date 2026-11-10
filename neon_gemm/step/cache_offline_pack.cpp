#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

static double gettime()
{
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}


void naive(int M, int N, int K, float *A, int lda, float *B, int ldb, float *C, int ldc) {
   
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++)
                sum += A[i * lda + k] * B[k * ldb + j];
            C[i * ldc + j] = sum;
        }
}

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

// 计算前一次打包整个 A、B：
// A 的块 (i, k) 放在 A_packed + i*K + k*mc，B 的块 (k, j) 放在 B_packed + k*N + j*kc
void cache_block(int M, int N, int K, float *A, int lda, float *B, int ldb, float *C, int ldc) {
    float *A_packed = (float *)malloc(M * K * sizeof(float));
    float *B_packed = (float *)malloc(K * N * sizeof(float));

    for (int i = 0; i < M; i += Mc) {
        const int mc = std::min(Mc, M - i);
        for (int k = 0; k < K; k += Kc) {
            const int kc = std::min(Kc, K - k);
            pack_A(mc, kc, &A[i * lda + k], lda, A_packed + i * K + k * mc);
        }
    }
    for (int k = 0; k < K; k += Kc) {
        const int kc = std::min(Kc, K - k);
        for (int j = 0; j < N; j += Nc) {
            const int nc = std::min(Nc, N - j);
            pack_B(kc, nc, &B[k * ldb + j], ldb, B_packed + k * N + j * kc);
        }
    }

    for (int i = 0; i < M; i++)
        memset(&C[i * ldc], 0, N * sizeof(float));

    for (int i = 0; i < M; i += Mc) {
        const int mc = std::min(Mc, M - i);
        for (int k = 0; k < K; k += Kc) {
            const int kc = std::min(Kc, K - k);
            const float *Ap = A_packed + i * K + k * mc;
            for (int j = 0; j < N; j += Nc) {
                const int nc = std::min(Nc, N - j);
                const float *Bp = B_packed + k * N + j * kc;
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

    free(A_packed);
    free(B_packed);
}

int main(int argc, char **argv)
{
    if (argc != 4) {
        std::fprintf(stderr, "usage: %s M N K\n", argv[0]);
        return 1;
    }

    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int K = atoi(argv[3]);

    float *A = (float *)aligned_alloc(64, M * K  * sizeof(float));
    float *B = (float *)aligned_alloc(64, K * N  * sizeof(float));
    float *C_naive  = (float *)aligned_alloc(64, M * N  * sizeof(float));
    float *C_opt = (float *)aligned_alloc(64, M * N  * sizeof(float));

    std::srand(time(NULL));
    for (int i = 0; i < M * K; i++) A[i] = (float)std::rand() / RAND_MAX;
    for (int i = 0; i < K * N; i++) B[i] = (float)std::rand() / RAND_MAX;

    const double begin = gettime();
    cache_block(M, N, K, A, K, B, N, C_naive, N);
    const double elapsed = gettime() - begin;

    naive(M, N, K, A, K, B, N, C_opt, N);
    float max_diff = 0.0f;
    for (int i = 0; i < M * N; i++)
        max_diff = std::max(max_diff, std::abs(C_naive[i] - C_opt[i]));

    const double gflops = 2.0 * M * N * K / elapsed / 1e9;
    std::printf("M=%d N=%d K=%d GFLOPS=%.3f\n", M, N, K, gflops);
    std::printf("Max difference: %.3f\n", max_diff);

    return max_diff > 1e-3f;
}

