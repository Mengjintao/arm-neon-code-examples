#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>
#include <algorithm>
#include <cmath>
#include <arm_neon.h>

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

void pack_A(int M, int K, float *A, int lda, float *A_packed) {
    for (int i = 0; i < M; i += 4)
        for (int k = 0; k < K; k++)
            for (int r = 0; r < 4; r++)
                A_packed[i * K + k * 4 + r] = A[(i + r) * lda + k];
}

void pack_B(int K, int N, float *B, int ldb, float *B_packed) {
    for (int k = 0; k < K; k++)
        for (int j = 0; j < N; j++)
            B_packed[k * N + j] = B[k * ldb + j];
}

void register_block(int M, int N, int K, float *A, int lda, float *B, int ldb, float *C, int ldc) {
    int i = 0;
    for (; i + 4 <= M; i += 4) {
        int j = 0;
        for (; j + 16 <= N; j += 16) {
            float32x4_t cv00, cv01, cv02, cv03;
            float32x4_t cv10, cv11, cv12, cv13;
            float32x4_t cv20, cv21, cv22, cv23;
            float32x4_t cv30, cv31, cv32, cv33;
            const float *ap = A + i * lda;
            const float *bp = B + j;
            float *cp = C + i * ldc + j;

            cv00 = vld1q_f32(cp);  cv01 = vld1q_f32(cp + 4);  cv02 = vld1q_f32(cp + 8);  cv03 = vld1q_f32(cp + 12);
            cp += ldc;
            cv10 = vld1q_f32(cp);  cv11 = vld1q_f32(cp + 4);  cv12 = vld1q_f32(cp + 8);  cv13 = vld1q_f32(cp + 12);
            cp += ldc;
            cv20 = vld1q_f32(cp);  cv21 = vld1q_f32(cp + 4);  cv22 = vld1q_f32(cp + 8);  cv23 = vld1q_f32(cp + 12);
            cp += ldc;
            cv30 = vld1q_f32(cp);  cv31 = vld1q_f32(cp + 4);  cv32 = vld1q_f32(cp + 8);  cv33 = vld1q_f32(cp + 12);

            float32x4_t av, bv0, bv1, bv2, bv3;
            for (int k = 0; k < K; k++) {
                av = vld1q_f32(ap);            // load A：打包后 a0–a3 连续，一条读进寄存器
                bv0 = vld1q_f32(bp + 0);
                bv1 = vld1q_f32(bp + 4);
                bv2 = vld1q_f32(bp + 8);
                bv3 = vld1q_f32(bp + 12);

                // dup：aᵢ 广播成 4 份，直接作为 FMLA 的参数
                cv00 = vfmaq_f32(cv00, bv0, vdupq_laneq_f32(av, 0));  cv10 = vfmaq_f32(cv10, bv0, vdupq_laneq_f32(av, 1));  cv20 = vfmaq_f32(cv20, bv0, vdupq_laneq_f32(av, 2));  cv30 = vfmaq_f32(cv30, bv0, vdupq_laneq_f32(av, 3));
                cv01 = vfmaq_f32(cv01, bv1, vdupq_laneq_f32(av, 0));  cv11 = vfmaq_f32(cv11, bv1, vdupq_laneq_f32(av, 1));  cv21 = vfmaq_f32(cv21, bv1, vdupq_laneq_f32(av, 2));  cv31 = vfmaq_f32(cv31, bv1, vdupq_laneq_f32(av, 3));
                cv02 = vfmaq_f32(cv02, bv2, vdupq_laneq_f32(av, 0));  cv12 = vfmaq_f32(cv12, bv2, vdupq_laneq_f32(av, 1));  cv22 = vfmaq_f32(cv22, bv2, vdupq_laneq_f32(av, 2));  cv32 = vfmaq_f32(cv32, bv2, vdupq_laneq_f32(av, 3));
                cv03 = vfmaq_f32(cv03, bv3, vdupq_laneq_f32(av, 0));  cv13 = vfmaq_f32(cv13, bv3, vdupq_laneq_f32(av, 1));  cv23 = vfmaq_f32(cv23, bv3, vdupq_laneq_f32(av, 2));  cv33 = vfmaq_f32(cv33, bv3, vdupq_laneq_f32(av, 3));

                ap += 4;
                bp += ldb;
            }

            cp = C + i * ldc + j;
            vst1q_f32(cp, cv00);  vst1q_f32(cp+4, cv01);  vst1q_f32(cp+8, cv02);  vst1q_f32(cp+12, cv03);
            cp += ldc;
            vst1q_f32(cp, cv10);  vst1q_f32(cp+4, cv11);  vst1q_f32(cp+8, cv12);  vst1q_f32(cp+12, cv13);
            cp += ldc;
            vst1q_f32(cp, cv20);  vst1q_f32(cp+4, cv21);  vst1q_f32(cp+8, cv22);  vst1q_f32(cp+12, cv23);
            cp += ldc;
            vst1q_f32(cp, cv30);  vst1q_f32(cp+4, cv31);  vst1q_f32(cp+8, cv32);  vst1q_f32(cp+12, cv33);
        }

        for (; j < N; j++)
            for (int r = 0; r < 4; r++)
                for (int k = 0; k < K; k++)
                    C[(i + r) * ldc + j] += A[i * lda + k * 4 + r] * B[k * ldb + j];
    }

    for (; i < M; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < K; k++)
                C[i * ldc + j] += A[(i / 4) * 4 * lda + k * 4 + i % 4] * B[k * ldb + j];
}

void cache_block(int M, int N, int K, float *A, int lda, float *B, int ldb, float *C, int ldc) {
    std::fill(C, C + M * N, 0.0f);

    for (int i = 0; i < M; i += Mc) {
        const int mc = Mc;
        for (int k = 0; k < K; k += Kc) {
            const int kc = Kc;
            float A_packed[Mc * Kc];
            pack_A(mc, kc, &A[i * lda + k], lda, A_packed);

            for (int j = 0; j < N; j += Nc) {
                const int nc = std::min(Nc, N - j);
                float B_packed[Kc * Nc];
                pack_B(kc, nc, &B[k * ldb + j], ldb, B_packed);
                register_block(mc, nc, kc, A_packed, kc, B_packed, nc,
                               &C[i * ldc + j], ldc);
            }
        }
    }
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

    float *A = (float *)std::malloc(M * K * sizeof(float));
    float *B = (float *)std::malloc(K * N * sizeof(float));
    float *C_naive = (float *)std::malloc(M * N * sizeof(float));
    float *C_opt = (float *)std::malloc(M * N * sizeof(float));

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
