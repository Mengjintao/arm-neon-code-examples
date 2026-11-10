#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>

static volatile float sink;

static double gettime()
{
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}


void naive(int M, int N, int K, float *A, int lda, float *B, int ldb, float *C, int ldc) {
    for (int i = 0; i < M; i++){
        memset(&C[i * ldc], 0, N * sizeof(float));
        for (int j = 0; j < N; j++) {
            for (int k = 0; k < K; k++)
                C[i * ldc + j] += A[i * lda + k] * B[k * ldb + j];
        }
    }
}
void ikj(int M, int N, int K, float *A, int lda, float *B, int ldb, float *C, int ldc) {
    for (int i = 0; i < M; i++){
        memset(&C[i * ldc], 0, N * sizeof(float));
        for (int k = 0; k < K; k++) {
            float a_ik = A[i * lda + k];
            for (int j = 0; j < N; j++)
                C[i * ldc + j] += a_ik * B[k * ldb + j];
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

    float *A = (float *)aligned_alloc(64, M * K  * sizeof(float));
    float *B = (float *)aligned_alloc(64, K * N  * sizeof(float));
    float *C_naive  = (float *)aligned_alloc(64, M * N  * sizeof(float));
    float *C_opt  = (float *)aligned_alloc(64, M * N  * sizeof(float));
    std::srand(time(NULL));
    for (int i = 0; i < M * K; i++) A[i] = (float)std::rand() / RAND_MAX;
    for (int i = 0; i < K * N; i++) B[i] = (float)std::rand() / RAND_MAX;

    const double begin = gettime();
    ikj(M, N, K, A, K, B, N, C_naive, N);
    const double elapsed = gettime() - begin;

    float max_diff = 0.0f;
    naive(M, N, K, A, K, B, N, C_opt, N);
    for (int i = 0; i < M * N; i++)
        max_diff = std::max(max_diff, std::abs(C_naive[i] - C_opt[i]));

    const double gflops = 2.0 * M * N * K / elapsed / 1e9;
    std::printf("M=%d N=%d K=%d GFLOPS=%.3f\n", M, N, K, gflops);
    std::printf("Max difference: %.3f\n", max_diff);

    return sink == 0.0f;
}

