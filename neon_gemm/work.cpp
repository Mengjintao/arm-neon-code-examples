#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <vector>

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

void opt(int M, int N, int K, float *A, int lda, float *B, int ldb, float *C, int ldc) {
    

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
    naive(M, N, K, A, K, B, N, C_naive, N);
    const double elapsed = gettime() - begin;

    const double begin_opt = gettime();
    opt(M, N, K, A, K, B, N, C_opt, N);
    const double elapsed_opt = gettime() - begin_opt;

    int errors = 0;
    for (int i = 0; i < M * N; ++i) {
        if (!(std::fabs(C_opt[i] - C_naive[i]) <= 1e-3f * std::fabs(C_naive[i]))) {
            if (errors < 5)
                std::fprintf(stderr, "mismatch at (%d, %d): naive=%f opt=%f\n",
                             i / N, i % N, C_naive[i], C_opt[i]);
            ++errors;
        }
    }
    if (errors) {
        std::fprintf(stderr, "FAILED: %d mismatches\n", errors);
        return 1;
    }

    const double gflops_naive = 2.0 * M * N * K / elapsed / 1e9;
    const double gflops = 2.0 * M * N * K / elapsed_opt / 1e9;
    std::printf("M=%d N=%d K=%d naive GFLOPS=%.3f opt GFLOPS=%.3f\n", M, N, K, gflops_naive, gflops);

    return 0;
}

