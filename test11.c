#include <arm_neon.h>
#include <math.h>
#include <stdio.h>

static void gemm_neon(int m, int n, int k, float alpha, const float *a,
                      int lda, const float *b, int ldb, float beta, float *c,
                      int ldc) {
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col + 4 <= n; col += 4) {
            float32x4_t acc = beta == 0.0f
                                  ? vdupq_n_f32(0.0f)
                                  : vmulq_n_f32(vld1q_f32(c + row * ldc + col),
                                                beta);

            for (int p = 0; p < k; ++p) {
                float32x4_t b_values = vld1q_f32(b + p * ldb + col);
                float a_value = alpha * a[row * lda + p];
                acc = vfmaq_n_f32(acc, b_values, a_value);
            }
            vst1q_f32(c + row * ldc + col, acc);
        }

        for (int col = n & ~3; col < n; ++col) {
            float acc = beta == 0.0f ? 0.0f : beta * c[row * ldc + col];
            for (int p = 0; p < k; ++p)
                acc += alpha * a[row * lda + p] * b[p * ldb + col];
            c[row * ldc + col] = acc;
        }
    }
}

static void gemm_scalar(int m, int n, int k, float alpha, const float *a,
                        int lda, const float *b, int ldb, float beta, float *c,
                        int ldc) {
    for (int row = 0; row < m; ++row) {
        for (int col = 0; col < n; ++col) {
            float acc = beta == 0.0f ? 0.0f : beta * c[row * ldc + col];
            for (int p = 0; p < k; ++p)
                acc += alpha * a[row * lda + p] * b[p * ldb + col];
            c[row * ldc + col] = acc;
        }
    }
}

static void print_matrix(const char *name, const float *matrix, int rows,
                         int cols, int stride) {
    puts(name);
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col)
            printf("%8.2f", matrix[row * stride + col]);
        putchar('\n');
    }
}

int main(void) {
    enum { M = 3, N = 5, K = 4 };
    const float alpha = 1.25f;
    const float beta = 0.5f;
    const float a[M * K] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
    };
    const float b[K * N] = {
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20,
    };
    const float initial_c[M * N] = {
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15,
    };
    float c_neon[M * N];
    float c_reference[M * N];

    for (int i = 0; i < M * N; ++i) {
        c_neon[i] = initial_c[i];
        c_reference[i] = initial_c[i];
    }

    gemm_neon(M, N, K, alpha, a, K, b, N, beta, c_neon, N);
    gemm_scalar(M, N, K, alpha, a, K, b, N, beta, c_reference, N);

    float max_error = 0.0f;
    for (int i = 0; i < M * N; ++i) {
        float error = fabsf(c_neon[i] - c_reference[i]);
        if (error > max_error) max_error = error;
    }

    print_matrix("C = alpha * A * B + beta * C:", c_neon, M, N, N);
    printf("Maximum error vs scalar: %.6g\n", max_error);
    return max_error > 1e-4f;
}
