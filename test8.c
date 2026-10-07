#include <arm_neon.h>
#include <math.h>
#include <stdio.h>

static float dot_scalar(const float *a, const float *b, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) sum += a[i] * b[i];
    return sum;
}

static float dot_neon(const float *a, const float *b, int n) {
    float32x4_t sum = vdupq_n_f32(0.0f);
    int i = 0;
    for (; i + 4 <= n; i += 4)
        sum = vfmaq_f32(sum, vld1q_f32(a + i), vld1q_f32(b + i));

    float result = vaddvq_f32(sum);
    for (; i < n; ++i) result += a[i] * b[i];
    return result;
}

int main(void) {
    const float a[] = {1, 2, 3, 4, 5, 6, 7, 8};
    const float b[] = {8, 7, 6, 5, 4, 3, 2, 1};
    const int n = sizeof(a) / sizeof(a[0]);
    float scalar = dot_scalar(a, b, n);
    float vector = dot_neon(a, b, n);

    printf("Scalar: %.1f\nNEON:   %.1f\n", scalar, vector);
    return fabsf(scalar - vector) > 1e-5f;
}
