#include <arm_neon.h>
#include <stdio.h>

static void print_matrix(const char *name, const double m[2][2]) {
    printf("%s\n", name);
    for (int i = 0; i < 2; ++i) printf("%.1f %.1f\n", m[i][0], m[i][1]);
}

int main(void) {
    const double matrix[2][2] = {{1, 2}, {3, 4}};
    double transposed[2][2];
    float64x2_t row0 = vld1q_f64(matrix[0]), row1 = vld1q_f64(matrix[1]);
    vst1q_f64(transposed[0], vzip1q_f64(row0, row1));
    vst1q_f64(transposed[1], vzip2q_f64(row0, row1));
    print_matrix("Before:", matrix);
    print_matrix("After:", transposed);
    return 0;
}