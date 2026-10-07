#include <arm_sme.h>
#include <stdio.h>

#define N 8

__arm_new("za")
static void matrix_multiply(const float a[N][N], const float b[N][N],
                            float c[N][N])
    __arm_streaming {
    svbool_t pg = svwhilelt_b32(0, N);

    svzero_za();
    for (int k = 0; k < N; ++k) {
        float a_column[N];
        for (int row = 0; row < N; ++row) a_column[row] = a[row][k];
        svmopa_za32_f32_m(0, pg, pg, svld1(pg, a_column), svld1(pg, b[k]));
    }

    for (int row = 0; row < N; ++row)
        svst1_hor_za32(0, row, pg, c[row]);
}

static void print_matrix(const char *name, const float m[N][N]) {
    puts(name);
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) printf("%5.0f", m[row][col]);
        putchar('\n');
    }
}

int main(void) {
    float a[N][N], b[N][N] = {0}, c[N][N] = {0};

    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {    
            a[row][col] = row * N + col + 1;
            b[row][col] = 1.0f;
        }
//        b[row][row] = 1.0f;
    }

    matrix_multiply(a, b, c);
    print_matrix("A:", a);
    print_matrix("B:", b);
    print_matrix("C = A x B:", c);
    return 0;
}
