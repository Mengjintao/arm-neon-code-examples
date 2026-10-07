#include <arm_neon.h>
#include <stdio.h>

static void transpose4x4(const float src[4][4], float dst[4][4]) {
/**
r0 = 1  2  3  4
r1 = 5  6  7  8
r2 = 9 10 11 12
r3 =13 14 15 16
 */
    float32x4_t r0 = vld1q_f32(src[0]), r1 = vld1q_f32(src[1]);
    float32x4_t r2 = vld1q_f32(src[2]), r3 = vld1q_f32(src[3]);

/*
z01 = 1 5 2 6 3 7 4 8
z23 = 9 13 10 14 11 15 12 16
*/
    float32x4x2_t z01 = vzipq_f32(r0, r1);  
    float32x4x2_t z23 = vzipq_f32(r2, r3);

/*
dst[0] = 1  5  9 13
dst[1] = 2  6 10 14
dst[2] = 3  7 11 15
dst[3] = 4  8 12 16
*/
    vst1q_f32(dst[0], vcombine_f32(vget_low_f32(z01.val[0]), vget_low_f32(z23.val[0])));
    vst1q_f32(dst[1], vcombine_f32(vget_high_f32(z01.val[0]), vget_high_f32(z23.val[0])));
    vst1q_f32(dst[2], vcombine_f32(vget_low_f32(z01.val[1]), vget_low_f32(z23.val[1])));
    vst1q_f32(dst[3], vcombine_f32(vget_high_f32(z01.val[1]), vget_high_f32(z23.val[1])));
}

int main(void) {
    const float matrix[4][4] = {{1, 2, 3, 4}, {5, 6, 7, 8},
                                {9, 10, 11, 12}, {13, 14, 15, 16}};
    float transposed[4][4];
    transpose4x4(matrix, transposed);
    puts("Before:");
    for (int i = 0; i < 4; ++i) printf("%.0f %.0f %.0f %.0f\n", matrix[i][0], matrix[i][1], matrix[i][2], matrix[i][3]);
    puts("After:");
    for (int i = 0; i < 4; ++i) printf("%.0f %.0f %.0f %.0f\n", transposed[i][0], transposed[i][1], transposed[i][2], transposed[i][3]);
    return 0;
}