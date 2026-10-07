#include <arm_neon.h>
#include <stdint.h>
#include <stdio.h>

int main(void) {
    const int8_t a[16] = {
        1, 2, 3, 4, 5, 6, 7, 8,
        1, 1, 1, 1, 1, 1, 1, 1
    };
    const int8_t b_columns[16] = {
        1, 1, 1, 1, 1, 1, 1, 1,
        1, 2, 3, 4, 5, 6, 7, 8
    };
    int32_t c[4] = {0};

    int32x4_t result = vmmlaq_s32(vld1q_s32(c),
                                  vld1q_s8(a),
                                  vld1q_s8(b_columns));
    vst1q_s32(c, result);

    printf("%d %d\n%d %d\n", c[0], c[1], c[2], c[3]);
    return 0;
}
