#include <arm_neon.h>
#include <stdint.h>
#include <stdio.h>
int main(void) {
    uint8_t src[16], lane_out[16], full_out[16];

    for (int i = 0; i < 16; ++i) src[i] = (uint8_t)(i + 1);

    uint8x16_t base = vdupq_n_u8(0xaa);
    uint8x16_t lane = vld1q_lane_u8(src + 7, base, 5);
    uint8x16_t full = vld1q_u8(src);

    vst1q_u8(lane_out, lane); 
    vst1q_u8(full_out, full);
    
    printf("vld1q_lane_u8: ");
    for (int i = 0; i < 16; ++i) printf("%02x ", lane_out[i]);
    printf("\nvld1q_u8:      ");
    for (int i = 0; i < 16; ++i) printf("%02x ", full_out[i]);
    putchar('\n');
    return 0;
}