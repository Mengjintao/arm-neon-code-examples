#include <arm_neon.h>
#include <stdint.h>
#include <stdio.h>

int main(void) {
    const uint16_t input[8] = {0x1234, 0x5678, 0x9abc, 0xdef0, 0x1357, 0x2468, 0xbeef, 0xcafe};
    uint8_t output[8];
    uint8x8_t narrowed = vmovn_u16(vld1q_u16(input));
    vst1_u8(output, narrowed);
    for (int i = 0; i < 8; ++i) printf("0x%04x -> 0x%02x\n", input[i], output[i]);
    return 0;
}