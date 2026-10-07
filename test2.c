#include <arm_neon.h>
#include <stdint.h>
#include <stdio.h>

static void show(const char *name, const uint8_t *v) {
    printf("%s:", name);
    for (int i = 0; i < 16; ++i) printf(" %3u", v[i]);
    puts("");
}

int main(void) {
    uint8_t a[16], b[16], sum[16], diff[16], product[16];

    for (int i = 0; i < 16; ++i) { a[i] = i; b[i] = i+1; }

    uint8x16_t va = vld1q_u8(a), vb = vld1q_u8(b);
    uint8x16_t vsum = vaddq_u8(va, vb);
    uint8x16_t vdiff = vsubq_u8(va, vb);
    uint8x16_t vproduct = vmulq_u8(va, vb);
    vst1q_u8(sum, vsum); vst1q_u8(diff, vdiff); vst1q_u8(product, vproduct);

    show("a", a); show("b", b); show("vaddq_u8", sum);
    show("vsubq_u8", diff); show("vmulq_u8", product);
    return 0;
}