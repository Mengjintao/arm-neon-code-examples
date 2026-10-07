#include <arm_neon.h>
#include <stdint.h>
#include <stdio.h>

static uint8x16_t add_u8(uint8x16_t a, uint8x16_t b) {
    uint8x16_t r;
    __asm__ volatile("add %0.16b, %1.16b, %2.16b" : "=w"(r) : "w"(a), "w"(b));
    return r;
}

static uint8x16_t sub_u8(uint8x16_t a, uint8x16_t b) {
    uint8x16_t r;
    __asm__ volatile("sub %0.16b, %1.16b, %2.16b" : "=w"(r) : "w"(a), "w"(b));
    return r;
}

static uint8x16_t mul_u8(uint8x16_t a, uint8x16_t b) {
    uint8x16_t r;
    __asm__ volatile("mul %0.16b, %1.16b, %2.16b" : "=w"(r) : "w"(a), "w"(b));
    return r;
}

static void show(const char *name, const uint8_t *v) {
    printf("%s:", name);
    for (int i = 0; i < 16; ++i) printf(" %3u", v[i]);
    puts("");
}

int main(void) {
    uint8_t a[16], b[16], sum[16], diff[16], product[16];
    for (int i = 0; i < 16; ++i) { a[i] = i; b[i] = i + 1; }
    uint8x16_t va = vld1q_u8(a), vb = vld1q_u8(b);
    vst1q_u8(sum, add_u8(va, vb));
    vst1q_u8(diff, sub_u8(va, vb));
    vst1q_u8(product, mul_u8(va, vb));
    show("a", a); show("b", b); show("add", sum);
    show("sub", diff); show("mul", product);
    return 0;
}