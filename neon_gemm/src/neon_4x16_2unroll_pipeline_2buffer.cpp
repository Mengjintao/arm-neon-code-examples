#include "gemm.h"
#include <arm_neon.h>

// Two-step pipeline from GEMM-optimize/section/kernel_4x16.cpp.
// Two operand banks, two K per iteration; no software prefetch.
void neon_4x16(
    int k_len,
    const float *A_pack,
    const float *B_pack,
    float *C, int ldc){
    if (k_len <= 0) return;

    float32x4_t cv00, cv01, cv02, cv03;
    float32x4_t cv10, cv11, cv12, cv13;
    float32x4_t cv20, cv21, cv22, cv23;
    float32x4_t cv30, cv31, cv32, cv33;
    const float *ap = A_pack;
    const float *bp = B_pack;
    float *cp = C;

    cv00 = vld1q_f32(cp);  cv01 = vld1q_f32(cp + 4);  cv02 = vld1q_f32(cp + 8);  cv03 = vld1q_f32(cp + 12);
    cp += ldc;
    cv10 = vld1q_f32(cp);  cv11 = vld1q_f32(cp + 4);  cv12 = vld1q_f32(cp + 8);  cv13 = vld1q_f32(cp + 12);
    cp += ldc;
    cv20 = vld1q_f32(cp);  cv21 = vld1q_f32(cp + 4);  cv22 = vld1q_f32(cp + 8);  cv23 = vld1q_f32(cp + 12);
    cp += ldc;
    cv30 = vld1q_f32(cp);  cv31 = vld1q_f32(cp + 4);  cv32 = vld1q_f32(cp + 8);  cv33 = vld1q_f32(cp + 12);


    float32x4_t bv0, bv1, bv2, bv3;
    float32x4_t bv4, bv5, bv6, bv7;
    float32x4_t a_reg0, a_reg1;
    int kr = 0;

    a_reg0 = vld1q_f32(ap);
    bv0 = vld1q_f32(bp +  0);
    bv1 = vld1q_f32(bp +  4);
    bv2 = vld1q_f32(bp +  8);
    bv3 = vld1q_f32(bp + 12);

    // Two-K pipeline; final pair/odd K are handled below.
    for (; kr + 3 < k_len; kr += 2) {
        a_reg1 = vld1q_f32(ap + 4);
        bv4 = vld1q_f32(bp + 16);
        bv5 = vld1q_f32(bp + 20);
        cv00 = vfmaq_laneq_f32(cv00, bv0, a_reg0, 0);  cv10 = vfmaq_laneq_f32(cv10, bv0, a_reg0, 1);  cv20 = vfmaq_laneq_f32(cv20, bv0, a_reg0, 2);  cv30 = vfmaq_laneq_f32(cv30, bv0, a_reg0, 3);
        cv01 = vfmaq_laneq_f32(cv01, bv1, a_reg0, 0);  cv11 = vfmaq_laneq_f32(cv11, bv1, a_reg0, 1);  cv21 = vfmaq_laneq_f32(cv21, bv1, a_reg0, 2);  cv31 = vfmaq_laneq_f32(cv31, bv1, a_reg0, 3);
        bv6 = vld1q_f32(bp + 24);
        bv7 = vld1q_f32(bp + 28);
        cv02 = vfmaq_laneq_f32(cv02, bv2, a_reg0, 0);  cv12 = vfmaq_laneq_f32(cv12, bv2, a_reg0, 1);  cv22 = vfmaq_laneq_f32(cv22, bv2, a_reg0, 2);  cv32 = vfmaq_laneq_f32(cv32, bv2, a_reg0, 3);
        cv03 = vfmaq_laneq_f32(cv03, bv3, a_reg0, 0);  cv13 = vfmaq_laneq_f32(cv13, bv3, a_reg0, 1);  cv23 = vfmaq_laneq_f32(cv23, bv3, a_reg0, 2);  cv33 = vfmaq_laneq_f32(cv33, bv3, a_reg0, 3);
        

        bp += 32;  ap += 8;

        a_reg0 = vld1q_f32(ap);
        bv0 = vld1q_f32(bp +  0);
        bv1 = vld1q_f32(bp +  4);
        cv00 = vfmaq_laneq_f32(cv00, bv4, a_reg1, 0);  cv10 = vfmaq_laneq_f32(cv10, bv4, a_reg1, 1);  cv20 = vfmaq_laneq_f32(cv20, bv4, a_reg1, 2);  cv30 = vfmaq_laneq_f32(cv30, bv4, a_reg1, 3);
        cv01 = vfmaq_laneq_f32(cv01, bv5, a_reg1, 0);  cv11 = vfmaq_laneq_f32(cv11, bv5, a_reg1, 1);  cv21 = vfmaq_laneq_f32(cv21, bv5, a_reg1, 2);  cv31 = vfmaq_laneq_f32(cv31, bv5, a_reg1, 3);
        bv2 = vld1q_f32(bp +  8);
        bv3 = vld1q_f32(bp + 12);
        cv02 = vfmaq_laneq_f32(cv02, bv6, a_reg1, 0);  cv12 = vfmaq_laneq_f32(cv12, bv6, a_reg1, 1);  cv22 = vfmaq_laneq_f32(cv22, bv6, a_reg1, 2);  cv32 = vfmaq_laneq_f32(cv32, bv6, a_reg1, 3);
        cv03 = vfmaq_laneq_f32(cv03, bv7, a_reg1, 0);  cv13 = vfmaq_laneq_f32(cv13, bv7, a_reg1, 1);  cv23 = vfmaq_laneq_f32(cv23, bv7, a_reg1, 2);  cv33 = vfmaq_laneq_f32(cv33, bv7, a_reg1, 3);
    }

    // Final pair does not unconditionally load beyond K.
    if (kr + 1 < k_len) {
        a_reg1 = vld1q_f32(ap + 4);
        bv4 = vld1q_f32(bp + 16);
        bv5 = vld1q_f32(bp + 20);
        cv00 = vfmaq_laneq_f32(cv00, bv0, a_reg0, 0);  cv10 = vfmaq_laneq_f32(cv10, bv0, a_reg0, 1);  cv20 = vfmaq_laneq_f32(cv20, bv0, a_reg0, 2);  cv30 = vfmaq_laneq_f32(cv30, bv0, a_reg0, 3);
        cv01 = vfmaq_laneq_f32(cv01, bv1, a_reg0, 0);  cv11 = vfmaq_laneq_f32(cv11, bv1, a_reg0, 1);  cv21 = vfmaq_laneq_f32(cv21, bv1, a_reg0, 2);  cv31 = vfmaq_laneq_f32(cv31, bv1, a_reg0, 3);
        bv6 = vld1q_f32(bp + 24);
        bv7 = vld1q_f32(bp + 28);
        cv02 = vfmaq_laneq_f32(cv02, bv2, a_reg0, 0);  cv12 = vfmaq_laneq_f32(cv12, bv2, a_reg0, 1);  cv22 = vfmaq_laneq_f32(cv22, bv2, a_reg0, 2);  cv32 = vfmaq_laneq_f32(cv32, bv2, a_reg0, 3);
        cv03 = vfmaq_laneq_f32(cv03, bv3, a_reg0, 0);  cv13 = vfmaq_laneq_f32(cv13, bv3, a_reg0, 1);  cv23 = vfmaq_laneq_f32(cv23, bv3, a_reg0, 2);  cv33 = vfmaq_laneq_f32(cv33, bv3, a_reg0, 3);

        if (kr + 2 < k_len) {
            ap += 8; bp += 32;
            a_reg0 = vld1q_f32(ap);
            bv0 = vld1q_f32(bp + 0);
            bv1 = vld1q_f32(bp + 4);
            bv2 = vld1q_f32(bp + 8);
            bv3 = vld1q_f32(bp + 12);
        }
        cv00 = vfmaq_laneq_f32(cv00, bv4, a_reg1, 0);  cv10 = vfmaq_laneq_f32(cv10, bv4, a_reg1, 1);  cv20 = vfmaq_laneq_f32(cv20, bv4, a_reg1, 2);  cv30 = vfmaq_laneq_f32(cv30, bv4, a_reg1, 3);
        cv01 = vfmaq_laneq_f32(cv01, bv5, a_reg1, 0);  cv11 = vfmaq_laneq_f32(cv11, bv5, a_reg1, 1);  cv21 = vfmaq_laneq_f32(cv21, bv5, a_reg1, 2);  cv31 = vfmaq_laneq_f32(cv31, bv5, a_reg1, 3);
        cv02 = vfmaq_laneq_f32(cv02, bv6, a_reg1, 0);  cv12 = vfmaq_laneq_f32(cv12, bv6, a_reg1, 1);  cv22 = vfmaq_laneq_f32(cv22, bv6, a_reg1, 2);  cv32 = vfmaq_laneq_f32(cv32, bv6, a_reg1, 3);
        cv03 = vfmaq_laneq_f32(cv03, bv7, a_reg1, 0);  cv13 = vfmaq_laneq_f32(cv13, bv7, a_reg1, 1);  cv23 = vfmaq_laneq_f32(cv23, bv7, a_reg1, 2);  cv33 = vfmaq_laneq_f32(cv33, bv7, a_reg1, 3);
        kr += 2;
    }
    if (kr < k_len) {
        cv00 = vfmaq_laneq_f32(cv00, bv0, a_reg0, 0);  cv10 = vfmaq_laneq_f32(cv10, bv0, a_reg0, 1);  cv20 = vfmaq_laneq_f32(cv20, bv0, a_reg0, 2);  cv30 = vfmaq_laneq_f32(cv30, bv0, a_reg0, 3);
        cv01 = vfmaq_laneq_f32(cv01, bv1, a_reg0, 0);  cv11 = vfmaq_laneq_f32(cv11, bv1, a_reg0, 1);  cv21 = vfmaq_laneq_f32(cv21, bv1, a_reg0, 2);  cv31 = vfmaq_laneq_f32(cv31, bv1, a_reg0, 3);
        cv02 = vfmaq_laneq_f32(cv02, bv2, a_reg0, 0);  cv12 = vfmaq_laneq_f32(cv12, bv2, a_reg0, 1);  cv22 = vfmaq_laneq_f32(cv22, bv2, a_reg0, 2);  cv32 = vfmaq_laneq_f32(cv32, bv2, a_reg0, 3);
        cv03 = vfmaq_laneq_f32(cv03, bv3, a_reg0, 0);  cv13 = vfmaq_laneq_f32(cv13, bv3, a_reg0, 1);  cv23 = vfmaq_laneq_f32(cv23, bv3, a_reg0, 2);  cv33 = vfmaq_laneq_f32(cv33, bv3, a_reg0, 3);
    }

    cp = C;
    vst1q_f32(cp, cv00);  vst1q_f32(cp+4, cv01);  vst1q_f32(cp+8, cv02);  vst1q_f32(cp+12, cv03);
    cp += ldc;
    vst1q_f32(cp, cv10);  vst1q_f32(cp+4, cv11);  vst1q_f32(cp+8, cv12);  vst1q_f32(cp+12, cv13);
    cp += ldc;
    vst1q_f32(cp, cv20);  vst1q_f32(cp+4, cv21);  vst1q_f32(cp+8, cv22);  vst1q_f32(cp+12, cv23);
    cp += ldc;
    vst1q_f32(cp, cv30);  vst1q_f32(cp+4, cv31);  vst1q_f32(cp+8, cv32);  vst1q_f32(cp+12, cv33);
}
