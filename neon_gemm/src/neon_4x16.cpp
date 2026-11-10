#include "gemm.h"
#include <arm_neon.h>

// C[4,16] += packed_A[K,4] * packed_B[K,16]. Caller initializes C.
void neon_4x16(int K, const float *packed_A, const float *packed_B,
               float *C, int ldc) {
    float32x4_t cv00, cv01, cv02, cv03;
    float32x4_t cv10, cv11, cv12, cv13;
    float32x4_t cv20, cv21, cv22, cv23;
    float32x4_t cv30, cv31, cv32, cv33;
    const float *ap = packed_A;
    const float *bp = packed_B;
    float *cp = C;

    cv00 = vld1q_f32(cp);  cv01 = vld1q_f32(cp + 4);  cv02 = vld1q_f32(cp + 8);  cv03 = vld1q_f32(cp + 12);
    cp += ldc;
    cv10 = vld1q_f32(cp);  cv11 = vld1q_f32(cp + 4);  cv12 = vld1q_f32(cp + 8);  cv13 = vld1q_f32(cp + 12);
    cp += ldc;
    cv20 = vld1q_f32(cp);  cv21 = vld1q_f32(cp + 4);  cv22 = vld1q_f32(cp + 8);  cv23 = vld1q_f32(cp + 12);
    cp += ldc;
    cv30 = vld1q_f32(cp);  cv31 = vld1q_f32(cp + 4);  cv32 = vld1q_f32(cp + 8);  cv33 = vld1q_f32(cp + 12);
    float32x4_t areg, bv0, bv1, bv2, bv3;
    int k = 0;
    for (; k + 1 <= K; k += 1) {
        areg = vld1q_f32(ap + 0 * MR);
        bv0 = vld1q_f32(bp + 0 * NR + 0);
        bv1 = vld1q_f32(bp + 0 * NR + 4);
        bv2 = vld1q_f32(bp + 0 * NR + 8);
        bv3 = vld1q_f32(bp + 0 * NR + 12);

        cv00 = vfmaq_laneq_f32(cv00, bv0, areg, 0);  cv10 = vfmaq_laneq_f32(cv10, bv0, areg, 1);  cv20 = vfmaq_laneq_f32(cv20, bv0, areg, 2);  cv30 = vfmaq_laneq_f32(cv30, bv0, areg, 3);
        cv01 = vfmaq_laneq_f32(cv01, bv1, areg, 0);  cv11 = vfmaq_laneq_f32(cv11, bv1, areg, 1);  cv21 = vfmaq_laneq_f32(cv21, bv1, areg, 2);  cv31 = vfmaq_laneq_f32(cv31, bv1, areg, 3);
        cv02 = vfmaq_laneq_f32(cv02, bv2, areg, 0);  cv12 = vfmaq_laneq_f32(cv12, bv2, areg, 1);  cv22 = vfmaq_laneq_f32(cv22, bv2, areg, 2);  cv32 = vfmaq_laneq_f32(cv32, bv2, areg, 3);
        cv03 = vfmaq_laneq_f32(cv03, bv3, areg, 0);  cv13 = vfmaq_laneq_f32(cv13, bv3, areg, 1);  cv23 = vfmaq_laneq_f32(cv23, bv3, areg, 2);  cv33 = vfmaq_laneq_f32(cv33, bv3, areg, 3);

        ap +=  MR;
        bp +=  NR;
    }

    for (; k < K; k++) {
        areg = vld1q_f32(ap);
        bv0 = vld1q_f32(bp + 0);
        bv1 = vld1q_f32(bp + 4);
        bv2 = vld1q_f32(bp + 8);
        bv3 = vld1q_f32(bp + 12);

        cv00 = vfmaq_laneq_f32(cv00, bv0, areg, 0);  cv10 = vfmaq_laneq_f32(cv10, bv0, areg, 1);  cv20 = vfmaq_laneq_f32(cv20, bv0, areg, 2);  cv30 = vfmaq_laneq_f32(cv30, bv0, areg, 3);
        cv01 = vfmaq_laneq_f32(cv01, bv1, areg, 0);  cv11 = vfmaq_laneq_f32(cv11, bv1, areg, 1);  cv21 = vfmaq_laneq_f32(cv21, bv1, areg, 2);  cv31 = vfmaq_laneq_f32(cv31, bv1, areg, 3);
        cv02 = vfmaq_laneq_f32(cv02, bv2, areg, 0);  cv12 = vfmaq_laneq_f32(cv12, bv2, areg, 1);  cv22 = vfmaq_laneq_f32(cv22, bv2, areg, 2);  cv32 = vfmaq_laneq_f32(cv32, bv2, areg, 3);
        cv03 = vfmaq_laneq_f32(cv03, bv3, areg, 0);  cv13 = vfmaq_laneq_f32(cv13, bv3, areg, 1);  cv23 = vfmaq_laneq_f32(cv23, bv3, areg, 2);  cv33 = vfmaq_laneq_f32(cv33, bv3, areg, 3);

        ap += MR;
        bp += NR;
    }

    cp = C;
    vst1q_f32(cp, cv00);  vst1q_f32(cp + 4, cv01);  vst1q_f32(cp + 8, cv02);  vst1q_f32(cp + 12, cv03);
    cp += ldc;
    vst1q_f32(cp, cv10);  vst1q_f32(cp + 4, cv11);  vst1q_f32(cp + 8, cv12);  vst1q_f32(cp + 12, cv13);
    cp += ldc;
    vst1q_f32(cp, cv20);  vst1q_f32(cp + 4, cv21);  vst1q_f32(cp + 8, cv22);  vst1q_f32(cp + 12, cv23);
    cp += ldc;
    vst1q_f32(cp, cv30);  vst1q_f32(cp + 4, cv31);  vst1q_f32(cp + 8, cv32);  vst1q_f32(cp + 12, cv33);
}
