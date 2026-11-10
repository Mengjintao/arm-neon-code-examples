// SPDX-License-Identifier: GPL-3.0-only
// From cpufb arm64/asm/_LDP_.S; add AAPCS64 callee-save preservation.
// Test-only code, not part of the GEMM implementation. See cpufb_LICENSE.
.text
.align 5

#void load_ldp_kernel(float*, int, int64_t);
#void load_ldr_kernel(float*, int64_t);
.global load_ldp_kernel
.type load_ldp_kernel, %function
load_ldp_kernel:
    stp d8, d9, [sp, #-64]!
    stp d10, d11, [sp, #16]
    stp d12, d13, [sp, #32]
    stp d14, d15, [sp, #48]
    mov x4, x2
    2:
    mov x3, x1
    mov x5, x0
    1:

    ldp q0, q1,[x5, #0]
    ldp q2, q3,[x5, #32]
    ldp q4, q5,[x5, #64]
    ldp q6, q7,[x5, #96]
    ldp q8, q9,[x5, #128]
    ldp q10, q11,[x5, #160]
    ldp q12, q13,[x5, #192]
    ldp q14, q15,[x5, #224]
    ldp q16, q17,[x5, #256]
    ldp q18, q19,[x5, #288]
    ldp q20, q21,[x5, #320]
    ldp q22, q23,[x5, #352]
    ldp q24, q25,[x5, #384]
    ldp q26, q27,[x5, #416]
    ldp q28, q29,[x5, #448]
    ldp q30, q31,[x5, #480]
    add x5, x5, #512

    subs x3, x3, #1
    bne 1b

    subs x4, x4, #1
    bne 2b
    ldp d10, d11, [sp, #16]
    ldp d12, d13, [sp, #32]
    ldp d14, d15, [sp, #48]
    ldp d8, d9, [sp], #64
    ret


.section .note.GNU-stack,"",%progbits
