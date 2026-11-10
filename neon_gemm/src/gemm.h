#pragma once

// Row-major FP32 GEMM: C = A * B (no transpose, alpha=1, beta=0).
// lda >= K, ldb >= N, ldc >= N. M,N >= 0, K > 0.
constexpr int MR = 4;
constexpr int NR = 16;
// Current main uses pc -> ic -> jc -> jr -> ir.
// A workspace: MC*KC; B workspace: KC*round_up(N, NR), not KC*NC.
constexpr int MC = 88;
constexpr int NC = 48;
constexpr int KC = 256;

void pack_A(int M, int K, const float* A, int lda, float* packed);
void pack_B(int K, int N, const float* B, int ldb, float* packed);
void neon_4x16(int K, const float* packed_A, const float* packed_B,
               float* C, int ldc);
// next_B is a valid K*NR-float panel; call on the last A micro-panel.
void neon_4x16_prefetch(int K, const float* packed_A, const float* packed_B,
                      float* C, int ldc, const float* next_B);
// Workspace: packed_A needs MC*KC floats.
// packed_B needs KC*((N+NR-1)/NR*NR) floats; NC must be a multiple of NR.
void gemm(int M, int N, int K, const float* A, int lda,
          const float* B, int ldb, float* C, int ldc,
          float* packed_A, float* packed_B);

// Each next pointer is nullptr or a valid packed panel of K*MR / K*NR floats.
void neon_4x16_next_AB(int K, const float* A, const float* B, float* C, int ldc,
                      const float* next_A, const float* next_B);
