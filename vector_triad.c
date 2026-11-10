#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#ifndef N
#define N 80000000
#endif
#ifndef NITER
#define NITER 20
#endif

int main(void) {
    _Static_assert(N > 0 && N <= SIZE_MAX / sizeof(double) / 4, "Invalid N");
    _Static_assert(NITER > 0, "NITER must be positive");
    double *a = malloc(4 * sizeof(double) * N);
    if (!a) {
        perror("malloc");
        return 1;
    }
    double *b = a + N, *c = b + N, *d = c + N;
    omp_set_dynamic(0);

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i)
        a[i] = b[i] = c[i] = d[i] = 1.0;

    int threads = 1;
    double start = omp_get_wtime();
#pragma omp parallel
    {
#pragma omp single
        threads = omp_get_num_threads();
        for (size_t j = 0; j < NITER; ++j) {
#pragma omp for simd schedule(static)
            for (size_t i = 0; i < N; ++i)
                a[i] = b[i] + c[i] * d[i];
            /* Preserve every iteration without disabling SIMD. */
            __asm__ volatile("" : : "r"(a), "r"(b), "r"(c), "r"(d) : "memory");
        }
    }
    double elapsed = omp_get_wtime() - start;
    size_t errors = 0;
#pragma omp parallel for reduction(+:errors) schedule(static)
    for (size_t i = 0; i < N; ++i)
        if (a[i] != 2.0 || b[i] != 1.0 || c[i] != 1.0 || d[i] != 1.0)
            ++errors;
    free(a);
    if (errors || elapsed <= 0.0) {
        fprintf(stderr, "Test failed: errors=%zu, time=%.9f; increase NITER if needed.\n",
                errors, elapsed);
        return 1;
    }
    double elements = (double)N * NITER;
    printf("N=%zu, NITER=%zu, threads=%d, time=%.6f s\n",
           (size_t)N, (size_t)NITER, threads, elapsed);
    printf("MFLOP/s=%.2f, logical GB/s=%.2f, validation=PASS\n",
           2 * elements / elapsed / 1e6, 32 * elements / elapsed / 1e9);
    return 0;
}
