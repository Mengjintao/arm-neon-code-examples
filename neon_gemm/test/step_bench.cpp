// 教学阶梯：把 step/*.cpp 的计算函数原样 include 进来，用统一的测法计时。
// 编译时 -DSTEP_FILE=... -DSTEP_FN=...，step 自己的 main 改名后不用。
// 测法：1024³，预热 WARM 次，再计时 REPS 次取中位数；结果与 double 参考全矩阵比对。
#include <cstring>
#define main step_original_main
#include STEP_FILE
#undef main
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <vector>

static double now_s()
{
    timespec t{};
    clock_gettime(CLOCK_MONOTONIC_RAW, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

int main(int argc, char** argv)
{
    const int n = argc > 1 ? std::atoi(argv[1]) : 1024;
    const int warm = argc > 2 ? std::atoi(argv[2]) : 2;
    const int reps = argc > 3 ? std::atoi(argv[3]) : 5;
    float* A = (float*)aligned_alloc(64, (size_t)n * n * 4);
    float* B = (float*)aligned_alloc(64, (size_t)n * n * 4);
    float* C = (float*)aligned_alloc(64, (size_t)n * n * 4);
    std::srand(1);
    for (size_t i = 0; i < (size_t)n * n; ++i) A[i] = (float)std::rand() / RAND_MAX;
    for (size_t i = 0; i < (size_t)n * n; ++i) B[i] = (float)std::rand() / RAND_MAX;

    for (int w = 0; w < warm; ++w) STEP_FN(n, n, n, A, n, B, n, C, n);
    std::vector<double> t;
    for (int r = 0; r < reps; ++r) {
        const double b = now_s();
        STEP_FN(n, n, n, A, n, B, n, C, n);
        t.push_back(now_s() - b);
    }
    std::sort(t.begin(), t.end());

    // 参考：double 累加，i-k-j 顺序。
    std::vector<double> ref((size_t)n * n, 0.0);
    for (int i = 0; i < n; ++i)
        for (int k = 0; k < n; ++k) {
            const double a = A[(size_t)i * n + k];
            for (int j = 0; j < n; ++j) ref[(size_t)i * n + j] += a * B[(size_t)k * n + j];
        }
    double err = 0;
    bool ok = true;
    for (size_t i = 0; i < (size_t)n * n; ++i) {
        const double e = std::abs(C[i] - ref[i]);
        err = std::max(err, e);
        if (!(e <= 1e-4 * (1.0 + std::abs(ref[i])))) ok = false;
    }
    const double flop = 2.0 * n * n * n;
    std::printf("STEP %s n=%d warm=%d reps=%d median_ms=%.3f median_GFLOPS=%.3f best_GFLOPS=%.3f worst_GFLOPS=%.3f check=%s max_error=%.3g\n",
                STEP_NAME, n, warm, reps, t[t.size() / 2] * 1e3, flop / t[t.size() / 2] / 1e9,
                flop / t.front() / 1e9, flop / t.back() / 1e9, ok ? "PASS" : "FAIL", err);
    return ok ? 0 : 1;
}
