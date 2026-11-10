// 消融实验测量：dlopen .ablation3/<名字>.so 的 run()，1024³。
// 先全部和 double 参考比对（含不整除的尺寸），再测 ROUNDS 轮：每轮随机打乱版本顺序，
// 每个版本预热 1 次再计时 reps 次。报告所有样本里最快的一次（best），中位数一并给出。
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <dlfcn.h>
#include <fstream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

using Fn = void (*)(int, int, int, const float*, int, const float*, int, float*, int, float*, float*);
static double now_s()
{
    timespec t{};
    clock_gettime(CLOCK_MONOTONIC_RAW, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}

int main()
{
    const int ROUNDS = 5, N = 1024;
    std::ifstream list(".ablation3/cases.txt");
    std::vector<std::string> names;
    std::vector<int> reps;
    std::vector<Fn> fn;
    for (std::string line; std::getline(list, line);) {
        std::istringstream in(line);
        std::string name;
        int r;
        in >> name >> r;
        void* h = dlopen((".ablation3/" + name + ".so").c_str(), RTLD_NOW | RTLD_LOCAL);
        if (!h) { std::puts(dlerror()); return 1; }
        names.push_back(name); reps.push_back(r); fn.push_back((Fn)dlsym(h, "run"));
        if (!fn.back()) return 1;
    }
    auto alloc = [](size_t n) { return (float*)aligned_alloc(64, (n * 4 + 63) / 64 * 64); };
    float *a = alloc((size_t)N * N), *b = alloc((size_t)N * N), *c = alloc((size_t)N * N);
    float *pa = alloc(256 * 256), *pb = alloc(256 * (size_t)(N + 16));
    std::srand(1);
    for (int i = 0; i < N * N; ++i) { a[i] = (float)std::rand() / RAND_MAX - 0.5f; b[i] = (float)std::rand() / RAND_MAX - 0.5f; }

    for (auto d : {std::vector<int>{1024, 1024, 1024}, {91, 53, 257}, {5, 17, 3}}) {
        const int m = d[0], n = d[1], k = d[2];
        std::vector<double> ref((size_t)m * n, 0);
        for (int i = 0; i < m; ++i)
            for (int p = 0; p < k; ++p)
                for (int j = 0; j < n; ++j) ref[(size_t)i * n + j] += (double)a[i * k + p] * b[p * n + j];
        for (size_t v = 0; v < fn.size(); ++v) {
            std::fill_n(c, (size_t)m * n, NAN);
            fn[v](m, n, k, a, k, b, n, c, n, pa, pb);
            for (size_t i = 0; i < (size_t)m * n; ++i)
                if (!(std::abs(c[i] - ref[i]) <= 1e-4 * (1 + std::abs(ref[i])))) {
                    std::printf("FAIL %s %d %d %d idx=%zu\n", names[v].c_str(), m, n, k, i);
                    return 2;
                }
        }
        std::printf("CHECK PASS all variants %d %d %d\n", m, n, k);
        std::fflush(stdout);
    }

    std::vector<std::vector<double>> ms(fn.size());
    std::vector<int> order(fn.size());
    for (size_t v = 0; v < fn.size(); ++v) order[v] = v;
    std::mt19937 rng(2026);
    for (int round = 1; round <= ROUNDS; ++round) {
        std::shuffle(order.begin(), order.end(), rng);
        for (int v : order) {
            fn[v](N, N, N, a, N, b, N, c, N, pa, pb);
            for (int s = 0; s < reps[v]; ++s) {
                const double t = now_s();
                fn[v](N, N, N, a, N, b, N, c, N, pa, pb);
                ms[v].push_back((now_s() - t) * 1e3);
                std::printf("SAMPLE round=%d variant=%s ms=%.4f\n", round, names[v].c_str(), ms[v].back());
            }
            std::fflush(stdout);
        }
    }
    const double flop = 2.0 * N * N * N;
    for (size_t v = 0; v < fn.size(); ++v) {
        auto t = ms[v];
        std::sort(t.begin(), t.end());
        std::printf("BEST variant=%s samples=%zu best_ms=%.4f best_GFLOPS=%.4f median_GFLOPS=%.4f worst_GFLOPS=%.4f\n",
                    names[v].c_str(), t.size(), t.front(), flop / t.front() / 1e6, flop / t[t.size() / 2] / 1e6,
                    flop / t.back() / 1e6);
    }
}
