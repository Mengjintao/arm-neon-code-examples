# 汇总：每个版本取所有样本里最快的一次（可以把几次 run.sh 的输出拼在一起传进来）。
import re, sys
t = open(sys.argv[1]).read()
best = {}
for n, g in re.findall(r'STEP ((?:L\d\d|X)_\w+) .*?best_GFLOPS=([\d.]+)', t):
    best[n] = max(best.get(n, 0), float(g))
stepped = set(best)      # 第 1–5 行和 offline pack 只用 step_bench 的结果
for n, g in re.findall(r'BEST variant=((?:L\d\d|X)_\w+) .*?best_GFLOPS=([\d.]+)', t):
    if n not in stepped:
        best[n] = max(best.get(n, 0), float(g))
assert 'FAIL' not in t and len(best) in (12, 14), best
prev = None
for n in sorted(k for k in best if k.startswith('L')):
    ch = '' if prev is None else f'{(best[n] / best[prev] - 1) * 100:+.1f}%'
    print(f'LADDER {n:14s} best_GFLOPS={best[n]:.3f} eff={best[n] / 41.6 * 100:.1f}% change={ch}')
    prev = n
for n in sorted(k for k in best if k.startswith('X')):
    print(f'SIDE {n:14s} best_GFLOPS={best[n]:.3f} eff={best[n] / 41.6 * 100:.1f}%')
