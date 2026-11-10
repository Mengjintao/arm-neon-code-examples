#!/usr/bin/env bash
# 第 30 页消融实验：12 个版本按幻灯片顺序累加，取最快一次。全部绑 CPU0。
# 第 6–12 行：同一进程交错测（test/ablation3/bench.cpp）。
# 第 1–5 行：教学代码和第 4–8 页一样用 test/step_bench.cpp 测（计算函数和 main 同一文件，
#   编译器会内联；ikj 对此敏感，经函数指针调用只有约 6.5 GFLOPS），5 轮 × 每轮 7 次，取最快一次。
set -euo pipefail
cd "$(dirname "$0")/../.."
exec > test/ablation3.out 2> test/ablation3.err
trap 'rm -rf .ablation3' EXIT
rm -rf .ablation3
python3 test/ablation3/prepare.py
flags='-O3 -mcpu=tsv110 -std=c++17'
echo "1024^3 CPU0 (taskset -c 0); $flags; 5 rounds, best sample; g++ $(g++ -dumpfullversion)"
cat .ablation3/cases.txt
while read -r name reps rest; do
    srcs=${rest%%|*}; extra=; [[ "$rest" == *"|"* ]] && extra=${rest#*|}
    objs=
    for f in $srcs; do
        o=.ablation3/${name}_$(basename "$f" .cpp).o
        e=; [[ "$f" == src/pack_s.cpp ]] && e=$extra
        g++ $flags -fPIC -c $e "$f" -o "$o"
        objs="$objs $o"
    done
    g++ $flags -fPIC -shared -Wl,-Bsymbolic $objs -o .ablation3/$name.so
done < .ablation3/cases.txt
echo "CODEGEN scalar pack q-register accesses (expect 0): $(objdump -d .ablation3/L06_register_pack_s.o | grep -cE '\bq[0-9]+\b' || true)"
g++ $flags test/ablation3/bench.cpp -ldl -o .ablation3/bench
taskset -c 0 .ablation3/bench

# 第 1–5 行，和旁支 offline pack（第 8 页）
STEPS=(
  "L01_ijk step/naive.cpp naive 1 1"
  "L02_ikj step/ikj.cpp ikj 2 7"
  "L03_cache step/cache.cpp cache_block 2 7"
  "L04_pack step/cache_inline_pack.cpp cache_block 2 7"
  "L05_kmn test/ablation3/cache_pack_kmn.cpp cache_block 2 7"
  "X_offline_pack step/cache_offline_pack.cpp cache_block 2 7"
)
for s in "${STEPS[@]}"; do
  read -r name file fn warm reps <<< "$s"
  g++ $flags -DSTEP_FILE="\"$PWD/$file\"" -DSTEP_FN=$fn -DSTEP_NAME="\"$name\"" test/step_bench.cpp -o .ablation3/step_$name
done
for r in 1 2 3 4 5; do
  for s in $(printf '%s\n' "${STEPS[@]}" | awk '{print $1}' | shuf); do
    for t in "${STEPS[@]}"; do
      read -r name file fn warm reps <<< "$t"
      [[ $name == "$s" ]] && { printf 'ROUND %d ' "$r"; taskset -c 0 .ablation3/step_$name 1024 "$warm" "$reps"; }
    done
  done
done
python3 test/ablation3/summary.py test/ablation3.out
