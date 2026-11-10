#!/usr/bin/env bash
# 表 2 加"分块参数""循环顺序"两步：test/ablation 的 12 个版本 + 4 个教学代码分块/顺序下的版本，
# 用 test/ablation/bench.cpp 原样一起测（同一进程、随机交错、3 轮 × 15 次）。test/ablation 不改。
set -euo pipefail
cd "$(dirname "$0")/../.."
exec > test/ablation2.out 2> test/ablation2.err
trap 'rm -rf .ablation' EXIT
rm -rf .ablation
python3 test/ablation/prepare.py
python3 test/ablation2/add_cases.py
flags='-O3 -mcpu=tsv110 -std=c++17'
echo "1024^3 CPU0 (taskset -c 0); $flags; 3 rounds x 15 samples median; g++ $(g++ -dumpfullversion)"
cat .ablation/cases.txt
while read -r name pack; do
    packfile=src/pack_s.cpp; if [[ "$pack" == vector ]]; then packfile=src/pack_v_buffer.cpp; fi
    g++ $flags -fPIC -shared -Wl,-Bsymbolic .ablation/${name}_driver.cpp .ablation/${name}_kernel.cpp "$packfile" -o .ablation/${name}.so
done < .ablation/cases.txt
g++ $flags test/ablation/bench.cpp -ldl -o .ablation/bench
taskset -c 0 .ablation/bench
python3 test/ablation2/summary.py test/ablation2.out
