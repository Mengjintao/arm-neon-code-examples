#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
exec > result.out 2> error.err
trap 'rm -rf .ablation' EXIT
cat test/ablation/experiments.txt
python3 test/ablation/prepare.py
flags='-O3 -mcpu=tsv110 -std=c++17'
echo "1024^3 CPU0 MC=88 NC=48 KC=256; $flags; 3 rounds x 15 samples median"
g++ --version | head -n 1
while read -r name pack; do
    packfile=src/pack_s.cpp; if [[ "$pack" == vector ]]; then packfile=src/pack_v_buffer.cpp; fi
    g++ $flags -fPIC -shared -Wl,-Bsymbolic .ablation/${name}_driver.cpp .ablation/${name}_kernel.cpp "$packfile" -o .ablation/${name}.so
done < .ablation/cases.txt
g++ $flags test/ablation/bench.cpp -ldl -o .ablation/bench
python3 test/ablation/codegen.py
taskset -c 0 .ablation/bench
python3 test/ablation/summary.py
