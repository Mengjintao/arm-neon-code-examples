#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
exec > result.out 2> error.err
trap 'rm -rf .time_components' EXIT
python3 test/time_components/prepare.py
flags='-O3 -mcpu=tsv110 -std=c++17'
for v in {0..7}; do g++ $flags -c .time_components/driver$v.cpp -o .time_components/driver$v.o; done
for v in 6 7; do g++ $flags -c .time_components/kernel$v.cpp -o .time_components/kernel$v.o; done
g++ $flags test/time_components/bench.cpp src/pack_s.cpp src/neon_4x16_2unroll_pipeline_2buffer_prfm.cpp .time_components/*.o -o .time_components/bench
python3 test/time_components/codegen.py
for seed in 1 2 3; do taskset -c 0 .time_components/bench "$seed" "$(cat .time_components/variant_count)"; done
