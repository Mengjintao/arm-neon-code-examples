#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec > result.out 2> error.err
trap 'rm -f test/.mc_capacity' EXIT
g++ -O3 -fno-schedule-insns -std=c++17 test/mc_capacity_bench.cpp src/pack_v_buffer.cpp src/neon_4x16_2unroll_pipeline_2buffer.cpp src/neon_4x16_2unroll_pipeline_2buffer_prfm.cpp -o test/.mc_capacity
g++ --version | head -1
# Two independent processes, same binary/data, different shuffled execution order.
taskset -c 0 ./test/.mc_capacity 1
taskset -c 0 ./test/.mc_capacity 2
