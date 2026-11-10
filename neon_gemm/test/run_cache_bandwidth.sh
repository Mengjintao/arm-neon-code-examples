#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
exec > result.out 2> error.err
trap 'rm -f test/.cache_bandwidth' EXIT
g++ -O3 -std=c++17 -fno-tree-loop-distribute-patterns test/cache_bandwidth.cpp test/cache_load_cpufb.S -o test/.cache_bandwidth
uname -a
g++ --version | head -1
for d in /sys/devices/system/cpu/cpu0/cache/index*; do
    echo "$d"
    cat "$d/level" "$d/type" "$d/size" "$d/coherency_line_size" "$d/shared_cpu_list"
done
./test/.cache_bandwidth "$@"
